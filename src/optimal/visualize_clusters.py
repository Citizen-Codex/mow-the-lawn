import argparse
import json
import os
from html import escape
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.optimal.classify import (
    DEFAULT_MOTIF_LENGTH,
    build_embedding,
    cluster_distance_columns,
    fit_cluster_model,
    motif_counts,
    prepare_paths,
    relabel_clusters,
    top_end_positions,
    top_motifs,
)


DEFAULT_CLUSTERED_CSV = (
    Path(__file__).resolve().parent
    / "library"
    / "clusters"
    / "optimal_paths_clustered.csv"
)
DEFAULT_SUMMARY_JSON = (
    Path(__file__).resolve().parent / "library" / "clusters" / "summary.json"
)
DEFAULT_SVG_OUTPUT = (
    Path(__file__).resolve().parent / "library" / "clusters" / "cluster_projection.svg"
)
DEFAULT_HTML_OUTPUT = (
    Path(__file__).resolve().parent / "library" / "clusters" / "cluster_projection.html"
)
DEFAULT_CSV_OUTPUT = (
    Path(__file__).resolve().parent / "library" / "clusters" / "cluster_projection.csv"
)
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 920
DEFAULT_POINT_RADIUS = 4.5
DEFAULT_MAX_COMPONENTS = 8
DEFAULT_RANDOM_STATE = 42
DEFAULT_STRATEGY_LIMIT = 8
REQUIRED_COLUMNS = {
    "index",
    "seed",
    "open_cells",
    "start_row",
    "start_col",
    "end_row",
    "end_col",
    "moves",
    "overlaps",
    "path",
    "grid",
    "image",
}
CLUSTER_COLORS = (
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#d97706",
    "#7c3aed",
    "#0891b2",
    "#db2777",
    "#4f46e5",
    "#059669",
    "#b45309",
)


def load_clustered_paths(csv_path: str | Path) -> pd.DataFrame:
    file_path = Path(csv_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Clustered CSV not found: {file_path}")

    dataframe = pd.read_csv(file_path)
    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Clustered CSV is missing required columns: {missing}")
    if dataframe.empty:
        raise ValueError("Clustered CSV is empty")

    return dataframe.reset_index(drop=True)


def load_summary(summary_path: str | Path | None) -> dict[str, object]:
    if summary_path is None:
        raise ValueError("summary_json is required to render the strategy explorer")

    file_path = Path(summary_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Summary JSON not found: {file_path}")

    return json.loads(file_path.read_text(encoding="utf-8"))


def is_grouped_summary(summary: dict[str, object]) -> bool:
    return "sizes" in summary and "clustering_strategy" not in summary


def resolve_summary_entry_path(summary_path: Path, relative_path: str) -> Path:
    return (summary_path.parent / relative_path).resolve()


def size_projection_dir(root: Path, size: int) -> Path:
    return root / f"size_{size}"


def strategy_key(embedding: str, algorithm: str, n_clusters: int) -> str:
    return f"{embedding}__{algorithm}__k{n_clusters}"


def strategy_label(spec: dict[str, object]) -> str:
    return f"{spec['embedding']} + {spec['algorithm']} (k={int(spec['n_clusters'])})"


def strategy_specs(
    summary: dict[str, object], strategy_limit: int
) -> list[dict[str, object]]:
    selected = dict(summary.get("clustering_strategy", {}))
    if not selected:
        raise ValueError("Summary JSON has no selected clustering strategy")

    selected_spec = {
        "embedding": str(selected["embedding"]),
        "algorithm": str(selected["algorithm"]),
        "n_clusters": int(selected["selected_k"]),
        "silhouette": float(selected["silhouette"]),
        "davies_bouldin": float(selected["davies_bouldin"]),
        "selected": True,
    }
    specs = [selected_spec]

    seen = {
        strategy_key(
            selected_spec["embedding"],
            selected_spec["algorithm"],
            selected_spec["n_clusters"],
        )
    }
    for candidate in summary.get("strategy_candidates", []):
        key = strategy_key(
            str(candidate["embedding"]),
            str(candidate["algorithm"]),
            int(candidate["n_clusters"]),
        )
        if key in seen:
            continue
        seen.add(key)
        specs.append(
            {
                "embedding": str(candidate["embedding"]),
                "algorithm": str(candidate["algorithm"]),
                "n_clusters": int(candidate["n_clusters"]),
                "silhouette": float(candidate["silhouette"]),
                "davies_bouldin": float(candidate["davies_bouldin"]),
                "selected": False,
            }
        )
        if len(specs) >= strategy_limit:
            break

    return specs


def relative_href(from_dir: Path, target_path: Path) -> str:
    return Path(os.path.relpath(target_path, from_dir)).as_posix()


def axis_bounds(values: np.ndarray) -> tuple[float, float]:
    minimum = float(values.min())
    maximum = float(values.max())
    span = maximum - minimum
    padding = 1.0 if span == 0 else span * 0.08
    return minimum - padding, maximum + padding


def scale_point(
    x_value: float,
    y_value: float,
    *,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    plot_x0: float,
    plot_y0: float,
    plot_width: float,
    plot_height: float,
) -> tuple[float, float]:
    x_ratio = 0.5 if max_x == min_x else (x_value - min_x) / (max_x - min_x)
    y_ratio = 0.5 if max_y == min_y else (y_value - min_y) / (max_y - min_y)
    x_pos = plot_x0 + x_ratio * plot_width
    y_pos = plot_y0 + plot_height - (y_ratio * plot_height)
    return x_pos, y_pos


def reduce_to_2d(features: np.ndarray) -> np.ndarray:
    if features.shape[1] >= 2:
        if features.shape[1] == 2:
            return features
        return PCA(n_components=2).fit_transform(features)

    if features.shape[1] == 1:
        return np.hstack([features, np.zeros((len(features), 1), dtype=float)])

    raise ValueError("Feature matrix must have at least one column")


def build_base_points(
    dataframe: pd.DataFrame,
    *,
    html_output: Path | None,
    asset_root: Path,
) -> list[dict[str, object]]:
    html_parent = html_output.parent if html_output is not None else None
    base_points: list[dict[str, object]] = []

    for point_id, row in enumerate(dataframe.to_dict("records")):
        source = str(row.get("source", "optimal") or "optimal")
        user = "" if pd.isna(row.get("user")) else str(row.get("user", ""))
        config_value = row.get("config_id")
        config_id = None if pd.isna(config_value) else int(config_value)
        config_label = (
            "" if pd.isna(row.get("config_label")) else str(row.get("config_label", ""))
        )
        image_path = asset_root / Path(str(row["image"]))
        image_href = (
            relative_href(html_parent, image_path) if html_parent is not None else ""
        )
        base_points.append(
            {
                "id": point_id,
                "index": int(row["index"]),
                "source": source,
                "source_id": str(row.get("source_id", f"{source}-{int(row['index'])}")),
                "user": user,
                "config_id": config_id,
                "config_label": config_label,
                "size": int(row["size"]),
                "seed": int(row["seed"]),
                "moves": int(row["moves"]),
                "overlaps": int(row["overlaps"]),
                "open_cells": int(row["open_cells"]),
                "end_row": int(row["end_row"]),
                "end_col": int(row["end_col"]),
                "path": str(row["path"]),
                "image": str(row["image"]),
                "image_href": image_href,
            }
        )

    return base_points


def build_strategy_views(
    dataframe: pd.DataFrame,
    *,
    summary: dict[str, object],
    html_output: Path | None,
    asset_root: Path,
    width: int,
    height: int,
    max_components: int,
    motif_length: int,
    strategy_limit: int,
    random_state: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], str]:
    legend_width = 280
    margin_left = 80
    margin_top = 70
    margin_bottom = 70
    margin_right = 40
    plot_x0 = margin_left
    plot_y0 = margin_top
    plot_width = width - margin_left - margin_right - legend_width
    plot_height = height - margin_top - margin_bottom

    base_points = build_base_points(
        dataframe,
        html_output=html_output,
        asset_root=asset_root,
    )
    prepared = prepare_paths(dataframe)
    global_motif_counts = motif_counts(prepared.normalized_paths, motif_length)
    specs = strategy_specs(summary, strategy_limit)
    selected_key = strategy_key(
        str(summary["clustering_strategy"]["embedding"]),
        str(summary["clustering_strategy"]["algorithm"]),
        int(summary["clustering_strategy"]["selected_k"]),
    )

    embedding_cache: dict[str, object] = {}
    projection_cache: dict[str, np.ndarray] = {}
    strategy_views: list[dict[str, object]] = []

    for spec in specs:
        embedding_name = str(spec["embedding"])
        if embedding_name not in embedding_cache:
            embedding_cache[embedding_name] = build_embedding(
                dataframe,
                embedding_name=embedding_name,
                max_components=max_components,
                random_state=random_state,
                prepared=prepared,
            )
        embedding_result = embedding_cache[embedding_name]

        if embedding_name not in projection_cache:
            projection_cache[embedding_name] = reduce_to_2d(embedding_result.features)
        projection = projection_cache[embedding_name]

        labels, centers = fit_cluster_model(
            embedding_result.features,
            algorithm=str(spec["algorithm"]),
            n_clusters=int(spec["n_clusters"]),
            random_state=random_state,
        )
        labels, centers = relabel_clusters(labels, centers)
        distances, ranks = cluster_distance_columns(
            embedding_result.features, labels, centers
        )

        assigned = dataframe.copy()
        for column_name in prepared.derived_features.columns:
            assigned[column_name] = prepared.derived_features[column_name].to_numpy()
        assigned["normalized_path"] = prepared.normalized_paths
        assigned["cluster"] = labels
        assigned["cluster_distance"] = distances
        assigned["cluster_rank"] = ranks
        assigned["projection_x"] = projection[:, 0]
        assigned["projection_y"] = projection[:, 1]

        min_x, max_x = axis_bounds(projection[:, 0])
        min_y, max_y = axis_bounds(projection[:, 1])
        cluster_ids = sorted(int(cluster_id) for cluster_id in np.unique(labels))
        color_map = {
            cluster_id: CLUSTER_COLORS[index % len(CLUSTER_COLORS)]
            for index, cluster_id in enumerate(cluster_ids)
        }

        point_overlays: list[dict[str, object]] = []
        for point_id, row in enumerate(assigned.to_dict("records")):
            screen_x, screen_y = scale_point(
                float(row["projection_x"]),
                float(row["projection_y"]),
                min_x=min_x,
                max_x=max_x,
                min_y=min_y,
                max_y=max_y,
                plot_x0=plot_x0,
                plot_y0=plot_y0,
                plot_width=plot_width,
                plot_height=plot_height,
            )
            point_overlays.append(
                {
                    "id": point_id,
                    "cluster": int(row["cluster"]),
                    "cluster_distance": round(float(row["cluster_distance"]), 4),
                    "cluster_rank": int(row["cluster_rank"]),
                    "projection_x": round(float(row["projection_x"]), 5),
                    "projection_y": round(float(row["projection_y"]), 5),
                    "screen_x": round(float(screen_x), 2),
                    "screen_y": round(float(screen_y), 2),
                    "color": color_map[int(row["cluster"])],
                }
            )

        cluster_views: list[dict[str, object]] = []
        for cluster_id in cluster_ids:
            subset = assigned[assigned["cluster"] == cluster_id].copy()
            source_series = (
                subset["source"]
                if "source" in subset.columns
                else pd.Series(["optimal"] * len(subset))
            )
            source_counts = {
                str(source): int(count)
                for source, count in source_series.fillna("optimal")
                .value_counts()
                .items()
            }
            motifs = top_motifs(
                subset["normalized_path"].tolist(),
                global_counts=global_motif_counts,
                motif_length=motif_length,
                min_motif_count=max(8, len(subset) // 20),
            )
            end_positions = top_end_positions(subset)
            lead_motif = motifs[0]["motif"] if motifs else ""
            lead_end = (
                end_positions[0] if end_positions else {"end_row": -1, "end_col": -1}
            )
            centroid_x, centroid_y = scale_point(
                float(subset["projection_x"].mean()),
                float(subset["projection_y"].mean()),
                min_x=min_x,
                max_x=max_x,
                min_y=min_y,
                max_y=max_y,
                plot_x0=plot_x0,
                plot_y0=plot_y0,
                plot_width=plot_width,
                plot_height=plot_height,
            )
            cluster_views.append(
                {
                    "cluster_id": cluster_id,
                    "color": color_map[cluster_id],
                    "size": int(len(subset)),
                    "hint": (
                        f"motif={lead_motif}, end=({lead_end['end_row']},{lead_end['end_col']})"
                    ),
                    "avg_moves": round(float(subset["moves"].mean()), 2),
                    "avg_overlaps": round(float(subset["overlaps"].mean()), 2),
                    "avg_turn_rate": round(float(subset["turn_rate"].mean()), 4),
                    "source_counts": source_counts,
                    "centroid_x": round(float(centroid_x), 2),
                    "centroid_y": round(float(centroid_y), 2),
                    "top_motifs": motifs[:3],
                    "top_end_positions": end_positions[:3],
                }
            )

        strategy_views.append(
            {
                "key": strategy_key(
                    embedding_name,
                    str(spec["algorithm"]),
                    int(spec["n_clusters"]),
                ),
                "label": strategy_label(spec),
                "embedding": embedding_name,
                "algorithm": str(spec["algorithm"]),
                "n_clusters": int(spec["n_clusters"]),
                "silhouette": round(float(spec["silhouette"]), 4),
                "davies_bouldin": round(float(spec["davies_bouldin"]), 4),
                "selected": bool(spec["selected"]),
                "feature_type": str(
                    embedding_result.metadata.get("type", embedding_name)
                ),
                "svd_components": embedding_result.metadata.get("svd_components"),
                "clusters": cluster_views,
                "points": point_overlays,
            }
        )

    return base_points, strategy_views, selected_key


def render_svg(
    base_points: list[dict[str, object]],
    selected_view: dict[str, object],
    *,
    width: int,
    height: int,
    point_radius: float,
) -> str:
    legend_width = 280
    margin_left = 80
    margin_top = 70
    margin_bottom = 70
    margin_right = 40
    plot_x0 = margin_left
    plot_y0 = margin_top
    plot_width = width - margin_left - margin_right - legend_width
    plot_height = height - margin_top - margin_bottom

    base_lookup = {int(point["id"]): point for point in base_points}
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#f8fafc" />',
        f'<text x="{margin_left}" y="32" fill="#0f172a" font-size="24" font-weight="700">Optimal Path Cluster Explorer</text>',
        f'<text x="{margin_left}" y="54" fill="#475569" font-size="13">Projection for {escape(str(selected_view["label"]))}; open the HTML explorer to switch strategies interactively.</text>',
        f'<rect x="{plot_x0}" y="{plot_y0}" width="{plot_width}" height="{plot_height}" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="12" />',
        f'<line x1="{plot_x0}" y1="{plot_y0 + plot_height / 2}" x2="{plot_x0 + plot_width}" y2="{plot_y0 + plot_height / 2}" stroke="#e2e8f0" stroke-width="1" />',
        f'<line x1="{plot_x0 + plot_width / 2}" y1="{plot_y0}" x2="{plot_x0 + plot_width / 2}" y2="{plot_y0 + plot_height}" stroke="#e2e8f0" stroke-width="1" />',
        f'<text x="{plot_x0 + plot_width / 2}" y="{height - 18}" text-anchor="middle" fill="#475569" font-size="12">Projection axis 0</text>',
        f'<text x="20" y="{plot_y0 + plot_height / 2}" text-anchor="middle" fill="#475569" font-size="12" transform="rotate(-90 20 {plot_y0 + plot_height / 2})">Projection axis 1</text>',
    ]

    for overlay in selected_view["points"]:
        point = base_lookup[int(overlay["id"])]
        is_human = str(point.get("source", "optimal")) == "human"
        point_label = str(point.get("source_id", f"optimal-{point['index']}"))
        user_suffix = (
            f" user={point['user']} config={point['config_label']}"
            if is_human and point.get("user")
            else ""
        )
        tooltip = escape(
            (
                f"record={point_label} source={point['source']} seed={point['seed']} "
                f"cluster={overlay['cluster']} "
                f"moves={point['moves']} overlaps={point['overlaps']} "
                f"path={str(point['path'])[:48]}{user_suffix}"
            )
        )
        radius = point_radius + 1.6 if is_human else point_radius
        stroke = "#0f172a" if is_human else "#ffffff"
        stroke_width = 1.5 if is_human else 0.9
        opacity = 0.96 if is_human else 0.78
        parts.append(
            f'<circle cx="{overlay["screen_x"]:.2f}" cy="{overlay["screen_y"]:.2f}" r="{radius}" fill="{overlay["color"]}" fill-opacity="{opacity}" stroke="{stroke}" stroke-width="{stroke_width}"><title>{tooltip}</title></circle>'
        )

    legend_x = plot_x0 + plot_width + 24
    legend_y = plot_y0 + 10
    parts.append(
        f'<text x="{legend_x}" y="{legend_y}" fill="#0f172a" font-size="16" font-weight="700">{escape(str(selected_view["label"]))}</text>'
    )
    parts.append(
        f'<text x="{legend_x}" y="{legend_y + 20}" fill="#475569" font-size="12">silhouette={selected_view["silhouette"]:.4f} | davies-bouldin={selected_view["davies_bouldin"]:.4f}</text>'
    )

    for index, cluster in enumerate(selected_view["clusters"]):
        parts.append(
            f'<circle cx="{cluster["centroid_x"]:.2f}" cy="{cluster["centroid_y"]:.2f}" r="9" fill="#ffffff" stroke="{cluster["color"]}" stroke-width="3" />'
        )
        parts.append(
            f'<text x="{cluster["centroid_x"]:.2f}" y="{cluster["centroid_y"] + 4:.2f}" text-anchor="middle" fill="#0f172a" font-size="10" font-weight="700">{cluster["cluster_id"]}</text>'
        )

        entry_y = legend_y + 56 + (index * 64)
        parts.append(
            f'<circle cx="{legend_x + 8}" cy="{entry_y - 5}" r="7" fill="{cluster["color"]}" />'
        )
        parts.append(
            f'<text x="{legend_x + 24}" y="{entry_y}" fill="#0f172a" font-size="13" font-weight="700">Cluster {cluster["cluster_id"]} ({cluster["size"]})</text>'
        )
        parts.append(
            f'<text x="{legend_x + 24}" y="{entry_y + 18}" fill="#475569" font-size="12">{escape(str(cluster["hint"]))}</text>'
        )
        parts.append(
            f'<text x="{legend_x + 24}" y="{entry_y + 36}" fill="#64748b" font-size="11">moves={cluster["avg_moves"]:.2f} overlaps={cluster["avg_overlaps"]:.2f}</text>'
        )

    parts.append(
        f'<text x="{legend_x}" y="{height - 24}" fill="#64748b" font-size="11">Use cluster_projection.html for strategy switching and point inspection.</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def render_html(
    payload: dict[str, object],
    *,
    width: int,
    height: int,
) -> str:
    plot_x0 = 80
    plot_y0 = 70
    plot_width = width - 80 - 40 - 280
    plot_height = height - 70 - 70
    template = """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Optimal Path Cluster Explorer</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      --bg: #f8fafc;
      --panel: #ffffff;
      --panel-border: #cbd5e1;
      --muted: #475569;
      --muted-soft: #64748b;
      --text: #0f172a;
      --accent: #0f766e;
      --accent-soft: #ccfbf1;
      --selected: #0f766e;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); }
    .app { display: grid; grid-template-columns: 380px 1fr; min-height: 100vh; }
    .sidebar { padding: 20px; border-right: 1px solid #e2e8f0; background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%); overflow-y: auto; }
    .main { padding: 20px; }
    h1 { margin: 0 0 8px; font-size: 24px; line-height: 1.2; }
    .lede { margin: 0 0 20px; color: var(--muted); font-size: 14px; line-height: 1.5; }
    .section { margin-bottom: 20px; }
    .section-title { margin: 0 0 10px; font-size: 12px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted-soft); }
    .search-row { display: grid; grid-template-columns: 1fr auto auto; gap: 8px; }
    input[type=\"search\"] { width: 100%; padding: 10px 12px; border: 1px solid var(--panel-border); border-radius: 10px; font: inherit; background: var(--panel); }
    button { padding: 10px 12px; border: 1px solid var(--panel-border); border-radius: 10px; background: var(--panel); font: inherit; color: var(--text); cursor: pointer; }
    button:hover { border-color: #94a3b8; }
    .toolbar { display: flex; gap: 8px; margin-top: 10px; }
    .strategy-list, .cluster-list { display: grid; gap: 10px; }
    .strategy-card, .cluster-card { padding: 12px; border: 1px solid var(--panel-border); border-radius: 14px; background: var(--panel); }
    .strategy-card.active { border-color: var(--selected); box-shadow: 0 0 0 2px rgba(15, 118, 110, 0.12); }
    .strategy-top, .cluster-top { display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: start; margin-bottom: 8px; }
    .strategy-name, .cluster-name { font-weight: 700; font-size: 14px; }
    .strategy-meta, .cluster-meta { color: var(--muted); font-size: 12px; line-height: 1.5; }
    .strategy-tags, .cluster-actions { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px; }
    .tag { display: inline-flex; align-items: center; padding: 4px 8px; border-radius: 999px; background: #eef2ff; color: #3730a3; font-size: 11px; font-weight: 600; }
    .tag.selected { background: var(--accent-soft); color: var(--selected); }
    .cluster-card { padding-bottom: 10px; }
    .cluster-top { grid-template-columns: auto 1fr auto; align-items: center; }
    .cluster-color { width: 12px; height: 12px; border-radius: 999px; }
    .cluster-actions button, .strategy-card button { padding: 6px 10px; font-size: 12px; }
    .details-card { padding: 14px; border: 1px solid var(--panel-border); border-radius: 16px; background: var(--panel); }
    .details-empty { color: var(--muted); font-size: 14px; line-height: 1.5; }
    .details-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px 14px; margin-bottom: 14px; }
    .label { display: block; color: var(--muted-soft); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 3px; }
    .value { font-size: 14px; font-weight: 600; }
    .path-block, .hint-block { margin-top: 12px; padding: 10px 12px; border-radius: 12px; background: #f8fafc; border: 1px solid #e2e8f0; font-size: 12px; line-height: 1.45; overflow-wrap: anywhere; }
    .path-block { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    .preview-link { color: #0f766e; font-size: 13px; font-weight: 600; text-decoration: none; }
    .preview-link:hover { text-decoration: underline; }
    .main-grid { display: grid; grid-template-columns: minmax(0, 1fr) 340px; gap: 20px; align-items: start; }
    .plot-shell { padding: 16px; border: 1px solid #cbd5e1; border-radius: 20px; background: var(--panel); box-shadow: 0 18px 50px rgba(15, 23, 42, 0.08); }
    .plot-header { display: flex; justify-content: space-between; gap: 12px; align-items: start; margin-bottom: 10px; }
    .plot-title { font-size: 18px; font-weight: 700; }
    .plot-subtitle { margin-top: 4px; color: var(--muted); font-size: 13px; }
    .plot-status { color: var(--muted); font-size: 13px; }
    .preview-shell { padding: 16px; border: 1px solid #cbd5e1; border-radius: 20px; background: var(--panel); box-shadow: 0 18px 50px rgba(15, 23, 42, 0.08); position: sticky; top: 20px; }
    .preview-header { margin-bottom: 12px; }
    .preview-title { font-size: 16px; font-weight: 700; }
    .preview-subtitle { margin-top: 4px; color: var(--muted); font-size: 13px; line-height: 1.5; }
    .preview-empty { color: var(--muted); font-size: 14px; line-height: 1.6; padding: 6px 0; }
    .preview-content { display: grid; gap: 12px; }
    .preview-meta { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px 12px; }
    .preview-content img { width: 100%; border-radius: 16px; border: 1px solid #cbd5e1; background: #ffffff; }
    .preview-actions { display: flex; justify-content: space-between; align-items: center; gap: 10px; }
    .preview-chip { display: inline-flex; align-items: center; padding: 4px 8px; border-radius: 999px; background: #eff6ff; color: #1d4ed8; font-size: 11px; font-weight: 700; }
    svg { width: 100%; height: auto; display: block; border-radius: 16px; background: #f8fafc; }
    .point { cursor: pointer; transition: opacity 120ms ease, transform 120ms ease; }
    .point:hover { opacity: 1; }
    .centroid-label { font-size: 10px; font-weight: 700; fill: #0f172a; text-anchor: middle; dominant-baseline: middle; pointer-events: none; }
    .status-bar { margin-top: 10px; color: var(--muted); font-size: 13px; }
    @media (max-width: 1180px) {
      .app { grid-template-columns: 1fr; }
      .sidebar { border-right: 0; border-bottom: 1px solid #e2e8f0; }
      .main-grid { grid-template-columns: 1fr; }
      .preview-shell { position: static; }
    }
  </style>
</head>
<body>
  <div class=\"app\">
    <aside class=\"sidebar\">
      <h1>Optimal Path Cluster Explorer</h1>
      <p class=\"lede\">Interactive local viewer for multiple evaluated clustering strategies. Switch embeddings or clustering algorithms, filter clusters, search by source/user/index/seed/path text, and click points to inspect their source SVGs.</p>

      <section class=\"section\">
        <h2 class=\"section-title\">Search</h2>
        <div class=\"search-row\">
          <input id=\"search\" type=\"search\" placeholder=\"Search source, user, index, seed, cluster, or path substring\" />
          <button id=\"clear-search\" type=\"button\">Clear</button>
          <button id=\"clear-selection\" type=\"button\">Reset</button>
        </div>
        <div class=\"toolbar\">
          <button id=\"show-all\" type=\"button\">Show all</button>
          <button id=\"hide-all\" type=\"button\">Hide all</button>
        </div>
      </section>

      <section class=\"section\">
        <h2 class=\"section-title\">Strategies</h2>
        <div id=\"strategy-list\" class=\"strategy-list\"></div>
      </section>

      <section class=\"section\">
        <h2 class=\"section-title\">Clusters</h2>
        <div id=\"cluster-list\" class=\"cluster-list\"></div>
      </section>

      <section class=\"section\">
        <h2 class=\"section-title\">Selection</h2>
        <div class=\"details-card\">
          <div id=\"selection-empty\" class=\"details-empty\">Click a point in the scatter plot to inspect its path and cluster details. The matching grid preview will appear beside the plot.</div>
          <div id=\"selection-content\" hidden>
            <div class=\"details-grid\">
              <div><span class=\"label\">Source</span><span id=\"detail-source\" class=\"value\"></span></div>
              <div><span class=\"label\">Index</span><span id=\"detail-index\" class=\"value\"></span></div>
              <div><span class=\"label\">User / Config</span><span id=\"detail-context\" class=\"value\"></span></div>
              <div><span class=\"label\">Seed</span><span id=\"detail-seed\" class=\"value\"></span></div>
              <div><span class=\"label\">Strategy</span><span id=\"detail-strategy\" class=\"value\"></span></div>
              <div><span class=\"label\">Cluster</span><span id=\"detail-cluster\" class=\"value\"></span></div>
              <div><span class=\"label\">Distance</span><span id=\"detail-distance\" class=\"value\"></span></div>
              <div><span class=\"label\">Moves</span><span id=\"detail-moves\" class=\"value\"></span></div>
              <div><span class=\"label\">Overlaps</span><span id=\"detail-overlaps\" class=\"value\"></span></div>
              <div><span class=\"label\">End</span><span id=\"detail-end\" class=\"value\"></span></div>
              <div><span class=\"label\">Projection</span><span id=\"detail-projection\" class=\"value\"></span></div>
              <div><span class=\"label\">Rank In Cluster</span><span id=\"detail-rank\" class=\"value\"></span></div>
            </div>
            <div>
              <span class=\"label\">Cluster Hint</span>
              <div id=\"detail-hint\" class=\"hint-block\"></div>
            </div>
            <div>
              <span class=\"label\">Path</span>
              <div id=\"detail-path\" class=\"path-block\"></div>
            </div>
          </div>
        </div>
      </section>
    </aside>

    <main class=\"main\">
      <div class=\"main-grid\">
        <div class=\"plot-shell\">
          <div class=\"plot-header\">
            <div>
              <div id=\"plot-title\" class=\"plot-title\"></div>
              <div id=\"plot-subtitle\" class=\"plot-subtitle\"></div>
            </div>
            <div id=\"visible-count\" class=\"plot-status\"></div>
          </div>

          <svg id=\"projection-svg\" viewBox=\"0 0 __WIDTH__ __HEIGHT__\" aria-label=\"Optimal path cluster projection\">
            <rect width=\"100%\" height=\"100%\" fill=\"#f8fafc\"></rect>
            <rect x=\"__PLOT_X0__\" y=\"__PLOT_Y0__\" width=\"__PLOT_WIDTH__\" height=\"__PLOT_HEIGHT__\" fill=\"#ffffff\" stroke=\"#cbd5e1\" stroke-width=\"1\" rx=\"14\"></rect>
            <line x1=\"__PLOT_X0__\" y1=\"__PLOT_MID_Y__\" x2=\"__PLOT_X0_PLUS_WIDTH__\" y2=\"__PLOT_MID_Y__\" stroke=\"#e2e8f0\" stroke-width=\"1\"></line>
            <line x1=\"__PLOT_MID_X__\" y1=\"__PLOT_Y0__\" x2=\"__PLOT_MID_X__\" y2=\"__PLOT_Y0_PLUS_HEIGHT__\" stroke=\"#e2e8f0\" stroke-width=\"1\"></line>
            <text x=\"__PLOT_MID_X__\" y=\"__HEIGHT_MINUS_18__\" text-anchor=\"middle\" fill=\"#64748b\" font-size=\"12\">Projection axis 0</text>
            <text x=\"22\" y=\"__PLOT_MID_Y__\" text-anchor=\"middle\" fill=\"#64748b\" font-size=\"12\" transform=\"rotate(-90 22 __PLOT_MID_Y__)\">Projection axis 1</text>
            <g id=\"centroid-layer\"></g>
            <g id=\"point-layer\"></g>
          </svg>

          <div id=\"status-bar\" class=\"status-bar\">Hover over points for quick context. Click to pin a selection in the sidebar.</div>
        </div>

        <aside class=\"preview-shell\">
          <div class=\"preview-header\">
            <div class=\"preview-title\">Selected Grid</div>
            <div class=\"preview-subtitle\">Keep the active path preview beside the clustering diagram while you compare geometry and cluster placement.</div>
          </div>
          <div id=\"preview-empty\" class=\"preview-empty\">Select a point to load its grid preview here.</div>
          <div id=\"preview-content\" class=\"preview-content\" hidden>
            <div class=\"preview-meta\">
              <div><span class=\"label\">Source</span><span id=\"preview-source\" class=\"value\"></span></div>
              <div><span class=\"label\">Index</span><span id=\"preview-index\" class=\"value\"></span></div>
              <div><span class=\"label\">Cluster</span><span id=\"preview-cluster\" class=\"value\"></span></div>
              <div><span class=\"label\">Strategy</span><span id=\"preview-strategy\" class=\"value\"></span></div>
              <div><span class=\"label\">End</span><span id=\"preview-end\" class=\"value\"></span></div>
            </div>
            <div class=\"preview-actions\">
              <span id=\"preview-chip\" class=\"preview-chip\"></span>
              <a id=\"detail-link\" class=\"preview-link\" href=\"#\" target=\"_blank\" rel=\"noreferrer\">Open source SVG</a>
            </div>
            <img id=\"detail-image\" alt=\"Selected path preview\" />
          </div>
        </aside>
      </div>
    </main>
  </div>

  <script id=\"projection-data\" type=\"application/json\">__PAYLOAD_JSON__</script>
  <script>
    const payload = JSON.parse(document.getElementById("projection-data").textContent);
    const basePoints = payload.points;
    const strategies = payload.strategies;
    const strategyMap = new Map(strategies.map((strategy) => [strategy.key, strategy]));
    const basePointLookup = new Map(basePoints.map((point) => [point.id, point]));

    const pointLayer = document.getElementById("point-layer");
    const centroidLayer = document.getElementById("centroid-layer");
    const strategyList = document.getElementById("strategy-list");
    const clusterList = document.getElementById("cluster-list");
    const searchInput = document.getElementById("search");
    const clearSearchButton = document.getElementById("clear-search");
    const clearSelectionButton = document.getElementById("clear-selection");
    const showAllButton = document.getElementById("show-all");
    const hideAllButton = document.getElementById("hide-all");
    const visibleCount = document.getElementById("visible-count");
    const plotTitle = document.getElementById("plot-title");
    const plotSubtitle = document.getElementById("plot-subtitle");
    const statusBar = document.getElementById("status-bar");

    const selectionEmpty = document.getElementById("selection-empty");
    const selectionContent = document.getElementById("selection-content");
    const detailSource = document.getElementById("detail-source");
    const detailIndex = document.getElementById("detail-index");
    const detailContext = document.getElementById("detail-context");
    const detailSeed = document.getElementById("detail-seed");
    const detailStrategy = document.getElementById("detail-strategy");
    const detailCluster = document.getElementById("detail-cluster");
    const detailDistance = document.getElementById("detail-distance");
    const detailMoves = document.getElementById("detail-moves");
    const detailOverlaps = document.getElementById("detail-overlaps");
    const detailEnd = document.getElementById("detail-end");
    const detailProjection = document.getElementById("detail-projection");
    const detailRank = document.getElementById("detail-rank");
    const detailHint = document.getElementById("detail-hint");
    const detailPath = document.getElementById("detail-path");
    const detailLink = document.getElementById("detail-link");
    const detailImage = document.getElementById("detail-image");
    const previewEmpty = document.getElementById("preview-empty");
    const previewContent = document.getElementById("preview-content");
    const previewSource = document.getElementById("preview-source");
    const previewIndex = document.getElementById("preview-index");
    const previewCluster = document.getElementById("preview-cluster");
    const previewStrategy = document.getElementById("preview-strategy");
    const previewEnd = document.getElementById("preview-end");
    const previewChip = document.getElementById("preview-chip");

    let currentStrategyKey = payload.selected_strategy_key;
    let currentStrategy = strategyMap.get(currentStrategyKey);
    let currentPointLookup = new Map();
    let currentClusterLookup = new Map();
    let activeClusters = new Set();
    let selectedPointId = null;

    const pointElements = new Map();
    const pointTitleNodes = new Map();
    const clusterCheckboxes = new Map();

    function strategyStatusText(strategy) {
      return "silhouette=" + strategy.silhouette.toFixed(4) + " | davies-bouldin=" + strategy.davies_bouldin.toFixed(4);
    }

    function getOverlay(pointId) {
      return currentPointLookup.get(pointId);
    }

    function getClusterInfo(clusterId) {
      return currentClusterLookup.get(clusterId);
    }

    function pointLabel(basePoint) {
      return basePoint.source_id || (basePoint.source + "-" + basePoint.index);
    }

    function pointContext(basePoint) {
      if (basePoint.source !== "human") return "optimal library";
      const parts = [];
      if (basePoint.user) parts.push(basePoint.user);
      if (basePoint.config_label) parts.push(basePoint.config_label);
      if (basePoint.size) parts.push(basePoint.size + "x" + basePoint.size);
      return parts.length ? parts.join(" | ") : "human submission";
    }

    function pointRadius(basePoint) {
      return basePoint.source === "human" ? payload.point_radius + 1.6 : payload.point_radius;
    }

    function matchesQuery(basePoint, overlay, rawQuery) {
      const query = rawQuery.trim().toLowerCase();
      if (!query) return true;
      return String(basePoint.index).includes(query)
        || pointLabel(basePoint).toLowerCase().includes(query)
        || String(basePoint.source || "").toLowerCase().includes(query)
        || String(basePoint.user || "").toLowerCase().includes(query)
        || String(basePoint.config_label || "").toLowerCase().includes(query)
        || String(basePoint.seed).includes(query)
        || String(overlay.cluster) === query
        || basePoint.path.toLowerCase().includes(query)
        || currentStrategy.embedding.toLowerCase().includes(query)
        || currentStrategy.algorithm.toLowerCase().includes(query);
    }

    function visibleForCurrentFilters(basePoint, overlay) {
      return activeClusters.has(overlay.cluster) && matchesQuery(basePoint, overlay, searchInput.value);
    }

    function updatePlotHeader() {
      plotTitle.textContent = currentStrategy.label;
      plotSubtitle.textContent = strategyStatusText(currentStrategy) + " | feature type=" + currentStrategy.feature_type;
    }

    function updateSelection(pointId) {
      if (pointId === null) {
        selectionEmpty.hidden = false;
        selectionContent.hidden = true;
        previewEmpty.hidden = false;
        previewContent.hidden = true;
        detailLink.href = "#";
        detailImage.removeAttribute("src");
        return;
      }
      const basePoint = basePointLookup.get(pointId);
      const overlay = getOverlay(pointId);
      if (!basePoint || !overlay) {
        selectionEmpty.hidden = false;
        selectionContent.hidden = true;
        previewEmpty.hidden = false;
        previewContent.hidden = true;
        return;
      }
      const clusterInfo = getClusterInfo(overlay.cluster);

      selectionEmpty.hidden = true;
      selectionContent.hidden = false;
      previewEmpty.hidden = true;
      previewContent.hidden = false;
      detailSource.textContent = pointLabel(basePoint) + " (" + basePoint.source + ")";
      detailIndex.textContent = String(basePoint.index);
      detailContext.textContent = pointContext(basePoint);
      detailSeed.textContent = String(basePoint.seed);
      detailStrategy.textContent = currentStrategy.label;
      detailCluster.textContent = String(overlay.cluster);
      detailDistance.textContent = overlay.cluster_distance.toFixed(4);
      detailMoves.textContent = String(basePoint.moves);
      detailOverlaps.textContent = String(basePoint.overlaps);
      detailEnd.textContent = "(" + basePoint.end_row + ", " + basePoint.end_col + ")";
      detailProjection.textContent = "(" + overlay.projection_x.toFixed(4) + ", " + overlay.projection_y.toFixed(4) + ")";
      detailRank.textContent = String(overlay.cluster_rank);
      detailHint.textContent = clusterInfo ? clusterInfo.hint : "No hint available";
      detailPath.textContent = basePoint.path;
      detailLink.href = basePoint.image_href;
      detailLink.textContent = "Open " + basePoint.image;
      detailImage.src = basePoint.image_href;
      detailImage.alt = "Path preview for index " + basePoint.index;
      previewSource.textContent = pointLabel(basePoint);
      previewIndex.textContent = String(basePoint.index);
      previewCluster.textContent = String(overlay.cluster);
      previewStrategy.textContent = currentStrategy.label;
      previewEnd.textContent = "(" + basePoint.end_row + ", " + basePoint.end_col + ")";
      previewChip.textContent = clusterInfo ? clusterInfo.hint : "No hint available";
    }

    function syncCheckboxes() {
      for (const cluster of currentStrategy.clusters) {
        const checkbox = clusterCheckboxes.get(cluster.cluster_id);
        if (checkbox) checkbox.checked = activeClusters.has(cluster.cluster_id);
      }
    }

    function renderStrategyList() {
      strategyList.innerHTML = "";
      for (const strategy of strategies) {
        const card = document.createElement("div");
        card.className = "strategy-card" + (strategy.key === currentStrategyKey ? " active" : "");

        const top = document.createElement("div");
        top.className = "strategy-top";

        const left = document.createElement("div");
        const name = document.createElement("div");
        name.className = "strategy-name";
        name.textContent = strategy.label;
        const meta = document.createElement("div");
        meta.className = "strategy-meta";
        meta.textContent = strategyStatusText(strategy);
        left.appendChild(name);
        left.appendChild(meta);

        const useButton = document.createElement("button");
        useButton.type = "button";
        useButton.textContent = strategy.key === currentStrategyKey ? "Active" : "Use";
        useButton.disabled = strategy.key === currentStrategyKey;
        useButton.addEventListener("click", () => setStrategy(strategy.key));

        top.appendChild(left);
        top.appendChild(useButton);

        const tags = document.createElement("div");
        tags.className = "strategy-tags";
        const featureTag = document.createElement("span");
        featureTag.className = "tag";
        featureTag.textContent = strategy.feature_type;
        tags.appendChild(featureTag);
        if (strategy.selected) {
          const selectedTag = document.createElement("span");
          selectedTag.className = "tag selected";
          selectedTag.textContent = "Selected by classifier";
          tags.appendChild(selectedTag);
        }

        card.appendChild(top);
        card.appendChild(tags);
        strategyList.appendChild(card);
      }
    }

    function renderClusterList() {
      clusterList.innerHTML = "";
      clusterCheckboxes.clear();
      for (const cluster of currentStrategy.clusters) {
        const card = document.createElement("div");
        card.className = "cluster-card";

        const top = document.createElement("div");
        top.className = "cluster-top";

        const color = document.createElement("span");
        color.className = "cluster-color";
        color.style.background = cluster.color;

        const label = document.createElement("label");
        label.className = "cluster-name";
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.checked = activeClusters.has(cluster.cluster_id);
        checkbox.style.marginRight = "8px";
        checkbox.addEventListener("change", () => {
          if (checkbox.checked) activeClusters.add(cluster.cluster_id);
          else activeClusters.delete(cluster.cluster_id);
          refreshPoints();
        });
        clusterCheckboxes.set(cluster.cluster_id, checkbox);
        label.appendChild(checkbox);
        label.append("Cluster " + cluster.cluster_id + " (" + cluster.size + ")");

        const onlyButton = document.createElement("button");
        onlyButton.type = "button";
        onlyButton.textContent = "Only";
        onlyButton.addEventListener("click", () => {
          activeClusters = new Set([cluster.cluster_id]);
          syncCheckboxes();
          refreshPoints();
        });

        top.appendChild(color);
        top.appendChild(label);
        top.appendChild(onlyButton);

        const meta = document.createElement("div");
        meta.className = "cluster-meta";
        const motifSummary = cluster.top_motifs.length
          ? cluster.top_motifs.map((motif) => motif.motif).join(", ")
          : "no dominant motifs";
        const sourceSummary = Object.entries(cluster.source_counts || {})
          .map(([source, count]) => source + "=" + count)
          .join(" | ");
        meta.innerHTML = "<div>" + cluster.hint + "</div>"
          + "<div>avg moves=" + cluster.avg_moves.toFixed(2) + " | avg overlaps=" + cluster.avg_overlaps.toFixed(2) + "</div>"
          + "<div>sources: " + (sourceSummary || "unknown") + "</div>"
          + "<div>motifs: " + motifSummary + "</div>";

        const actions = document.createElement("div");
        actions.className = "cluster-actions";

        const showButton = document.createElement("button");
        showButton.type = "button";
        showButton.textContent = "Show";
        showButton.addEventListener("click", () => {
          activeClusters.add(cluster.cluster_id);
          syncCheckboxes();
          refreshPoints();
        });

        const hideButton = document.createElement("button");
        hideButton.type = "button";
        hideButton.textContent = "Hide";
        hideButton.addEventListener("click", () => {
          activeClusters.delete(cluster.cluster_id);
          syncCheckboxes();
          refreshPoints();
        });

        actions.appendChild(showButton);
        actions.appendChild(hideButton);

        card.appendChild(top);
        card.appendChild(meta);
        card.appendChild(actions);
        clusterList.appendChild(card);
      }
    }

    function renderCentroids() {
      centroidLayer.innerHTML = "";
      for (const cluster of currentStrategy.clusters) {
        const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
        group.dataset.cluster = String(cluster.cluster_id);

        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", String(cluster.centroid_x));
        circle.setAttribute("cy", String(cluster.centroid_y));
        circle.setAttribute("r", "9");
        circle.setAttribute("fill", "#ffffff");
        circle.setAttribute("stroke", cluster.color);
        circle.setAttribute("stroke-width", "3");

        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", String(cluster.centroid_x));
        text.setAttribute("y", String(cluster.centroid_y + 1));
        text.setAttribute("class", "centroid-label");
        text.textContent = String(cluster.cluster_id);

        group.appendChild(circle);
        group.appendChild(text);
        centroidLayer.appendChild(group);
      }
    }

    function renderPoints() {
      for (const basePoint of basePoints) {
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("class", "point");
        circle.setAttribute("r", String(payload.point_radius));
        const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
        circle.appendChild(title);

        circle.addEventListener("mouseenter", () => {
          const overlay = getOverlay(basePoint.id);
          statusBar.textContent = "strategy=" + currentStrategy.label
            + " | record=" + pointLabel(basePoint)
            + " seed=" + basePoint.seed
            + " cluster=" + overlay.cluster
            + " moves=" + basePoint.moves
            + " overlaps=" + basePoint.overlaps;
        });
        circle.addEventListener("mouseleave", () => {
          if (selectedPointId === null) {
            statusBar.textContent = "Hover over points for quick context. Click to pin a selection in the sidebar.";
            return;
          }
          const selectedBase = basePointLookup.get(selectedPointId);
          const selectedOverlay = getOverlay(selectedPointId);
          if (!selectedBase || !selectedOverlay) {
            statusBar.textContent = "Hover over points for quick context. Click to pin a selection in the sidebar.";
            return;
          }
          statusBar.textContent = "Pinned selection: strategy=" + currentStrategy.label
            + " | record=" + pointLabel(selectedBase)
            + " cluster=" + selectedOverlay.cluster;
        });
        circle.addEventListener("click", () => {
          selectedPointId = basePoint.id;
          updateSelection(selectedPointId);
          statusBar.textContent = "Pinned selection: strategy=" + currentStrategy.label
            + " | record=" + pointLabel(basePoint)
            + " cluster=" + getOverlay(basePoint.id).cluster;
          refreshPoints();
        });

        pointElements.set(basePoint.id, circle);
        pointTitleNodes.set(basePoint.id, title);
        pointLayer.appendChild(circle);
      }
    }

    function applyStrategyToPoints() {
      currentPointLookup = new Map(currentStrategy.points.map((point) => [point.id, point]));
      currentClusterLookup = new Map(currentStrategy.clusters.map((cluster) => [cluster.cluster_id, cluster]));

      for (const basePoint of basePoints) {
        const overlay = getOverlay(basePoint.id);
        const circle = pointElements.get(basePoint.id);
        const title = pointTitleNodes.get(basePoint.id);
        if (!overlay || !circle || !title) continue;
        circle.setAttribute("cx", String(overlay.screen_x));
        circle.setAttribute("cy", String(overlay.screen_y));
        circle.dataset.baseRadius = String(pointRadius(basePoint));
        circle.dataset.baseStroke = basePoint.source === "human" ? "#0f172a" : "#ffffff";
        circle.dataset.baseStrokeWidth = basePoint.source === "human" ? "1.5" : "0.9";
        circle.dataset.baseOpacity = basePoint.source === "human" ? "0.96" : "0.78";
        circle.setAttribute("fill", overlay.color);
        circle.setAttribute("stroke", circle.dataset.baseStroke);
        circle.setAttribute("stroke-width", circle.dataset.baseStrokeWidth);
        circle.setAttribute("fill-opacity", circle.dataset.baseOpacity);
        title.textContent = "strategy=" + currentStrategy.label
          + " | record=" + pointLabel(basePoint)
          + " source=" + basePoint.source
          + " seed=" + basePoint.seed
          + " cluster=" + overlay.cluster
          + " moves=" + basePoint.moves
          + " overlaps=" + basePoint.overlaps
          + " | path=" + basePoint.path.slice(0, 48);
      }
    }

    function refreshPoints() {
      let visible = 0;
      const visibleByCluster = new Map();
      for (const basePoint of basePoints) {
        const overlay = getOverlay(basePoint.id);
        const circle = pointElements.get(basePoint.id);
        if (!overlay || !circle) continue;
        const isVisible = visibleForCurrentFilters(basePoint, overlay);
        circle.style.display = isVisible ? "" : "none";
        if (isVisible) {
          visible += 1;
          visibleByCluster.set(
            overlay.cluster,
            (visibleByCluster.get(overlay.cluster) || 0) + 1,
          );
        }

        const selected = selectedPointId === basePoint.id;
        const baseRadius = Number(circle.dataset.baseRadius || payload.point_radius);
        const baseStroke = circle.dataset.baseStroke || "#ffffff";
        const baseStrokeWidth = circle.dataset.baseStrokeWidth || "0.9";
        const baseOpacity = circle.dataset.baseOpacity || "0.78";
        circle.setAttribute("r", selected ? String(baseRadius + 1.8) : String(baseRadius));
        circle.setAttribute("stroke", selected ? "#0f172a" : baseStroke);
        circle.setAttribute("stroke-width", selected ? "2.1" : baseStrokeWidth);
        circle.setAttribute("fill-opacity", isVisible ? (selected ? "1" : baseOpacity) : "0");
      }

      for (const group of centroidLayer.children) {
        const clusterId = Number(group.dataset.cluster);
        group.style.display = visibleByCluster.get(clusterId) ? "" : "none";
      }

      visibleCount.textContent = visible + " / " + basePoints.length + " points visible";
      if (selectedPointId !== null) {
        updateSelection(selectedPointId);
      }
    }

    function setStrategy(key) {
      currentStrategyKey = key;
      currentStrategy = strategyMap.get(key);
      activeClusters = new Set(currentStrategy.clusters.map((cluster) => cluster.cluster_id));
      renderStrategyList();
      renderClusterList();
      renderCentroids();
      applyStrategyToPoints();
      updatePlotHeader();
      refreshPoints();
      if (selectedPointId === null) {
        updateSelection(null);
      } else {
        updateSelection(selectedPointId);
      }
    }

    clearSearchButton.addEventListener("click", () => {
      searchInput.value = "";
      refreshPoints();
    });
    clearSelectionButton.addEventListener("click", () => {
      selectedPointId = null;
      updateSelection(null);
      statusBar.textContent = "Hover over points for quick context. Click to pin a selection in the sidebar.";
      refreshPoints();
    });
    showAllButton.addEventListener("click", () => {
      activeClusters = new Set(currentStrategy.clusters.map((cluster) => cluster.cluster_id));
      syncCheckboxes();
      refreshPoints();
    });
    hideAllButton.addEventListener("click", () => {
      activeClusters.clear();
      syncCheckboxes();
      refreshPoints();
    });
    searchInput.addEventListener("input", refreshPoints);

    renderPoints();
    setStrategy(payload.selected_strategy_key);
  </script>
</body>
</html>
"""

    return (
        template.replace("__WIDTH__", str(width))
        .replace("__HEIGHT__", str(height))
        .replace("__PLOT_X0__", str(plot_x0))
        .replace("__PLOT_Y0__", str(plot_y0))
        .replace("__PLOT_WIDTH__", str(plot_width))
        .replace("__PLOT_HEIGHT__", str(plot_height))
        .replace("__PLOT_MID_X__", str(plot_x0 + (plot_width / 2)))
        .replace("__PLOT_MID_Y__", str(plot_y0 + (plot_height / 2)))
        .replace("__PLOT_X0_PLUS_WIDTH__", str(plot_x0 + plot_width))
        .replace("__PLOT_Y0_PLUS_HEIGHT__", str(plot_y0 + plot_height))
        .replace("__HEIGHT_MINUS_18__", str(height - 18))
        .replace("__PAYLOAD_JSON__", json.dumps(payload))
    )


def render_size_index_html(entries: list[dict[str, object]]) -> str:
    items = "\n".join(
        (
            "<li>"
            f'<a href="{escape(str(entry["html_href"]))}">{int(entry["size"])}x{int(entry["size"])}</a>'
            f"<span> rows={int(entry['row_count'])} | "
            f"sources={escape(str(entry['sources']))} | "
            f"strategy={escape(str(entry['strategy']))}</span>"
            "</li>"
        )
        for entry in entries
    )
    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Cluster Explorer Index</title>
  <style>
    body {{ margin: 0; padding: 32px; font-family: Inter, ui-sans-serif, system-ui, sans-serif; background: #f8fafc; color: #0f172a; }}
    .card {{ max-width: 760px; padding: 24px; border: 1px solid #cbd5e1; border-radius: 20px; background: #ffffff; box-shadow: 0 18px 50px rgba(15, 23, 42, 0.08); }}
    h1 {{ margin: 0 0 10px; font-size: 28px; }}
    p {{ margin: 0 0 20px; color: #475569; line-height: 1.6; }}
    ul {{ margin: 0; padding-left: 20px; display: grid; gap: 12px; }}
    li {{ line-height: 1.6; }}
    a {{ font-weight: 700; color: #0f766e; text-decoration: none; margin-right: 10px; }}
    a:hover {{ text-decoration: underline; }}
    span {{ color: #475569; }}
  </style>
</head>
<body>
  <div class=\"card\">
    <h1>Cluster Explorer By Grid Size</h1>
    <p>Each explorer now keeps human and generated paths separated by board size. Open a size-specific page to inspect the clusters for that grid size only.</p>
    <ul>{items}</ul>
  </div>
</body>
</html>
"""


def visualize_clusters_2d(
    clustered_csv: str | Path = DEFAULT_CLUSTERED_CSV,
    *,
    summary_json: str | Path | None = DEFAULT_SUMMARY_JSON,
    size: int | None = None,
    svg_output: str | Path = DEFAULT_SVG_OUTPUT,
    html_output: str | Path | None = DEFAULT_HTML_OUTPUT,
    csv_output: str | Path = DEFAULT_CSV_OUTPUT,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    point_radius: float = DEFAULT_POINT_RADIUS,
    max_components: int = DEFAULT_MAX_COMPONENTS,
    motif_length: int = DEFAULT_MOTIF_LENGTH,
    strategy_limit: int = DEFAULT_STRATEGY_LIMIT,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers")
    if point_radius <= 0:
        raise ValueError("point_radius must be positive")
    if max_components < 2:
        raise ValueError("max_components must be at least 2")
    if motif_length < 2:
        raise ValueError("motif_length must be at least 2")
    if strategy_limit <= 0:
        raise ValueError("strategy_limit must be a positive integer")

    clustered_path = Path(clustered_csv)
    summary_path = None if summary_json is None else Path(summary_json)
    svg_path = Path(svg_output)
    html_path = None if html_output is None else Path(html_output)
    csv_path = Path(csv_output)

    summary = load_summary(summary_path)
    if is_grouped_summary(summary):
        size_entries = list(summary.get("sizes", []))
        if not size_entries:
            raise ValueError("Grouped summary JSON is missing size entries")

        if size is not None:
            selected_entry = next(
                (entry for entry in size_entries if int(entry["size"]) == size),
                None,
            )
            if selected_entry is None:
                available_sizes = ", ".join(
                    str(int(entry["size"])) for entry in size_entries
                )
                raise ValueError(
                    f"Size {size} was not found in the grouped summary. Available sizes: {available_sizes}"
                )

            size_dir = size_projection_dir(
                summary_path.parent, int(selected_entry["size"])
            )
            return visualize_clusters_2d(
                resolve_summary_entry_path(
                    summary_path, str(selected_entry["clustered_csv"])
                ),
                summary_json=resolve_summary_entry_path(
                    summary_path, str(selected_entry["summary_json"])
                ),
                size=None,
                svg_output=size_dir / "cluster_projection.svg",
                html_output=size_dir / "cluster_projection.html",
                csv_output=size_dir / "cluster_projection.csv",
                width=width,
                height=height,
                point_radius=point_radius,
                max_components=max_components,
                motif_length=motif_length,
                strategy_limit=strategy_limit,
                random_state=random_state,
            )

        generated_entries: list[dict[str, object]] = []
        for entry in size_entries:
            size_value = int(entry["size"])
            size_dir = size_projection_dir(summary_path.parent, size_value)
            visualize_clusters_2d(
                resolve_summary_entry_path(summary_path, str(entry["clustered_csv"])),
                summary_json=resolve_summary_entry_path(
                    summary_path, str(entry["summary_json"])
                ),
                size=None,
                svg_output=size_dir / "cluster_projection.svg",
                html_output=size_dir / "cluster_projection.html",
                csv_output=size_dir / "cluster_projection.csv",
                width=width,
                height=height,
                point_radius=point_radius,
                max_components=max_components,
                motif_length=motif_length,
                strategy_limit=strategy_limit,
                random_state=random_state,
            )
            generated_entries.append(
                {
                    "size": size_value,
                    "row_count": int(entry["row_count"]),
                    "sources": ", ".join(
                        f"{source}={count}"
                        for source, count in dict(entry["source_counts"]).items()
                    ),
                    "strategy": (
                        f"{entry['selected_strategy']['embedding']} + "
                        f"{entry['selected_strategy']['algorithm']} "
                        f"(k={int(entry['selected_strategy']['selected_k'])})"
                    ),
                    "html_href": relative_href(
                        html_path.parent
                        if html_path is not None
                        else summary_path.parent,
                        size_dir / "cluster_projection.html",
                    ),
                    "svg_path": str(size_dir / "cluster_projection.svg"),
                    "csv_path": str(size_dir / "cluster_projection.csv"),
                }
            )

        if html_path is not None:
            html_path.parent.mkdir(parents=True, exist_ok=True)
            html_path.write_text(
                render_size_index_html(generated_entries),
                encoding="utf-8",
            )
            print(f"Wrote {html_path}")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(generated_entries).to_csv(csv_path, index=False)
        print(f"Wrote {csv_path}")
        return pd.DataFrame(generated_entries)

    if "clustering_strategy" not in summary:
        raise ValueError("Summary JSON is missing 'clustering_strategy'")

    dataframe = load_clustered_paths(clustered_path)
    asset_root = Path(
        str(summary.get("asset_root", Path(__file__).resolve().parent / "library"))
    )
    base_points, strategy_views, selected_key = build_strategy_views(
        dataframe,
        summary=summary,
        html_output=html_path,
        asset_root=asset_root,
        width=width,
        height=height,
        max_components=max_components,
        motif_length=motif_length,
        strategy_limit=strategy_limit,
        random_state=random_state,
    )
    selected_view = next(
        strategy for strategy in strategy_views if strategy["key"] == selected_key
    )
    selected_overlay = {int(point["id"]): point for point in selected_view["points"]}

    projected = dataframe.copy()
    projected["cluster"] = [
        selected_overlay[index]["cluster"] for index in range(len(projected))
    ]
    projected["cluster_distance"] = [
        selected_overlay[index]["cluster_distance"] for index in range(len(projected))
    ]
    projected["cluster_rank"] = [
        selected_overlay[index]["cluster_rank"] for index in range(len(projected))
    ]
    projected["projection_x"] = [
        selected_overlay[index]["projection_x"] for index in range(len(projected))
    ]
    projected["projection_y"] = [
        selected_overlay[index]["projection_y"] for index in range(len(projected))
    ]

    if html_path is not None:
        base_point_lookup = {int(point["id"]): point for point in base_points}
        projected["image_href"] = [
            base_point_lookup[index]["image_href"] for index in range(len(projected))
        ]

    svg_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if html_path is not None:
        html_path.parent.mkdir(parents=True, exist_ok=True)

    projected.to_csv(csv_path, index=False)

    svg_text = render_svg(
        base_points,
        selected_view,
        width=width,
        height=height,
        point_radius=point_radius,
    )
    svg_path.write_text(svg_text, encoding="utf-8")

    if html_path is not None:
        html_text = render_html(
            {
                "selected_strategy_key": selected_key,
                "point_radius": point_radius,
                "points": base_points,
                "strategies": strategy_views,
            },
            width=width,
            height=height,
        )
        html_path.write_text(html_text, encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {svg_path}")
    if html_path is not None:
        print(f"Wrote {html_path}")
    return projected


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Project clustered optimal paths into 2D and render SVG/HTML explorers "
            "with multiple evaluated clustering strategies"
        )
    )
    parser.add_argument(
        "--clustered-csv",
        default=str(DEFAULT_CLUSTERED_CSV),
        help="Path to the clustered optimal-path CSV",
    )
    parser.add_argument(
        "--summary-json",
        default=str(DEFAULT_SUMMARY_JSON),
        help="Summary JSON from the classifier, including evaluated strategy candidates",
    )
    parser.add_argument(
        "--size",
        type=int,
        help=(
            "Optional grid size to render when the summary JSON is grouped by size; "
            "omit it to generate explorers for every size and a root index page"
        ),
    )
    parser.add_argument(
        "--svg-output",
        default=str(DEFAULT_SVG_OUTPUT),
        help="Output path for the SVG scatter plot",
    )
    parser.add_argument(
        "--html-output",
        default=str(DEFAULT_HTML_OUTPUT),
        help="Output path for the interactive HTML explorer",
    )
    parser.add_argument(
        "--csv-output",
        default=str(DEFAULT_CSV_OUTPUT),
        help="Output path for the projected 2D coordinates CSV",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Output width in pixels (default: {DEFAULT_WIDTH})",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"Output height in pixels (default: {DEFAULT_HEIGHT})",
    )
    parser.add_argument(
        "--point-radius",
        type=float,
        default=DEFAULT_POINT_RADIUS,
        help=f"Point radius in pixels (default: {DEFAULT_POINT_RADIUS})",
    )
    parser.add_argument(
        "--max-components",
        type=int,
        default=DEFAULT_MAX_COMPONENTS,
        help=(
            "Maximum embedding components to reuse while rebuilding strategy views "
            f"(default: {DEFAULT_MAX_COMPONENTS})"
        ),
    )
    parser.add_argument(
        "--motif-length",
        type=int,
        default=DEFAULT_MOTIF_LENGTH,
        help=(
            "Substring length used for per-cluster hint generation "
            f"(default: {DEFAULT_MOTIF_LENGTH})"
        ),
    )
    parser.add_argument(
        "--strategy-limit",
        type=int,
        default=DEFAULT_STRATEGY_LIMIT,
        help=(
            "Number of evaluated strategies to embed in the HTML explorer "
            f"(default: {DEFAULT_STRATEGY_LIMIT})"
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Random seed for rebuilding strategy views (default: {DEFAULT_RANDOM_STATE})",
    )
    args = parser.parse_args()

    visualize_clusters_2d(
        args.clustered_csv,
        summary_json=args.summary_json,
        size=args.size,
        svg_output=args.svg_output,
        html_output=args.html_output,
        csv_output=args.csv_output,
        width=args.width,
        height=args.height,
        point_radius=args.point_radius,
        max_components=args.max_components,
        motif_length=args.motif_length,
        strategy_limit=args.strategy_limit,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
