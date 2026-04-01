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

from src.optimal.classify import build_path_embedding, normalize_path


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
REQUIRED_COLUMNS = {"index", "seed", "path", "cluster", "moves", "overlaps", "image"}
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

    return dataframe


def load_cluster_hints(summary_path: str | Path | None) -> dict[int, str]:
    if summary_path is None:
        return {}

    file_path = Path(summary_path)
    if not file_path.exists():
        return {}

    payload = json.loads(file_path.read_text(encoding="utf-8"))
    hints: dict[int, str] = {}
    for cluster in payload.get("clusters", []):
        cluster_id = int(cluster["cluster_id"])
        hints[cluster_id] = str(cluster.get("hint", ""))
    return hints


def project_paths_to_2d(
    dataframe: pd.DataFrame,
    *,
    max_components: int,
    random_state: int,
) -> np.ndarray:
    paths = [normalize_path(str(path)) for path in dataframe["path"]]
    embedding, _ = build_path_embedding(
        paths,
        max_components=max(max_components, 2),
        random_state=random_state,
    )
    if embedding.shape[1] < 2:
        raise ValueError("Whole-path embedding did not produce at least 2 dimensions")
    return embedding[:, :2]


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


def relative_href(from_dir: Path, target_path: Path) -> str:
    return Path(os.path.relpath(target_path, from_dir)).as_posix()


def render_svg(
    dataframe: pd.DataFrame,
    *,
    hints: dict[int, str],
    svg_output: Path,
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

    min_x, max_x = axis_bounds(dataframe["projection_x"].to_numpy())
    min_y, max_y = axis_bounds(dataframe["projection_y"].to_numpy())

    cluster_ids = sorted(
        int(cluster_id) for cluster_id in dataframe["cluster"].unique()
    )
    color_map = {
        cluster_id: CLUSTER_COLORS[index % len(CLUSTER_COLORS)]
        for index, cluster_id in enumerate(cluster_ids)
    }

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#f8fafc" />',
        f'<text x="{margin_left}" y="32" fill="#0f172a" font-size="24" font-weight="700">Optimal Path Clusters (2D)</text>',
        f'<text x="{margin_left}" y="54" fill="#475569" font-size="13">Projection uses the first two dimensions of the whole-path TF-IDF/SVD embedding.</text>',
        f'<rect x="{plot_x0}" y="{plot_y0}" width="{plot_width}" height="{plot_height}" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="12" />',
    ]

    mid_x = plot_x0 + plot_width / 2
    mid_y = plot_y0 + plot_height / 2
    parts.append(
        f'<line x1="{plot_x0}" y1="{mid_y}" x2="{plot_x0 + plot_width}" y2="{mid_y}" stroke="#e2e8f0" stroke-width="1" />'
    )
    parts.append(
        f'<line x1="{mid_x}" y1="{plot_y0}" x2="{mid_x}" y2="{plot_y0 + plot_height}" stroke="#e2e8f0" stroke-width="1" />'
    )
    parts.append(
        f'<text x="{plot_x0 + plot_width / 2}" y="{height - 18}" text-anchor="middle" fill="#475569" font-size="12">SVD component 0</text>'
    )
    parts.append(
        f'<text x="20" y="{plot_y0 + plot_height / 2}" text-anchor="middle" fill="#475569" font-size="12" transform="rotate(-90 20 {plot_y0 + plot_height / 2})">SVD component 1</text>'
    )

    for row in dataframe.to_dict("records"):
        cluster_id = int(row["cluster"])
        color = color_map[cluster_id]
        cx, cy = scale_point(
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
        tooltip = escape(
            (
                f"index={int(row['index'])} seed={int(row['seed'])} cluster={cluster_id} "
                f"moves={int(row['moves'])} overlaps={int(row['overlaps'])} "
                f"path={str(row['path'])[:48]}"
            )
        )
        parts.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{point_radius}" fill="{color}" fill-opacity="0.76" stroke="#ffffff" stroke-width="0.9"><title>{tooltip}</title></circle>'
        )

    legend_x = plot_x0 + plot_width + 24
    legend_y = plot_y0 + 10
    parts.append(
        f'<text x="{legend_x}" y="{legend_y}" fill="#0f172a" font-size="16" font-weight="700">Clusters</text>'
    )

    for index, cluster_id in enumerate(cluster_ids):
        subset = dataframe[dataframe["cluster"] == cluster_id]
        cx_mean = float(subset["projection_x"].mean())
        cy_mean = float(subset["projection_y"].mean())
        centroid_x, centroid_y = scale_point(
            cx_mean,
            cy_mean,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            plot_x0=plot_x0,
            plot_y0=plot_y0,
            plot_width=plot_width,
            plot_height=plot_height,
        )
        color = color_map[cluster_id]
        parts.append(
            f'<circle cx="{centroid_x:.2f}" cy="{centroid_y:.2f}" r="9" fill="#ffffff" stroke="{color}" stroke-width="3" />'
        )
        parts.append(
            f'<text x="{centroid_x:.2f}" y="{centroid_y + 4:.2f}" text-anchor="middle" fill="#0f172a" font-size="10" font-weight="700">{cluster_id}</text>'
        )

        entry_y = legend_y + 28 + (index * 64)
        hint = escape(hints.get(cluster_id, ""))
        parts.append(
            f'<circle cx="{legend_x + 8}" cy="{entry_y - 5}" r="7" fill="{color}" />'
        )
        parts.append(
            f'<text x="{legend_x + 24}" y="{entry_y}" fill="#0f172a" font-size="13" font-weight="700">Cluster {cluster_id} ({len(subset)})</text>'
        )
        if hint:
            parts.append(
                f'<text x="{legend_x + 24}" y="{entry_y + 18}" fill="#475569" font-size="12">{hint}</text>'
            )
        parts.append(
            f'<text x="{legend_x + 24}" y="{entry_y + 36}" fill="#64748b" font-size="11">moves={subset["moves"].mean():.2f} overlaps={subset["overlaps"].mean():.2f}</text>'
        )

    parts.append(
        f'<text x="{legend_x}" y="{height - 24}" fill="#64748b" font-size="11">Hover points to inspect index, seed, and path preview.</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def render_html(
    dataframe: pd.DataFrame,
    *,
    hints: dict[int, str],
    html_output: Path,
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

    min_x, max_x = axis_bounds(dataframe["projection_x"].to_numpy())
    min_y, max_y = axis_bounds(dataframe["projection_y"].to_numpy())

    cluster_ids = sorted(
        int(cluster_id) for cluster_id in dataframe["cluster"].unique()
    )
    color_map = {
        cluster_id: CLUSTER_COLORS[index % len(CLUSTER_COLORS)]
        for index, cluster_id in enumerate(cluster_ids)
    }

    points: list[dict[str, object]] = []
    for point_id, row in enumerate(dataframe.to_dict("records")):
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
        points.append(
            {
                "id": point_id,
                "cluster": int(row["cluster"]),
                "index": int(row["index"]),
                "seed": int(row["seed"]),
                "moves": int(row["moves"]),
                "overlaps": int(row["overlaps"]),
                "end_row": int(row["end_row"]),
                "end_col": int(row["end_col"]),
                "cluster_distance": round(float(row["cluster_distance"]), 4),
                "projection_x": round(float(row["projection_x"]), 5),
                "projection_y": round(float(row["projection_y"]), 5),
                "screen_x": round(float(screen_x), 2),
                "screen_y": round(float(screen_y), 2),
                "path": str(row["path"]),
                "image": str(row["image"]),
                "image_href": str(row["image_href"]),
                "color": color_map[int(row["cluster"])],
            }
        )

    clusters: list[dict[str, object]] = []
    for cluster_id in cluster_ids:
        subset = dataframe[dataframe["cluster"] == cluster_id]
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
        clusters.append(
            {
                "cluster_id": cluster_id,
                "color": color_map[cluster_id],
                "size": int(len(subset)),
                "hint": hints.get(cluster_id, ""),
                "avg_moves": round(float(subset["moves"].mean()), 2),
                "avg_overlaps": round(float(subset["overlaps"].mean()), 2),
                "centroid_x": round(float(centroid_x), 2),
                "centroid_y": round(float(centroid_y), 2),
            }
        )

    payload = {
        "plot": {
            "width": width,
            "height": height,
            "plot_x0": plot_x0,
            "plot_y0": plot_y0,
            "plot_width": plot_width,
            "plot_height": plot_height,
            "mid_x": plot_x0 + plot_width / 2,
            "mid_y": plot_y0 + plot_height / 2,
            "point_radius": point_radius,
        },
        "clusters": clusters,
        "points": points,
    }
    payload_json = json.dumps(payload)

    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Optimal Path Cluster Explorer</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      --bg: #f8fafc;
      --panel: #ffffff;
      --panel-border: #cbd5e1;
      --muted: #475569;
      --muted-soft: #64748b;
      --text: #0f172a;
      --accent: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); }}
    .app {{ display: grid; grid-template-columns: 340px 1fr; min-height: 100vh; }}
    .sidebar {{ padding: 20px; border-right: 1px solid #e2e8f0; background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%); }}
    .main {{ padding: 20px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; line-height: 1.2; }}
    .lede {{ margin: 0 0 20px; color: var(--muted); font-size: 14px; line-height: 1.5; }}
    .section {{ margin-bottom: 20px; }}
    .section-title {{ margin: 0 0 10px; font-size: 12px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted-soft); }}
    .search-row {{ display: grid; grid-template-columns: 1fr auto auto; gap: 8px; }}
    input[type=\"search\"] {{ width: 100%; padding: 10px 12px; border: 1px solid var(--panel-border); border-radius: 10px; font: inherit; background: var(--panel); }}
    button {{ padding: 10px 12px; border: 1px solid var(--panel-border); border-radius: 10px; background: var(--panel); font: inherit; color: var(--text); cursor: pointer; }}
    button:hover {{ border-color: #94a3b8; }}
    .toolbar {{ display: flex; gap: 8px; margin-top: 10px; }}
    .cluster-list {{ display: grid; gap: 10px; }}
    .cluster-card {{ padding: 12px; border: 1px solid var(--panel-border); border-radius: 14px; background: var(--panel); }}
    .cluster-top {{ display: grid; grid-template-columns: auto 1fr auto; gap: 10px; align-items: center; margin-bottom: 8px; }}
    .cluster-color {{ width: 12px; height: 12px; border-radius: 999px; }}
    .cluster-name {{ font-weight: 700; font-size: 14px; }}
    .cluster-meta {{ color: var(--muted); font-size: 12px; line-height: 1.5; }}
    .cluster-actions {{ display: flex; gap: 6px; margin-top: 8px; }}
    .cluster-actions button {{ padding: 6px 10px; font-size: 12px; }}
    .details-card {{ padding: 14px; border: 1px solid var(--panel-border); border-radius: 16px; background: var(--panel); }}
    .details-empty {{ color: var(--muted); font-size: 14px; line-height: 1.5; }}
    .details-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px 14px; margin-bottom: 14px; }}
    .label {{ display: block; color: var(--muted-soft); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 3px; }}
    .value {{ font-size: 14px; font-weight: 600; }}
    .path-block {{ margin-top: 12px; padding: 10px 12px; border-radius: 12px; background: #f8fafc; border: 1px solid #e2e8f0; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; line-height: 1.45; overflow-wrap: anywhere; }}
    .preview {{ margin-top: 14px; display: grid; gap: 10px; }}
    .preview img {{ width: 100%; border-radius: 14px; border: 1px solid #cbd5e1; background: #ffffff; }}
    .preview a {{ color: #0f766e; font-size: 13px; font-weight: 600; text-decoration: none; }}
    .preview a:hover {{ text-decoration: underline; }}
    .plot-shell {{ padding: 16px; border: 1px solid #cbd5e1; border-radius: 20px; background: var(--panel); box-shadow: 0 18px 50px rgba(15, 23, 42, 0.08); }}
    .plot-header {{ display: flex; justify-content: space-between; gap: 12px; align-items: baseline; margin-bottom: 10px; }}
    .plot-title {{ font-size: 18px; font-weight: 700; }}
    .plot-status {{ color: var(--muted); font-size: 13px; }}
    svg {{ width: 100%; height: auto; display: block; border-radius: 16px; background: #f8fafc; }}
    .point {{ cursor: pointer; transition: opacity 120ms ease, transform 120ms ease; }}
    .point:hover {{ opacity: 1; }}
    .centroid-label {{ font-size: 10px; font-weight: 700; fill: #0f172a; text-anchor: middle; dominant-baseline: middle; pointer-events: none; }}
    .status-bar {{ margin-top: 10px; color: var(--muted); font-size: 13px; }}
    @media (max-width: 1100px) {{
      .app {{ grid-template-columns: 1fr; }}
      .sidebar {{ border-right: 0; border-bottom: 1px solid #e2e8f0; }}
    }}
  </style>
</head>
<body>
  <div class=\"app\">
    <aside class=\"sidebar\">
      <h1>Optimal Path Cluster Explorer</h1>
      <p class=\"lede\">Interactive local viewer for the 2D whole-path projection. Filter clusters, search by index/seed/path text, and click points to inspect their source SVGs.</p>

      <section class=\"section\">
        <h2 class=\"section-title\">Search</h2>
        <div class=\"search-row\">
          <input id=\"search\" type=\"search\" placeholder=\"Search index, seed, cluster, or path substring\" />
          <button id=\"clear-search\" type=\"button\">Clear</button>
          <button id=\"clear-selection\" type=\"button\">Reset</button>
        </div>
        <div class=\"toolbar\">
          <button id=\"show-all\" type=\"button\">Show all</button>
          <button id=\"hide-all\" type=\"button\">Hide all</button>
        </div>
      </section>

      <section class=\"section\">
        <h2 class=\"section-title\">Clusters</h2>
        <div id=\"cluster-list\" class=\"cluster-list\"></div>
      </section>

      <section class=\"section\">
        <h2 class=\"section-title\">Selection</h2>
        <div class=\"details-card\">
          <div id=\"selection-empty\" class=\"details-empty\">Click a point in the scatter plot to inspect its path, cluster assignment, and source SVG preview.</div>
          <div id=\"selection-content\" hidden>
            <div class=\"details-grid\">
              <div><span class=\"label\">Index</span><span id=\"detail-index\" class=\"value\"></span></div>
              <div><span class=\"label\">Seed</span><span id=\"detail-seed\" class=\"value\"></span></div>
              <div><span class=\"label\">Cluster</span><span id=\"detail-cluster\" class=\"value\"></span></div>
              <div><span class=\"label\">Distance</span><span id=\"detail-distance\" class=\"value\"></span></div>
              <div><span class=\"label\">Moves</span><span id=\"detail-moves\" class=\"value\"></span></div>
              <div><span class=\"label\">Overlaps</span><span id=\"detail-overlaps\" class=\"value\"></span></div>
              <div><span class=\"label\">End</span><span id=\"detail-end\" class=\"value\"></span></div>
              <div><span class=\"label\">Projection</span><span id=\"detail-projection\" class=\"value\"></span></div>
            </div>
            <div>
              <span class=\"label\">Path</span>
              <div id=\"detail-path\" class=\"path-block\"></div>
            </div>
            <div class=\"preview\">
              <a id=\"detail-link\" href=\"#\" target=\"_blank\" rel=\"noreferrer\">Open source SVG</a>
              <img id=\"detail-image\" alt=\"Selected path preview\" />
            </div>
          </div>
        </div>
      </section>
    </aside>

    <main class=\"main\">
      <div class=\"plot-shell\">
        <div class=\"plot-header\">
          <div class=\"plot-title\">2D Cluster Projection</div>
          <div id=\"visible-count\" class=\"plot-status\"></div>
        </div>

        <svg id=\"projection-svg\" viewBox=\"0 0 {width} {height}\" aria-label=\"Optimal path cluster projection\">
          <rect width=\"100%\" height=\"100%\" fill=\"#f8fafc\"></rect>
          <rect x=\"{plot_x0}\" y=\"{plot_y0}\" width=\"{plot_width}\" height=\"{plot_height}\" fill=\"#ffffff\" stroke=\"#cbd5e1\" stroke-width=\"1\" rx=\"14\"></rect>
          <line x1=\"{plot_x0}\" y1=\"{plot_y0 + plot_height / 2}\" x2=\"{plot_x0 + plot_width}\" y2=\"{plot_y0 + plot_height / 2}\" stroke=\"#e2e8f0\" stroke-width=\"1\"></line>
          <line x1=\"{plot_x0 + plot_width / 2}\" y1=\"{plot_y0}\" x2=\"{plot_x0 + plot_width / 2}\" y2=\"{plot_y0 + plot_height}\" stroke=\"#e2e8f0\" stroke-width=\"1\"></line>
          <text x=\"{plot_x0 + plot_width / 2}\" y=\"{height - 18}\" text-anchor=\"middle\" fill=\"#64748b\" font-size=\"12\">SVD component 0</text>
          <text x=\"22\" y=\"{plot_y0 + plot_height / 2}\" text-anchor=\"middle\" fill=\"#64748b\" font-size=\"12\" transform=\"rotate(-90 22 {plot_y0 + plot_height / 2})\">SVD component 1</text>
          <g id=\"centroid-layer\"></g>
          <g id=\"point-layer\"></g>
        </svg>

        <div id=\"status-bar\" class=\"status-bar\">Hover over points for quick context. Click to pin a selection in the sidebar.</div>
      </div>
    </main>
  </div>

  <script id=\"projection-data\" type=\"application/json\">{payload_json}</script>
  <script>
    const payload = JSON.parse(document.getElementById("projection-data").textContent);
    const points = payload.points;
    const clusters = payload.clusters;
    const plot = payload.plot;

    const pointLayer = document.getElementById("point-layer");
    const centroidLayer = document.getElementById("centroid-layer");
    const clusterList = document.getElementById("cluster-list");
    const searchInput = document.getElementById("search");
    const clearSearchButton = document.getElementById("clear-search");
    const clearSelectionButton = document.getElementById("clear-selection");
    const showAllButton = document.getElementById("show-all");
    const hideAllButton = document.getElementById("hide-all");
    const visibleCount = document.getElementById("visible-count");
    const statusBar = document.getElementById("status-bar");

    const selectionEmpty = document.getElementById("selection-empty");
    const selectionContent = document.getElementById("selection-content");
    const detailIndex = document.getElementById("detail-index");
    const detailSeed = document.getElementById("detail-seed");
    const detailCluster = document.getElementById("detail-cluster");
    const detailDistance = document.getElementById("detail-distance");
    const detailMoves = document.getElementById("detail-moves");
    const detailOverlaps = document.getElementById("detail-overlaps");
    const detailEnd = document.getElementById("detail-end");
    const detailProjection = document.getElementById("detail-projection");
    const detailPath = document.getElementById("detail-path");
    const detailLink = document.getElementById("detail-link");
    const detailImage = document.getElementById("detail-image");

    const activeClusters = new Set(clusters.map((cluster) => cluster.cluster_id));
    const pointElements = new Map();
    const centroidElements = new Map();
    const checkboxElements = new Map();
    const pointLookup = new Map(points.map((point) => [point.id, point]));
    let selectedPointId = null;

    function matchesQuery(point, rawQuery) {{
      const query = rawQuery.trim().toLowerCase();
      if (!query) return true;
      return String(point.index).includes(query)
        || String(point.seed).includes(query)
        || String(point.cluster) === query
        || point.path.toLowerCase().includes(query);
    }}

    function visibleForCurrentFilters(point) {{
      return activeClusters.has(point.cluster) && matchesQuery(point, searchInput.value);
    }}

    function updateSelection(point) {{
      if (!point) {{
        selectionEmpty.hidden = false;
        selectionContent.hidden = true;
        return;
      }}
      selectionEmpty.hidden = true;
      selectionContent.hidden = false;
      detailIndex.textContent = point.index;
      detailSeed.textContent = point.seed;
      detailCluster.textContent = point.cluster;
      detailDistance.textContent = point.cluster_distance.toFixed(4);
      detailMoves.textContent = point.moves;
      detailOverlaps.textContent = point.overlaps;
      detailEnd.textContent = `(${{point.end_row}}, ${{point.end_col}})`;
      detailProjection.textContent = `(${{point.projection_x.toFixed(4)}}, ${{point.projection_y.toFixed(4)}})`;
      detailPath.textContent = point.path;
      detailLink.href = point.image_href;
      detailLink.textContent = `Open ${{point.image}}`;
      detailImage.src = point.image_href;
      detailImage.alt = `Path preview for index ${{point.index}}`;
    }}

    function syncCheckboxes() {{
      for (const cluster of clusters) {{
        const checkbox = checkboxElements.get(cluster.cluster_id);
        if (checkbox) checkbox.checked = activeClusters.has(cluster.cluster_id);
      }}
    }}

    function refreshPoints() {{
      let visible = 0;
      for (const point of points) {{
        const element = pointElements.get(point.id);
        const isVisible = visibleForCurrentFilters(point);
        element.style.display = isVisible ? "" : "none";
        if (isVisible) visible += 1;
        const selected = selectedPointId === point.id;
        element.setAttribute("r", selected ? String(plot.point_radius + 2.4) : String(plot.point_radius));
        element.setAttribute("stroke", selected ? "#0f172a" : "#ffffff");
        element.setAttribute("stroke-width", selected ? "2.1" : "0.9");
        element.setAttribute("fill-opacity", isVisible ? (selected ? "1" : "0.78") : "0");
      }}

      for (const cluster of clusters) {{
        const centroid = centroidElements.get(cluster.cluster_id);
        const clusterVisible = points.some((point) => point.cluster === cluster.cluster_id && visibleForCurrentFilters(point));
        if (centroid) centroid.style.display = clusterVisible ? "" : "none";
      }}

      visibleCount.textContent = `${{visible}} / ${{points.length}} points visible`;

      if (selectedPointId !== null) {{
        const selectedPoint = pointLookup.get(selectedPointId);
        updateSelection(selectedPoint);
      }}
    }}

    function renderClusters() {{
      for (const cluster of clusters) {{
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
        checkbox.checked = true;
        checkbox.style.marginRight = "8px";
        checkbox.addEventListener("change", () => {{
          if (checkbox.checked) activeClusters.add(cluster.cluster_id);
          else activeClusters.delete(cluster.cluster_id);
          refreshPoints();
        }});
        checkboxElements.set(cluster.cluster_id, checkbox);
        label.appendChild(checkbox);
        label.append(`Cluster ${{cluster.cluster_id}} (${{cluster.size}})`);

        const onlyButton = document.createElement("button");
        onlyButton.type = "button";
        onlyButton.textContent = "Only";
        onlyButton.addEventListener("click", () => {{
          activeClusters.clear();
          activeClusters.add(cluster.cluster_id);
          syncCheckboxes();
          refreshPoints();
        }});

        top.appendChild(color);
        top.appendChild(label);
        top.appendChild(onlyButton);

        const meta = document.createElement("div");
        meta.className = "cluster-meta";
        meta.innerHTML = `
          <div>${{cluster.hint || "No hint available"}}</div>
          <div>avg moves=${{cluster.avg_moves.toFixed(2)}} | avg overlaps=${{cluster.avg_overlaps.toFixed(2)}}</div>
        `;

        const actions = document.createElement("div");
        actions.className = "cluster-actions";

        const showButton = document.createElement("button");
        showButton.type = "button";
        showButton.textContent = "Show";
        showButton.addEventListener("click", () => {{
          activeClusters.add(cluster.cluster_id);
          syncCheckboxes();
          refreshPoints();
        }});

        const hideButton = document.createElement("button");
        hideButton.type = "button";
        hideButton.textContent = "Hide";
        hideButton.addEventListener("click", () => {{
          activeClusters.delete(cluster.cluster_id);
          syncCheckboxes();
          refreshPoints();
        }});

        actions.appendChild(showButton);
        actions.appendChild(hideButton);

        card.appendChild(top);
        card.appendChild(meta);
        card.appendChild(actions);
        clusterList.appendChild(card);

        const centroidGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
        const centroidCircle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        centroidCircle.setAttribute("cx", cluster.centroid_x);
        centroidCircle.setAttribute("cy", cluster.centroid_y);
        centroidCircle.setAttribute("r", "9");
        centroidCircle.setAttribute("fill", "#ffffff");
        centroidCircle.setAttribute("stroke", cluster.color);
        centroidCircle.setAttribute("stroke-width", "3");

        const centroidLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
        centroidLabel.setAttribute("x", cluster.centroid_x);
        centroidLabel.setAttribute("y", cluster.centroid_y + 1);
        centroidLabel.setAttribute("class", "centroid-label");
        centroidLabel.textContent = cluster.cluster_id;

        centroidGroup.appendChild(centroidCircle);
        centroidGroup.appendChild(centroidLabel);
        centroidLayer.appendChild(centroidGroup);
        centroidElements.set(cluster.cluster_id, centroidGroup);
      }}
    }}

    function renderPoints() {{
      for (const point of points) {{
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("class", "point");
        circle.setAttribute("cx", point.screen_x);
        circle.setAttribute("cy", point.screen_y);
        circle.setAttribute("r", plot.point_radius);
        circle.setAttribute("fill", point.color);
        circle.setAttribute("stroke", "#ffffff");
        circle.setAttribute("stroke-width", "0.9");
        circle.setAttribute("fill-opacity", "0.78");
        circle.addEventListener("mouseenter", () => {{
          statusBar.textContent = `index=${{point.index}} seed=${{point.seed}} cluster=${{point.cluster}} moves=${{point.moves}} overlaps=${{point.overlaps}}`;
        }});
        circle.addEventListener("mouseleave", () => {{
          statusBar.textContent = selectedPointId === null
            ? "Hover over points for quick context. Click to pin a selection in the sidebar."
            : `Pinned selection: index=${{pointLookup.get(selectedPointId).index}} cluster=${{pointLookup.get(selectedPointId).cluster}}`;
        }});
        circle.addEventListener("click", () => {{
          selectedPointId = point.id;
          updateSelection(point);
          statusBar.textContent = `Pinned selection: index=${{point.index}} cluster=${{point.cluster}}`;
          refreshPoints();
        }});
        pointElements.set(point.id, circle);
        pointLayer.appendChild(circle);
      }}
    }}

    clearSearchButton.addEventListener("click", () => {{
      searchInput.value = "";
      refreshPoints();
    }});
    clearSelectionButton.addEventListener("click", () => {{
      selectedPointId = null;
      updateSelection(null);
      statusBar.textContent = "Hover over points for quick context. Click to pin a selection in the sidebar.";
      refreshPoints();
    }});
    showAllButton.addEventListener("click", () => {{
      for (const cluster of clusters) activeClusters.add(cluster.cluster_id);
      syncCheckboxes();
      refreshPoints();
    }});
    hideAllButton.addEventListener("click", () => {{
      activeClusters.clear();
      syncCheckboxes();
      refreshPoints();
    }});
    searchInput.addEventListener("input", refreshPoints);

    renderClusters();
    renderPoints();
    refreshPoints();
  </script>
</body>
</html>
"""


def visualize_clusters_2d(
    clustered_csv: str | Path = DEFAULT_CLUSTERED_CSV,
    *,
    summary_json: str | Path | None = DEFAULT_SUMMARY_JSON,
    svg_output: str | Path = DEFAULT_SVG_OUTPUT,
    html_output: str | Path | None = DEFAULT_HTML_OUTPUT,
    csv_output: str | Path = DEFAULT_CSV_OUTPUT,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    point_radius: float = DEFAULT_POINT_RADIUS,
    max_components: int = DEFAULT_MAX_COMPONENTS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers")
    if point_radius <= 0:
        raise ValueError("point_radius must be positive")

    clustered_path = Path(clustered_csv)
    svg_path = Path(svg_output)
    html_path = None if html_output is None else Path(html_output)
    csv_path = Path(csv_output)

    dataframe = load_clustered_paths(clustered_path)
    projection = project_paths_to_2d(
        dataframe,
        max_components=max_components,
        random_state=random_state,
    )

    projected = dataframe.copy()
    projected["projection_x"] = projection[:, 0]
    projected["projection_y"] = projection[:, 1]

    library_root = clustered_path.parent.parent
    if html_path is not None:
        projected["image_href"] = [
            relative_href(
                html_path.parent,
                library_root / Path(str(image_path)),
            )
            for image_path in projected["image"]
        ]
    else:
        projected["image_href"] = ["" for _ in range(len(projected))]

    svg_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if html_path is not None:
        html_path.parent.mkdir(parents=True, exist_ok=True)
    projected.sort_values(["cluster", "cluster_rank", "index"]).to_csv(
        csv_path,
        index=False,
    )

    hints = load_cluster_hints(summary_json)
    svg_text = render_svg(
        projected,
        hints=hints,
        svg_output=svg_path,
        width=width,
        height=height,
        point_radius=point_radius,
    )
    svg_path.write_text(svg_text, encoding="utf-8")

    if html_path is not None:
        html_text = render_html(
            projected,
            hints=hints,
            html_output=html_path,
            width=width,
            height=height,
            point_radius=point_radius,
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
            "Project clustered optimal paths into 2D and render SVG/HTML explorers"
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
        help="Optional cluster summary JSON used to annotate the legend",
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
        help=f"SVG width in pixels (default: {DEFAULT_WIDTH})",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"SVG height in pixels (default: {DEFAULT_HEIGHT})",
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
            "Maximum whole-path SVD components to compute before projecting to the first two "
            f"dimensions (default: {DEFAULT_MAX_COMPONENTS})"
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Random seed for the embedding (default: {DEFAULT_RANDOM_STATE})",
    )
    args = parser.parse_args()

    visualize_clusters_2d(
        args.clustered_csv,
        summary_json=args.summary_json,
        svg_output=args.svg_output,
        html_output=args.html_output,
        csv_output=args.csv_output,
        width=args.width,
        height=args.height,
        point_radius=args.point_radius,
        max_components=args.max_components,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
