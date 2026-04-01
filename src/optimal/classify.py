import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


DEFAULT_DATASET = Path("src/optimal/library/optimal_paths.csv")
DEFAULT_OUTPUT_DIRNAME = "clusters"
DEFAULT_MIN_CLUSTERS = 2
DEFAULT_MAX_CLUSTERS = 8
DEFAULT_EXAMPLES_PER_CLUSTER = 5
DEFAULT_RANDOM_STATE = 42
DEFAULT_MAX_COMPONENTS = 8
DEFAULT_MOTIF_LENGTH = 5
DEFAULT_MIN_MOTIF_COUNT = 8
PATH_NGRAM_RANGE = (3, 8)
VALID_PATH_CHARS = frozenset("udlr")
REQUIRED_COLUMNS = {
    "index",
    "size",
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


@dataclass(frozen=True)
class ClusterEvaluation:
    n_clusters: int
    silhouette: float


def normalize_path(path: str) -> str:
    normalized = path.strip().lower()
    if not normalized:
        raise ValueError("Path cannot be empty")
    if any(ch not in VALID_PATH_CHARS for ch in normalized):
        raise ValueError("Path must contain only the characters: u, d, l, r")
    return normalized


def load_optimal_paths(csv_path: str | Path) -> pd.DataFrame:
    file_path = Path(csv_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    dataframe = pd.read_csv(file_path)
    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV is missing required columns: {missing}")
    if dataframe.empty:
        raise ValueError("Dataset is empty")

    return dataframe


def run_lengths(path: str) -> list[int]:
    if not path:
        return [0]

    runs: list[int] = []
    current = path[0]
    count = 1
    for move in path[1:]:
        if move == current:
            count += 1
            continue
        runs.append(count)
        current = move
        count = 1

    runs.append(count)
    return runs


def turn_count(path: str) -> int:
    return sum(1 for left, right in zip(path, path[1:]) if left != right)


def build_summary_features(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    normalized_paths: list[str] = []
    derived_rows: list[dict[str, float | str]] = []

    for row in dataframe.to_dict("records"):
        path = normalize_path(str(row["path"]))
        move_counts = Counter(path)
        runs = run_lengths(path)
        total_moves = len(path)

        normalized_paths.append(path)
        derived_rows.append(
            {
                "move_fraction_u": move_counts["u"] / total_moves,
                "move_fraction_d": move_counts["d"] / total_moves,
                "move_fraction_l": move_counts["l"] / total_moves,
                "move_fraction_r": move_counts["r"] / total_moves,
                "turn_rate": turn_count(path) / max(total_moves - 1, 1),
                "mean_run_length": float(np.mean(runs)),
                "max_run_length": float(max(runs)),
                "min_run_length": float(min(runs)),
                "run_length_std": float(np.std(runs)),
            }
        )

    return pd.DataFrame(derived_rows), normalized_paths


def build_path_embedding(
    paths: list[str],
    *,
    max_components: int,
    random_state: int,
) -> tuple[np.ndarray, list[str]]:
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=PATH_NGRAM_RANGE,
        lowercase=False,
    )
    path_matrix = vectorizer.fit_transform(paths)
    n_components = min(
        max_components,
        path_matrix.shape[0] - 1,
        path_matrix.shape[1] - 1,
    )
    if n_components < 2:
        raise ValueError("Unable to derive enough whole-path embedding components")

    embedding = TruncatedSVD(
        n_components=n_components, random_state=random_state
    ).fit_transform(path_matrix)
    component_names = [f"svd_component_{index}" for index in range(embedding.shape[1])]
    return embedding, component_names


def evaluate_cluster_counts(
    features: np.ndarray,
    *,
    min_clusters: int,
    max_clusters: int,
    random_state: int,
) -> list[ClusterEvaluation]:
    if len(features) < 2:
        raise ValueError("Need at least two paths to build clusters")

    upper_bound = min(max_clusters, len(features) - 1)
    lower_bound = min(min_clusters, upper_bound)
    if lower_bound < 2 or upper_bound < 2:
        raise ValueError("Cluster search range must allow at least 2 clusters")

    evaluations: list[ClusterEvaluation] = []
    for n_clusters in range(lower_bound, upper_bound + 1):
        model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
        labels = model.fit_predict(features)
        score = float(silhouette_score(features, labels))
        evaluations.append(ClusterEvaluation(n_clusters=n_clusters, silhouette=score))

    return evaluations


def choose_cluster_count(evaluations: list[ClusterEvaluation]) -> int:
    if not evaluations:
        raise ValueError("No cluster evaluations available")

    best = max(evaluations, key=lambda item: (item.silhouette, -item.n_clusters))
    return best.n_clusters


def relabel_clusters(
    labels: np.ndarray,
    centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    label_counts = Counter(int(label) for label in labels)
    old_order = sorted(label_counts, key=lambda label: (-label_counts[label], label))
    label_map = {old_label: new_label for new_label, old_label in enumerate(old_order)}
    remapped_labels = np.array([label_map[int(label)] for label in labels], dtype=int)
    remapped_centers = centers[old_order]
    return remapped_labels, remapped_centers


def cluster_distance_columns(
    features: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.zeros(len(features), dtype=float)
    ranks = np.zeros(len(features), dtype=int)

    for cluster_id in range(len(centers)):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_distances = np.linalg.norm(
            features[cluster_indices] - centers[cluster_id],
            axis=1,
        )
        distances[cluster_indices] = cluster_distances
        ranked_positions = np.argsort(cluster_distances)
        for rank, position in enumerate(ranked_positions, start=1):
            ranks[cluster_indices[position]] = rank

    return distances, ranks


def motif_counts(paths: list[str], motif_length: int) -> Counter[str]:
    counts: Counter[str] = Counter()
    for path in paths:
        if len(path) < motif_length:
            continue
        for offset in range(len(path) - motif_length + 1):
            counts[path[offset : offset + motif_length]] += 1
    return counts


def top_end_positions(subset: pd.DataFrame, limit: int = 5) -> list[dict[str, int]]:
    counts = (
        subset.groupby(["end_row", "end_col"])
        .size()
        .sort_values(ascending=False)
        .head(limit)
    )
    positions: list[dict[str, int]] = []
    for (end_row, end_col), count in counts.items():
        positions.append(
            {
                "end_row": int(end_row),
                "end_col": int(end_col),
                "count": int(count),
            }
        )
    return positions


def top_motifs(
    paths: list[str],
    *,
    global_counts: Counter[str],
    motif_length: int,
    min_motif_count: int,
    limit: int = 5,
) -> list[dict[str, int | float | str]]:
    local_counts = motif_counts(paths, motif_length)
    local_total = sum(local_counts.values())
    global_total = sum(global_counts.values())
    if local_total == 0 or global_total == 0:
        return []

    scored: list[tuple[float, int, str]] = []
    for motif, count in local_counts.items():
        if count < min_motif_count:
            continue
        lift = (count / local_total) / (global_counts[motif] / global_total)
        scored.append((lift, count, motif))

    scored.sort(reverse=True)
    motifs: list[dict[str, int | float | str]] = []
    for lift, count, motif in scored[:limit]:
        motifs.append(
            {
                "motif": motif,
                "count": int(count),
                "lift": round(float(lift), 4),
            }
        )
    return motifs


def exemplar_rows(
    subset: pd.DataFrame,
    *,
    examples_per_cluster: int,
) -> list[dict[str, int | float | str | dict[str, int]]]:
    ordered = subset.sort_values(["cluster_distance", "index"]).head(
        examples_per_cluster
    )
    examples: list[dict[str, int | float | str | dict[str, int]]] = []
    for row in ordered.to_dict("records"):
        examples.append(
            {
                "index": int(row["index"]),
                "seed": int(row["seed"]),
                "image": str(row["image"]),
                "distance_to_centroid": round(float(row["cluster_distance"]), 4),
                "moves": int(row["moves"]),
                "overlaps": int(row["overlaps"]),
                "end": {
                    "row": int(row["end_row"]),
                    "col": int(row["end_col"]),
                },
                "path_preview": str(row["path"])[:32],
            }
        )
    return examples


def build_summary(
    clustered: pd.DataFrame,
    *,
    dataset_path: Path,
    output_dir: Path,
    component_names: list[str],
    evaluations: list[ClusterEvaluation],
    selected_k: int,
    selection_mode: str,
    selected_silhouette: float,
    motif_length: int,
    examples_per_cluster: int,
) -> dict[str, object]:
    global_motif_counts = motif_counts(clustered["path"].tolist(), motif_length)
    clusters: list[dict[str, object]] = []
    total_rows = len(clustered)

    for cluster_id in sorted(clustered["cluster"].unique()):
        subset = clustered[clustered["cluster"] == cluster_id].copy()
        motifs = top_motifs(
            subset["path"].tolist(),
            global_counts=global_motif_counts,
            motif_length=motif_length,
            min_motif_count=max(DEFAULT_MIN_MOTIF_COUNT, len(subset) // 20),
        )
        end_positions = top_end_positions(subset)
        lead_motif = motifs[0]["motif"] if motifs else ""
        lead_end = end_positions[0] if end_positions else {"end_row": -1, "end_col": -1}

        clusters.append(
            {
                "cluster_id": int(cluster_id),
                "size": int(len(subset)),
                "share": round(float(len(subset) / total_rows), 4),
                "hint": (
                    f"motif={lead_motif}, end=({lead_end['end_row']},{lead_end['end_col']})"
                ),
                "avg_moves": round(float(subset["moves"].mean()), 3),
                "avg_overlaps": round(float(subset["overlaps"].mean()), 3),
                "avg_open_cells": round(float(subset["open_cells"].mean()), 3),
                "avg_turn_rate": round(float(subset["turn_rate"].mean()), 4),
                "avg_run_length": round(float(subset["mean_run_length"].mean()), 4),
                "avg_move_fraction": {
                    "u": round(float(subset["move_fraction_u"].mean()), 4),
                    "d": round(float(subset["move_fraction_d"].mean()), 4),
                    "l": round(float(subset["move_fraction_l"].mean()), 4),
                    "r": round(float(subset["move_fraction_r"].mean()), 4),
                },
                "top_motifs": motifs,
                "top_end_positions": end_positions,
                "examples": exemplar_rows(
                    subset,
                    examples_per_cluster=examples_per_cluster,
                ),
            }
        )

    return {
        "dataset": str(dataset_path),
        "output_dir": str(output_dir),
        "row_count": total_rows,
        "clustering_features": {
            "type": "full_path_tfidf_svd",
            "ngram_range": list(PATH_NGRAM_RANGE),
            "svd_components": len(component_names),
            "component_names": component_names,
            "motif_length_for_summary": motif_length,
        },
        "cluster_selection": {
            "mode": selection_mode,
            "selected_k": int(selected_k),
            "selected_silhouette": round(float(selected_silhouette), 4),
            "candidates": [
                {
                    "n_clusters": int(evaluation.n_clusters),
                    "silhouette": round(float(evaluation.silhouette), 4),
                }
                for evaluation in evaluations
            ],
        },
        "clusters": clusters,
    }


def cluster_optimal_paths(
    csv_path: str | Path = DEFAULT_DATASET,
    *,
    output_dir: str | Path | None = None,
    clusters: int | None = None,
    min_clusters: int = DEFAULT_MIN_CLUSTERS,
    max_clusters: int = DEFAULT_MAX_CLUSTERS,
    examples_per_cluster: int = DEFAULT_EXAMPLES_PER_CLUSTER,
    max_components: int = DEFAULT_MAX_COMPONENTS,
    motif_length: int = DEFAULT_MOTIF_LENGTH,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, object]:
    if min_clusters <= 0 or max_clusters <= 0:
        raise ValueError("Cluster bounds must be positive integers")
    if min_clusters > max_clusters:
        raise ValueError("min_clusters must be less than or equal to max_clusters")
    if clusters is not None and clusters < 2:
        raise ValueError("clusters must be at least 2")
    if examples_per_cluster <= 0:
        raise ValueError("examples_per_cluster must be a positive integer")
    if max_components < 2:
        raise ValueError("max_components must be at least 2")
    if motif_length < 2:
        raise ValueError("motif_length must be at least 2")

    dataset_path = Path(csv_path)
    resolved_output_dir = (
        Path(output_dir)
        if output_dir is not None
        else dataset_path.parent / DEFAULT_OUTPUT_DIRNAME
    )

    dataframe = load_optimal_paths(dataset_path)
    derived_features, normalized_paths = build_summary_features(dataframe)
    embedding, component_names = build_path_embedding(
        normalized_paths,
        max_components=max_components,
        random_state=random_state,
    )

    evaluations: list[ClusterEvaluation] = []
    if clusters is None:
        evaluations = evaluate_cluster_counts(
            embedding,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            random_state=random_state,
        )
        selected_k = choose_cluster_count(evaluations)
        selection_mode = "auto"
    else:
        selected_k = clusters
        selection_mode = "manual"

    if selected_k >= len(dataframe):
        raise ValueError(
            "clusters must be smaller than the number of rows in the dataset"
        )

    model = KMeans(n_clusters=selected_k, n_init=20, random_state=random_state)
    labels = model.fit_predict(embedding)
    labels, centers = relabel_clusters(labels, model.cluster_centers_)
    selected_silhouette = float(silhouette_score(embedding, labels))
    distances, ranks = cluster_distance_columns(embedding, labels, centers)

    clustered = pd.concat([dataframe, derived_features], axis=1)
    clustered["cluster"] = labels
    clustered["cluster_distance"] = distances
    clustered["cluster_rank"] = ranks

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    clustered_output = resolved_output_dir / "optimal_paths_clustered.csv"
    clustered.sort_values(["cluster", "cluster_rank", "index"]).to_csv(
        clustered_output,
        index=False,
    )

    summary = build_summary(
        clustered,
        dataset_path=dataset_path,
        output_dir=resolved_output_dir,
        component_names=component_names,
        evaluations=evaluations,
        selected_k=selected_k,
        selection_mode=selection_mode,
        selected_silhouette=selected_silhouette,
        motif_length=motif_length,
        examples_per_cluster=examples_per_cluster,
    )
    summary_output = resolved_output_dir / "summary.json"
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"Clustered {len(clustered)} optimal paths into {selected_k} groups "
        f"(silhouette={selected_silhouette:.4f})"
    )
    if evaluations:
        print("Candidate cluster counts:")
        for evaluation in evaluations:
            print(
                f"  k={evaluation.n_clusters}: silhouette={evaluation.silhouette:.4f}"
            )
    print("Top clusters:")
    for cluster in summary["clusters"]:
        cluster_id = cluster["cluster_id"]
        size = cluster["size"]
        hint = cluster["hint"]
        avg_moves = cluster["avg_moves"]
        avg_overlaps = cluster["avg_overlaps"]
        print(
            f"  cluster {cluster_id}: size={size} | moves={avg_moves} | "
            f"overlaps={avg_overlaps} | {hint}"
        )
    print(f"Wrote {clustered_output}")
    print(f"Wrote {summary_output}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Group optimal path strings into unsupervised whole-path clusters"
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to the optimal-path CSV dataset",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for clustered CSV and summary JSON outputs",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        help="Use a fixed number of clusters instead of auto-selecting by silhouette score",
    )
    parser.add_argument(
        "--min-clusters",
        type=int,
        default=DEFAULT_MIN_CLUSTERS,
        help=(
            "Minimum cluster count to test when auto-selecting "
            f"(default: {DEFAULT_MIN_CLUSTERS})"
        ),
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=DEFAULT_MAX_CLUSTERS,
        help=(
            "Maximum cluster count to test when auto-selecting "
            f"(default: {DEFAULT_MAX_CLUSTERS})"
        ),
    )
    parser.add_argument(
        "--examples-per-cluster",
        type=int,
        default=DEFAULT_EXAMPLES_PER_CLUSTER,
        help=(
            "Number of exemplar paths to include per cluster in the summary "
            f"(default: {DEFAULT_EXAMPLES_PER_CLUSTER})"
        ),
    )
    parser.add_argument(
        "--max-components",
        type=int,
        default=DEFAULT_MAX_COMPONENTS,
        help=(
            "Maximum SVD components for the whole-path embedding "
            f"(default: {DEFAULT_MAX_COMPONENTS})"
        ),
    )
    parser.add_argument(
        "--motif-length",
        type=int,
        default=DEFAULT_MOTIF_LENGTH,
        help=(
            "Substring length to use for cluster summaries "
            f"(default: {DEFAULT_MOTIF_LENGTH})"
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Random seed for clustering (default: {DEFAULT_RANDOM_STATE})",
    )
    args = parser.parse_args()

    cluster_optimal_paths(
        args.dataset,
        output_dir=args.output_dir,
        clusters=args.clusters,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
        examples_per_cluster=args.examples_per_cluster,
        max_components=args.max_components,
        motif_length=args.motif_length,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
