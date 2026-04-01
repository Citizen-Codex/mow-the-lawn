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
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.shared_types import MOVE_DELTAS


DEFAULT_DATASET = Path("src/optimal/library/optimal_paths.csv")
DEFAULT_OUTPUT_DIRNAME = "clusters"
DEFAULT_MIN_CLUSTERS = 3
DEFAULT_MAX_CLUSTERS = 8
DEFAULT_EXAMPLES_PER_CLUSTER = 5
DEFAULT_RANDOM_STATE = 42
DEFAULT_MAX_COMPONENTS = 8
DEFAULT_MOTIF_LENGTH = 5
DEFAULT_MIN_MOTIF_COUNT = 8
DEFAULT_EMBEDDING = "auto"
DEFAULT_ALGORITHM = "auto"
TFIDF_NGRAM_RANGE = (3, 8)
COUNT_NGRAM_RANGE = (2, 5)
VALID_PATH_CHARS = frozenset("udlr")
EMBEDDING_CHOICES = (
    "auto",
    "tfidf_svd",
    "count_svd",
    "pattern_stats",
    "visit_order_svd",
)
ALGORITHM_CHOICES = ("auto", "kmeans", "agglomerative")
EMBEDDING_PREFERENCE = {
    "count_svd": 4,
    "tfidf_svd": 3,
    "pattern_stats": 2,
    "visit_order_svd": 1,
}
ALGORITHM_PREFERENCE = {"kmeans": 2, "agglomerative": 1}
PATTERN_STAT_COLUMNS = [
    "moves",
    "overlaps",
    "open_cells",
    "end_row",
    "end_col",
    "move_fraction_u",
    "move_fraction_d",
    "move_fraction_l",
    "move_fraction_r",
    "turn_rate",
    "mean_run_length",
    "max_run_length",
    "min_run_length",
    "run_length_std",
]
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
class PreparedPaths:
    normalized_paths: list[str]
    derived_features: pd.DataFrame


@dataclass(frozen=True)
class EmbeddingResult:
    name: str
    features: np.ndarray
    metadata: dict[str, object]


@dataclass(frozen=True)
class CandidateResult:
    embedding: str
    algorithm: str
    n_clusters: int
    silhouette: float
    davies_bouldin: float
    feature_metadata: dict[str, object]


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


def trace_path_points(
    start_row: int, start_col: int, path: str
) -> list[tuple[int, int]]:
    row = start_row
    col = start_col
    points = [(row, col)]
    for move in path:
        delta_row, delta_col = MOVE_DELTAS[move]
        row += delta_row
        col += delta_col
        points.append((row, col))
    return points


def prepare_paths(dataframe: pd.DataFrame) -> PreparedPaths:
    normalized_paths: list[str] = []
    derived_rows: list[dict[str, float]] = []

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

    return PreparedPaths(
        normalized_paths=normalized_paths,
        derived_features=pd.DataFrame(derived_rows),
    )


def _build_ngram_embedding(
    paths: list[str],
    *,
    vectorizer_kind: str,
    ngram_range: tuple[int, int],
    max_components: int,
    random_state: int,
) -> tuple[np.ndarray, dict[str, object]]:
    vectorizer_cls = TfidfVectorizer if vectorizer_kind == "tfidf" else CountVectorizer
    vectorizer = vectorizer_cls(
        analyzer="char",
        ngram_range=ngram_range,
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
        n_components=n_components,
        random_state=random_state,
    ).fit_transform(path_matrix)
    return embedding, {
        "type": f"whole_path_{vectorizer_kind}_svd",
        "vectorizer": vectorizer_kind,
        "ngram_range": list(ngram_range),
        "svd_components": int(embedding.shape[1]),
        "component_names": [
            f"svd_component_{index}" for index in range(embedding.shape[1])
        ],
    }


def _build_pattern_stats_embedding(
    dataframe: pd.DataFrame,
    prepared: PreparedPaths,
) -> tuple[np.ndarray, dict[str, object]]:
    numeric_frame = pd.concat(
        [
            dataframe[["moves", "overlaps", "open_cells", "end_row", "end_col"]],
            prepared.derived_features,
        ],
        axis=1,
    )
    embedding = StandardScaler().fit_transform(numeric_frame[PATTERN_STAT_COLUMNS])
    return embedding, {
        "type": "pattern_stats",
        "feature_columns": PATTERN_STAT_COLUMNS,
        "svd_components": None,
        "component_names": PATTERN_STAT_COLUMNS,
    }


def _build_visit_order_embedding(
    dataframe: pd.DataFrame,
    prepared: PreparedPaths,
    *,
    max_components: int,
    random_state: int,
) -> tuple[np.ndarray, dict[str, object]]:
    rows: list[np.ndarray] = []

    for record, path in zip(
        dataframe.to_dict("records"), prepared.normalized_paths, strict=True
    ):
        grid_rows = str(record["grid"]).split("/")
        size = len(grid_rows)
        first_visit = np.full((size, size), -1.0)
        revisit = np.zeros((size, size), dtype=float)
        blocked = np.array(
            [
                [1.0 if cell == "0" else 0.0 for cell in grid_row]
                for grid_row in grid_rows
            ],
            dtype=float,
        )
        points = trace_path_points(
            int(record["start_row"]), int(record["start_col"]), path
        )
        total_steps = max(len(points) - 1, 1)
        visit_counts = Counter(points)
        for step, (row, col) in enumerate(points):
            if first_visit[row, col] < 0:
                first_visit[row, col] = step / total_steps
        for (row, col), count in visit_counts.items():
            revisit[row, col] = max(count - 1, 0)

        rows.append(
            np.concatenate(
                [
                    first_visit.ravel(),
                    revisit.ravel(),
                    blocked.ravel(),
                ]
            )
        )

    raw = StandardScaler().fit_transform(np.vstack(rows))
    n_components = min(max_components, raw.shape[0] - 1, raw.shape[1] - 1)
    if n_components < 2:
        raise ValueError("Unable to derive enough visit-order embedding components")

    embedding = TruncatedSVD(
        n_components=n_components,
        random_state=random_state,
    ).fit_transform(raw)
    return embedding, {
        "type": "visit_order_svd",
        "feature_blocks": ["first_visit_order", "revisit_count", "blocked_mask"],
        "svd_components": int(embedding.shape[1]),
        "component_names": [
            f"visit_component_{index}" for index in range(embedding.shape[1])
        ],
    }


def build_embedding(
    dataframe: pd.DataFrame,
    *,
    embedding_name: str,
    max_components: int = DEFAULT_MAX_COMPONENTS,
    random_state: int = DEFAULT_RANDOM_STATE,
    prepared: PreparedPaths | None = None,
) -> EmbeddingResult:
    if embedding_name not in EMBEDDING_CHOICES or embedding_name == "auto":
        raise ValueError(f"Unsupported embedding: {embedding_name}")

    prepared_paths = prepared if prepared is not None else prepare_paths(dataframe)

    if embedding_name == "tfidf_svd":
        features, metadata = _build_ngram_embedding(
            prepared_paths.normalized_paths,
            vectorizer_kind="tfidf",
            ngram_range=TFIDF_NGRAM_RANGE,
            max_components=max_components,
            random_state=random_state,
        )
    elif embedding_name == "count_svd":
        features, metadata = _build_ngram_embedding(
            prepared_paths.normalized_paths,
            vectorizer_kind="count",
            ngram_range=COUNT_NGRAM_RANGE,
            max_components=max_components,
            random_state=random_state,
        )
    elif embedding_name == "pattern_stats":
        features, metadata = _build_pattern_stats_embedding(dataframe, prepared_paths)
    else:
        features, metadata = _build_visit_order_embedding(
            dataframe,
            prepared_paths,
            max_components=max_components,
            random_state=random_state,
        )

    return EmbeddingResult(name=embedding_name, features=features, metadata=metadata)


def fit_cluster_model(
    features: np.ndarray,
    *,
    algorithm: str,
    n_clusters: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if algorithm == "kmeans":
        model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
        labels = model.fit_predict(features)
        centers = model.cluster_centers_
        return labels.astype(int), centers

    if algorithm == "agglomerative":
        labels = AgglomerativeClustering(
            n_clusters=n_clusters, linkage="ward"
        ).fit_predict(features)
        centers = np.vstack(
            [
                features[labels == cluster_id].mean(axis=0)
                for cluster_id in range(n_clusters)
            ]
        )
        return labels.astype(int), centers

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def candidate_sort_key(
    candidate: CandidateResult,
) -> tuple[float, float, int, int, int]:
    return (
        candidate.silhouette,
        -candidate.davies_bouldin,
        candidate.n_clusters,
        EMBEDDING_PREFERENCE.get(candidate.embedding, 0),
        ALGORITHM_PREFERENCE.get(candidate.algorithm, 0),
    )


def search_candidates(
    embeddings: dict[str, EmbeddingResult],
    *,
    algorithms: list[str],
    cluster_counts: list[int],
    random_state: int,
) -> list[CandidateResult]:
    candidates: list[CandidateResult] = []
    for embedding_name, embedding in embeddings.items():
        for algorithm in algorithms:
            for n_clusters in cluster_counts:
                labels, _ = fit_cluster_model(
                    embedding.features,
                    algorithm=algorithm,
                    n_clusters=n_clusters,
                    random_state=random_state,
                )
                candidates.append(
                    CandidateResult(
                        embedding=embedding_name,
                        algorithm=algorithm,
                        n_clusters=n_clusters,
                        silhouette=float(silhouette_score(embedding.features, labels)),
                        davies_bouldin=float(
                            davies_bouldin_score(embedding.features, labels)
                        ),
                        feature_metadata=embedding.metadata,
                    )
                )
    return sorted(candidates, key=candidate_sort_key, reverse=True)


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
    selected_candidate: CandidateResult,
    candidates: list[CandidateResult],
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
        "clustering_strategy": {
            "embedding": selected_candidate.embedding,
            "algorithm": selected_candidate.algorithm,
            "selected_k": int(selected_candidate.n_clusters),
            "silhouette": round(float(selected_candidate.silhouette), 4),
            "davies_bouldin": round(float(selected_candidate.davies_bouldin), 4),
        },
        "clustering_features": {
            **selected_candidate.feature_metadata,
            "motif_length_for_summary": motif_length,
        },
        "strategy_candidates": [
            {
                "embedding": candidate.embedding,
                "algorithm": candidate.algorithm,
                "n_clusters": int(candidate.n_clusters),
                "silhouette": round(float(candidate.silhouette), 4),
                "davies_bouldin": round(float(candidate.davies_bouldin), 4),
            }
            for candidate in candidates
        ],
        "clusters": clusters,
    }


def cluster_optimal_paths(
    csv_path: str | Path = DEFAULT_DATASET,
    *,
    output_dir: str | Path | None = None,
    embedding: str = DEFAULT_EMBEDDING,
    algorithm: str = DEFAULT_ALGORITHM,
    clusters: int | None = None,
    min_clusters: int = DEFAULT_MIN_CLUSTERS,
    max_clusters: int = DEFAULT_MAX_CLUSTERS,
    examples_per_cluster: int = DEFAULT_EXAMPLES_PER_CLUSTER,
    max_components: int = DEFAULT_MAX_COMPONENTS,
    motif_length: int = DEFAULT_MOTIF_LENGTH,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, object]:
    if embedding not in EMBEDDING_CHOICES:
        raise ValueError(f"embedding must be one of: {', '.join(EMBEDDING_CHOICES)}")
    if algorithm not in ALGORITHM_CHOICES:
        raise ValueError(f"algorithm must be one of: {', '.join(ALGORITHM_CHOICES)}")
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
    prepared = prepare_paths(dataframe)

    embedding_names = (
        [
            "tfidf_svd",
            "count_svd",
            "pattern_stats",
            "visit_order_svd",
        ]
        if embedding == "auto"
        else [embedding]
    )
    algorithm_names = (
        ["kmeans", "agglomerative"] if algorithm == "auto" else [algorithm]
    )
    cluster_counts = (
        [clusters]
        if clusters is not None
        else list(range(min_clusters, max_clusters + 1))
    )

    if len(dataframe) <= max(cluster_counts):
        raise ValueError(
            "Requested cluster count must be smaller than the number of rows in the dataset"
        )

    embeddings = {
        embedding_name: build_embedding(
            dataframe,
            embedding_name=embedding_name,
            max_components=max_components,
            random_state=random_state,
            prepared=prepared,
        )
        for embedding_name in embedding_names
    }
    candidates = search_candidates(
        embeddings,
        algorithms=algorithm_names,
        cluster_counts=cluster_counts,
        random_state=random_state,
    )
    if not candidates:
        raise ValueError("No candidate clustering strategies were evaluated")

    selected_candidate = candidates[0]
    selected_embedding = embeddings[selected_candidate.embedding]
    labels, centers = fit_cluster_model(
        selected_embedding.features,
        algorithm=selected_candidate.algorithm,
        n_clusters=selected_candidate.n_clusters,
        random_state=random_state,
    )
    labels, centers = relabel_clusters(labels, centers)
    distances, ranks = cluster_distance_columns(
        selected_embedding.features, labels, centers
    )

    clustered = pd.concat([dataframe, prepared.derived_features], axis=1)
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
        selected_candidate=selected_candidate,
        candidates=candidates,
        motif_length=motif_length,
        examples_per_cluster=examples_per_cluster,
    )
    summary_output = resolved_output_dir / "summary.json"
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        "Selected strategy: "
        f"embedding={selected_candidate.embedding} | "
        f"algorithm={selected_candidate.algorithm} | "
        f"k={selected_candidate.n_clusters} | "
        f"silhouette={selected_candidate.silhouette:.4f} | "
        f"davies_bouldin={selected_candidate.davies_bouldin:.4f}"
    )
    print("Top candidate strategies:")
    for candidate in candidates[:10]:
        print(
            f"  {candidate.embedding} + {candidate.algorithm}: "
            f"k={candidate.n_clusters} | silhouette={candidate.silhouette:.4f} | "
            f"davies_bouldin={candidate.davies_bouldin:.4f}"
        )
    print("Top clusters:")
    for cluster in summary["clusters"]:
        print(
            f"  cluster {cluster['cluster_id']}: size={cluster['size']} | "
            f"moves={cluster['avg_moves']} | overlaps={cluster['avg_overlaps']} | "
            f"{cluster['hint']}"
        )
    print(f"Wrote {clustered_output}")
    print(f"Wrote {summary_output}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Group optimal path strings into unsupervised clusters while comparing "
            "alternative embeddings and clustering algorithms"
        )
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
        "--embedding",
        choices=EMBEDDING_CHOICES,
        default=DEFAULT_EMBEDDING,
        help=(
            "Embedding strategy to use. 'auto' compares multiple approaches "
            f"(default: {DEFAULT_EMBEDDING})"
        ),
    )
    parser.add_argument(
        "--algorithm",
        choices=ALGORITHM_CHOICES,
        default=DEFAULT_ALGORITHM,
        help=(
            "Clustering algorithm to use. 'auto' compares supported algorithms "
            f"(default: {DEFAULT_ALGORITHM})"
        ),
    )
    parser.add_argument(
        "--clusters",
        type=int,
        help="Use a fixed number of clusters instead of searching a range",
    )
    parser.add_argument(
        "--min-clusters",
        type=int,
        default=DEFAULT_MIN_CLUSTERS,
        help=(
            "Minimum cluster count to test when searching "
            f"(default: {DEFAULT_MIN_CLUSTERS})"
        ),
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=DEFAULT_MAX_CLUSTERS,
        help=(
            "Maximum cluster count to test when searching "
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
            "Maximum SVD components for the sequence and visit-order embeddings "
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
        embedding=args.embedding,
        algorithm=args.algorithm,
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
