import argparse
import csv
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

from src.grid import create_random_grid
from src.optimal.library import render_solution_svg
from src.shared_types import MOVE_DELTAS


DEFAULT_DATASET = Path("src/optimal/library/optimal_paths.csv")
DEFAULT_HUMAN_RESULTS = Path("data/mow_test_rows.csv")
DEFAULT_OUTPUT_DIRNAME = "clusters"
DEFAULT_HUMAN_IMAGE_DIRNAME = "human_images"
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
HUMAN_CONFIG_SPECS = {
    0: {"size": 7, "seed": 59},
    1: {"size": 7, "seed": 59},
    2: {"size": 10, "seed": 59},
    3: {"size": 14, "seed": 3},
}
MOVE_CODES_BY_DELTA = {delta: move for move, delta in MOVE_DELTAS.items()}


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


def _serialize_grid(grid: list[list[int]]) -> str:
    return "/".join("".join(str(cell) for cell in row) for row in grid)


def _open_cell_count(grid: list[list[int]]) -> int:
    return sum(cell for row in grid for cell in row)


def _human_points(raw_result: str) -> list[tuple[int, int]]:
    raw_points = json.loads(raw_result)
    return [(int(point["y"]), int(point["x"])) for point in raw_points]


def _human_path_string(points: list[tuple[int, int]]) -> str:
    if len(points) < 2:
        return ""

    moves: list[str] = []
    for previous, current in zip(points, points[1:]):
        delta = (current[0] - previous[0], current[1] - previous[1])
        move = MOVE_CODES_BY_DELTA.get(delta)
        if move is None:
            raise ValueError(f"Human result contains a non-adjacent move: {delta}")
        moves.append(move)
    return "".join(moves)


def _point_overlap_count(points: list[tuple[int, int]]) -> int:
    visit_counts = Counter(points)
    return sum(count - 1 for count in visit_counts.values() if count > 1)


def _ensure_source_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    enriched = dataframe.copy()
    if "source" not in enriched.columns:
        enriched["source"] = "optimal"
    else:
        enriched["source"] = enriched["source"].fillna("optimal")

    if "source_id" not in enriched.columns:
        enriched["source_id"] = [
            f"optimal-{int(index)}" for index in enriched["index"].tolist()
        ]
    else:
        enriched["source_id"] = enriched["source_id"].fillna("")

    if "user" not in enriched.columns:
        enriched["user"] = ""
    else:
        enriched["user"] = enriched["user"].fillna("")

    if "config_id" not in enriched.columns:
        enriched["config_id"] = pd.Series([pd.NA] * len(enriched), dtype="Int64")

    if "config_label" not in enriched.columns:
        enriched["config_label"] = ""
    else:
        enriched["config_label"] = enriched["config_label"].fillna("")

    return enriched


def load_human_paths(
    csv_path: str | Path,
    *,
    asset_root: Path,
) -> pd.DataFrame:
    file_path = Path(csv_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Human results file not found: {file_path}")

    image_dir = asset_root / DEFAULT_HUMAN_IMAGE_DIRNAME
    image_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    with file_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            config_id = int(row["config"])
            spec = HUMAN_CONFIG_SPECS.get(config_id)
            if spec is None:
                raise ValueError(f"Unknown human-results config: {config_id}")

            grid = create_random_grid(spec["size"], spec["seed"])
            points = _human_points(row["result"])
            path_string = _human_path_string(points)
            image_name = f"human_config{config_id}_attempt{int(row['id']):04d}.svg"
            image_path = image_dir / image_name
            image_path.write_text(
                render_solution_svg(
                    grid,
                    {
                        "start": points[0] if points else None,
                        "moves": list(path_string),
                    },
                    title=(
                        f"Human Attempt {int(row['id']):04d} | "
                        f"config={config_id} | user={row['user']}"
                    ),
                ),
                encoding="utf-8",
            )

            rows.append(
                {
                    "index": int(row["id"]),
                    "size": spec["size"],
                    "seed": spec["seed"],
                    "open_cells": _open_cell_count(grid),
                    "start_row": -1 if not points else points[0][0],
                    "start_col": -1 if not points else points[0][1],
                    "end_row": -1 if not points else points[-1][0],
                    "end_col": -1 if not points else points[-1][1],
                    "moves": len(path_string),
                    "overlaps": _point_overlap_count(points),
                    "path": path_string,
                    "grid": _serialize_grid(grid),
                    "image": str(image_path.relative_to(asset_root)),
                    "source": "human",
                    "source_id": f"human-{int(row['id'])}",
                    "user": row["user"],
                    "config_id": config_id,
                    "config_label": f"config {config_id}",
                }
            )

    return pd.DataFrame(rows)


def load_analysis_paths(
    csv_path: str | Path,
    *,
    human_results_path: str | Path | None,
) -> tuple[pd.DataFrame, int]:
    dataset_path = Path(csv_path)
    optimal = _ensure_source_columns(load_optimal_paths(dataset_path))
    if human_results_path is None:
        return optimal, 0

    human_path = Path(human_results_path)
    if not human_path.exists():
        return optimal, 0

    human = load_human_paths(human_path, asset_root=dataset_path.parent)
    combined = pd.concat([optimal, human], ignore_index=True)
    combined["config_id"] = combined["config_id"].astype("Int64")
    return combined, len(human)


def source_counts(dataframe: pd.DataFrame) -> dict[str, int]:
    series = (
        dataframe["source"]
        if "source" in dataframe.columns
        else pd.Series(["optimal"] * len(dataframe))
    )
    return {
        str(source): int(count)
        for source, count in series.fillna("optimal").value_counts().items()
    }


def size_output_dir(root: Path, size: int) -> Path:
    return root / f"size_{size}"


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
    records = dataframe.to_dict("records")
    max_size = max(len(str(record["grid"]).split("/")) for record in records)
    rows: list[np.ndarray] = []

    for record, path in zip(records, prepared.normalized_paths, strict=True):
        grid_rows = str(record["grid"]).split("/")
        size = len(grid_rows)
        first_visit = np.full((max_size, max_size), -1.0)
        revisit = np.zeros((max_size, max_size), dtype=float)
        blocked = np.ones((max_size, max_size), dtype=float)
        blocked[:size, :size] = np.array(
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
) -> list[dict[str, object]]:
    ordered = subset.sort_values(["cluster_distance", "index"]).head(
        examples_per_cluster
    )
    examples: list[dict[str, object]] = []
    for row in ordered.to_dict("records"):
        examples.append(
            {
                "index": int(row["index"]),
                "source": str(row.get("source", "optimal")),
                "source_id": str(row.get("source_id", f"optimal-{int(row['index'])}")),
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
        if str(row.get("source", "optimal")) == "human":
            examples[-1]["user"] = str(row.get("user", ""))
            examples[-1]["config_label"] = str(row.get("config_label", ""))
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
    overall_source_counts = source_counts(clustered)
    size = int(clustered["size"].iloc[0])

    for cluster_id in sorted(clustered["cluster"].unique()):
        subset = clustered[clustered["cluster"] == cluster_id].copy()
        cluster_source_counts = source_counts(subset)
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
                "source_counts": cluster_source_counts,
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
        "size": size,
        "asset_root": str(dataset_path.parent),
        "output_dir": str(output_dir),
        "row_count": total_rows,
        "source_counts": overall_source_counts,
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


def _cluster_size_group(
    dataframe: pd.DataFrame,
    *,
    dataset_path: Path,
    output_dir: Path,
    embedding: str,
    algorithm: str,
    clusters: int | None,
    min_clusters: int,
    max_clusters: int,
    examples_per_cluster: int,
    max_components: int,
    motif_length: int,
    random_state: int,
) -> dict[str, object]:
    size = int(dataframe["size"].iloc[0])
    prepared = prepare_paths(dataframe)

    embedding_names = (
        ["tfidf_svd", "count_svd", "pattern_stats", "visit_order_svd"]
        if embedding == "auto"
        else [embedding]
    )
    algorithm_names = (
        ["kmeans", "agglomerative"] if algorithm == "auto" else [algorithm]
    )
    requested_cluster_counts = (
        [clusters]
        if clusters is not None
        else list(range(min_clusters, max_clusters + 1))
    )
    cluster_counts = [
        count for count in requested_cluster_counts if count < len(dataframe)
    ]
    if not cluster_counts:
        raise ValueError(
            f"No valid cluster counts remain for size {size}; "
            f"requested={requested_cluster_counts}, rows={len(dataframe)}"
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
        raise ValueError(
            f"No candidate clustering strategies were evaluated for size {size}"
        )

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

    output_dir.mkdir(parents=True, exist_ok=True)
    clustered_output = output_dir / "optimal_paths_clustered.csv"
    clustered.sort_values(["cluster", "cluster_rank", "index"]).to_csv(
        clustered_output,
        index=False,
    )

    summary = build_summary(
        clustered,
        dataset_path=dataset_path,
        output_dir=output_dir,
        selected_candidate=selected_candidate,
        candidates=candidates,
        motif_length=motif_length,
        examples_per_cluster=examples_per_cluster,
    )
    summary_output = output_dir / "summary.json"
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"Size {size}: embedding={selected_candidate.embedding} | "
        f"algorithm={selected_candidate.algorithm} | "
        f"k={selected_candidate.n_clusters} | "
        f"silhouette={selected_candidate.silhouette:.4f} | "
        f"davies_bouldin={selected_candidate.davies_bouldin:.4f}"
    )
    for cluster in summary["clusters"]:
        print(
            f"  cluster {cluster['cluster_id']}: size={cluster['size']} | "
            f"moves={cluster['avg_moves']} | overlaps={cluster['avg_overlaps']} | "
            f"{cluster['hint']}"
        )
    print(
        f"  sources: "
        + ", ".join(
            f"{source}={count}" for source, count in summary["source_counts"].items()
        )
    )
    print(f"  wrote {clustered_output}")
    print(f"  wrote {summary_output}")

    return {
        "size": size,
        "row_count": int(len(dataframe)),
        "source_counts": summary["source_counts"],
        "clustered_csv": clustered_output,
        "summary_json": summary_output,
        "selected_strategy": summary["clustering_strategy"],
    }


def cluster_optimal_paths(
    csv_path: str | Path = DEFAULT_DATASET,
    *,
    output_dir: str | Path | None = None,
    human_results_path: str | Path | None = DEFAULT_HUMAN_RESULTS,
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

    dataframe, human_row_count = load_analysis_paths(
        dataset_path,
        human_results_path=human_results_path,
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    size_results: list[dict[str, object]] = []
    for size in sorted(int(value) for value in dataframe["size"].unique()):
        subset = dataframe[dataframe["size"] == size].copy().reset_index(drop=True)
        size_results.append(
            _cluster_size_group(
                subset,
                dataset_path=dataset_path,
                output_dir=size_output_dir(resolved_output_dir, size),
                embedding=embedding,
                algorithm=algorithm,
                clusters=clusters,
                min_clusters=min_clusters,
                max_clusters=max_clusters,
                examples_per_cluster=examples_per_cluster,
                max_components=max_components,
                motif_length=motif_length,
                random_state=random_state,
            )
        )

    summary = {
        "dataset": str(dataset_path),
        "output_dir": str(resolved_output_dir),
        "grouped_by": "size",
        "row_count": int(len(dataframe)),
        "source_counts": source_counts(dataframe),
        "sizes": [
            {
                "size": int(result["size"]),
                "row_count": int(result["row_count"]),
                "source_counts": result["source_counts"],
                "clustered_csv": Path(result["clustered_csv"])
                .relative_to(resolved_output_dir)
                .as_posix(),
                "summary_json": Path(result["summary_json"])
                .relative_to(resolved_output_dir)
                .as_posix(),
                "selected_strategy": result["selected_strategy"],
            }
            for result in size_results
        ],
    }
    summary_output = resolved_output_dir / "summary.json"
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        "Dataset composition: "
        + ", ".join(
            f"{source}={count}" for source, count in summary["source_counts"].items()
        )
    )
    if human_results_path is not None and human_row_count == 0:
        print(f"Human results not included (missing file): {human_results_path}")
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
        "--human-results",
        default=str(DEFAULT_HUMAN_RESULTS),
        help=(
            "Optional human-results CSV to append to the clustering dataset "
            f"(default: {DEFAULT_HUMAN_RESULTS})"
        ),
    )
    parser.add_argument(
        "--skip-human-results",
        action="store_true",
        help="Disable loading human solutions from the CSV",
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
        human_results_path=None if args.skip_human_results else args.human_results,
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
