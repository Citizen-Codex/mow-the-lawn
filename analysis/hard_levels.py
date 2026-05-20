"""Search for hard levels by sweeping the grid generator and ranking by complexity.

Two-stage pipeline:
  1. Cheap pass: generate many candidates across size/regime/seed, solve with
     concorde + snake + spiral, compute structural and path metrics, then rank.
  2. Expensive pass: on the top-N from stage 1, run an edge-forbidding probe
     against concorde to estimate how rigid the optimum is. Re-rank with this
     factor folded in and surface the final top 10.

Outputs:
  - analysis/hard_levels_candidates.csv : every candidate with all metrics
  - analysis/hard_levels_top/*.svg      : SVG renderings of the surfaced 10
  - stdout                              : ranked summary with reproduction params
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.concorde.backend import solve_symmetric_tsp
from src.concorde.metric_closure import build_metric_closure
from src.concorde.solver import concorde_solver
from src.concorde.tsp_path import build_tsp_path_instance, extract_path_from_tour
from src.grid import create_random_grid
from src.optimal.library import render_solution_svg
from src.shared_types import MOVE_DELTAS, Grid, Path as SolverPath
from src.solvers import find_start, snake_solver, spiral_solver


REGIMES: dict[str, dict] = {
    "default": {
        "cluster_count_range": None,
        "cluster_size_range": None,
        "removed_fraction_range": (0.20, 0.40),
    },
    "dispersed": {
        "cluster_count_range": (12, 25),
        "cluster_size_range": (1, 2),
        "removed_fraction_range": (0.18, 0.28),
    },
    "medium_clusters": {
        "cluster_count_range": (3, 4),
        "cluster_size_range": (6, 12),
        "removed_fraction_range": (0.30, 0.40),
    },
    "single_mass": {
        "cluster_count_range": (1, 2),
        "cluster_size_range": (25, 60),
        "removed_fraction_range": (0.25, 0.35),
    },
}
SIZES = (8, 10, 12, 14)
OUTPUT_DIR = Path(__file__).resolve().parent
CSV_PATH = OUTPUT_DIR / "hard_levels_candidates.csv"
TOP_DIR = OUTPUT_DIR / "hard_levels_top"


@dataclass
class Candidate:
    regime: str
    size: int
    seed: int
    cluster_count_range: tuple[int, int]
    cluster_size_range: tuple[int, int]
    removed_fraction_range: tuple[float, float]
    grid: Grid
    optimal_path: SolverPath
    snake_moves: int
    spiral_moves: int
    open_cells: int
    metrics: dict[str, float] = field(default_factory=dict)


def _path_moves(path: SolverPath) -> list[str]:
    return list(path["moves"])


def _trace_points(path: SolverPath) -> list[tuple[int, int]]:
    if path["start"] is None:
        return []
    points = [path["start"]]
    x, y = path["start"]
    for move in path["moves"]:
        dx, dy = MOVE_DELTAS[move]  # type: ignore[index]
        x, y = x + dx, y + dy
        points.append((x, y))
    return points


def _overlap_count(path: SolverPath) -> int:
    counts = Counter(_trace_points(path))
    return sum(c - 1 for c in counts.values() if c > 1)


def _turn_count(moves: list[str]) -> int:
    return sum(1 for a, b in zip(moves, moves[1:]) if a != b)


def _move_entropy(moves: list[str]) -> float:
    if not moves:
        return 0.0
    counts = Counter(moves)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def _spatial_turn_entropy(
    points: list[tuple[int, int]], moves: list[str], n: int, bins: int = 3
) -> float:
    if not moves:
        return 0.0
    bucket_counts: dict[tuple[int, int], int] = defaultdict(int)
    for idx in range(1, len(moves)):
        if moves[idx] == moves[idx - 1]:
            continue
        # Place the turn at the cell where it occurs (the i-th point).
        row, col = points[idx]
        br = min(bins - 1, (row * bins) // n)
        bc = min(bins - 1, (col * bins) // n)
        bucket_counts[(br, bc)] += 1
    if not bucket_counts:
        return 0.0
    total = sum(bucket_counts.values())
    probs = [c / total for c in bucket_counts.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def _open_cell_graph(grid: Grid) -> dict[tuple[int, int], list[tuple[int, int]]]:
    n = len(grid)
    adj: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for r in range(n):
        for c in range(n):
            if grid[r][c] != 1:
                continue
            for dr, dc in MOVE_DELTAS.values():
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 1:
                    adj[(r, c)].append((nr, nc))
    return adj


def _bridges_and_articulations(
    adj: dict[tuple[int, int], list[tuple[int, int]]],
) -> tuple[int, int]:
    """Iterative Tarjan: returns (n_bridges, n_articulation_points)."""
    disc: dict[tuple[int, int], int] = {}
    low: dict[tuple[int, int], int] = {}
    articulation: set[tuple[int, int]] = set()
    bridges = 0
    timer = 0

    for root in list(adj.keys()):
        if root in disc:
            continue
        # iterative DFS with parent + iterator state
        stack: list[tuple[tuple[int, int], tuple[int, int] | None, int]] = [
            (root, None, 0)
        ]
        root_children = 0
        while stack:
            node, parent, ni = stack[-1]
            if ni == 0:
                disc[node] = low[node] = timer
                timer += 1
            neighbors = adj[node]
            if ni < len(neighbors):
                stack[-1] = (node, parent, ni + 1)
                nb = neighbors[ni]
                if nb == parent:
                    continue
                if nb not in disc:
                    if node == root:
                        root_children += 1
                    stack.append((nb, node, 0))
                else:
                    low[node] = min(low[node], disc[nb])
            else:
                stack.pop()
                if parent is not None:
                    low[parent] = min(low[parent], low[node])
                    if low[node] > disc[parent]:
                        bridges += 1
                    if low[node] >= disc[parent] and parent != root:
                        articulation.add(parent)
        if root_children > 1:
            articulation.add(root)

    return bridges, len(articulation)


def _dead_ends_and_choice_points(
    adj: dict[tuple[int, int], list[tuple[int, int]]],
) -> tuple[int, int]:
    dead = sum(1 for v in adj.values() if len(v) == 1)
    choice = sum(1 for v in adj.values() if len(v) >= 3)
    return dead, choice


def _compute_cheap_metrics(cand: Candidate) -> None:
    moves = _path_moves(cand.optimal_path)
    points = _trace_points(cand.optimal_path)
    overlap = _overlap_count(cand.optimal_path)
    turns = _turn_count(moves)
    best_heuristic = min(cand.snake_moves, cand.spiral_moves)
    heuristic_gap = (
        (best_heuristic - len(moves)) / len(moves) if moves else 0.0
    )
    adj = _open_cell_graph(cand.grid)
    bridges, articulations = _bridges_and_articulations(adj)
    dead_ends, choice_points = _dead_ends_and_choice_points(adj)

    cand.metrics.update(
        {
            "optimal_moves": float(len(moves)),
            "optimal_overlap": float(overlap),
            "snake_moves": float(cand.snake_moves),
            "spiral_moves": float(cand.spiral_moves),
            "heuristic_gap": float(heuristic_gap),
            "turn_count": float(turns),
            "turn_density": float(turns / len(moves)) if moves else 0.0,
            "move_entropy": _move_entropy(moves),
            "spatial_turn_entropy": _spatial_turn_entropy(
                points, moves, cand.size
            ),
            "bridges": float(bridges),
            "articulations": float(articulations),
            "dead_ends": float(dead_ends),
            "choice_points": float(choice_points),
            "open_cells": float(cand.open_cells),
        }
    )


def _alt_optima_fraction(
    grid: Grid, *, samples: int = 8, rng: random.Random
) -> tuple[float, int, int]:
    """Probe `samples` random edges of the optimum tour by forbidding each in
    turn and re-solving. Return (alt_fraction, alt_count, probed)."""
    start = find_start(grid)
    if start is None:
        return 0.0, 0, 0

    mc = build_metric_closure(grid)
    start_index = mc.index_by_point[start]
    reachable = sorted(mc.parents[start_index].keys(), key=lambda p: mc.index_by_point[p])
    reachable_indices = [mc.index_by_point[p] for p in reachable]
    if len(reachable_indices) <= 2:
        return 0.0, 0, 0

    reduced_dist = mc.dist[np.ix_(reachable_indices, reachable_indices)]
    reduced_start = reachable_indices.index(start_index)

    instance = build_tsp_path_instance(reduced_dist, start_index=reduced_start)
    base_tour = solve_symmetric_tsp(instance.matrix)
    base_cost = _tour_cost(base_tour, instance.matrix)

    # Edges of the TSP-Path tour that connect two real (non-v_star) cities.
    # We forbid these one at a time by inflating the corresponding entries
    # of `reduced_dist` and rebuilding the TSP-Path instance.
    n_reduced = reduced_dist.shape[0]
    v_star = instance.v_star_index
    candidate_edges: list[tuple[int, int]] = []
    for a, b in zip(base_tour, base_tour[1:] + base_tour[:1]):
        if a == v_star or b == v_star:
            continue
        if a < n_reduced and b < n_reduced:
            candidate_edges.append((min(a, b), max(a, b)))

    if not candidate_edges:
        return 0.0, 0, 0

    probe_edges = candidate_edges if len(candidate_edges) <= samples else rng.sample(
        candidate_edges, samples
    )

    forbid_value = max(int(reduced_dist.max()) * (n_reduced + 1), 10**6)
    alt_count = 0
    for u, v in probe_edges:
        perturbed = reduced_dist.copy()
        perturbed[u, v] = forbid_value
        perturbed[v, u] = forbid_value
        try:
            alt_instance = build_tsp_path_instance(perturbed, start_index=reduced_start)
            alt_tour = solve_symmetric_tsp(alt_instance.matrix)
            alt_cost = _tour_cost(alt_tour, alt_instance.matrix)
            # alt_cost subtracts the (potentially inflated) forbid edge — compare
            # via re-running cost only over the original reduced_dist of the
            # alt tour's reduced path.
            alt_reduced_path = extract_path_from_tour(alt_tour, alt_instance)
            alt_real_cost = sum(
                int(reduced_dist[alt_reduced_path[i], alt_reduced_path[i + 1]])
                for i in range(len(alt_reduced_path) - 1)
            )
            base_reduced_path = extract_path_from_tour(base_tour, instance)
            base_real_cost = sum(
                int(reduced_dist[base_reduced_path[i], base_reduced_path[i + 1]])
                for i in range(len(base_reduced_path) - 1)
            )
            if alt_real_cost == base_real_cost:
                alt_count += 1
        except Exception:
            # If concorde can't find a tour with that edge forbidden, treat
            # the edge as essential (no alternative optimum).
            continue

    return alt_count / len(probe_edges), alt_count, len(probe_edges)


def _tour_cost(tour: list[int], matrix: np.ndarray) -> int:
    cost = 0
    for i in range(len(tour)):
        a = tour[i]
        b = tour[(i + 1) % len(tour)]
        cost += int(matrix[a, b])
    return cost


def _percentile_rank(values: list[float]) -> list[float]:
    """Return per-value percentile in [0, 1] with average-tie handling."""
    if not values:
        return []
    paired = sorted(enumerate(values), key=lambda kv: kv[1])
    ranks = [0.0] * len(values)
    n = len(values)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and paired[j + 1][1] == paired[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0
        for k in range(i, j + 1):
            orig_idx = paired[k][0]
            ranks[orig_idx] = avg_rank / max(1, n - 1)
        i = j + 1
    return ranks


def _composite_score(
    candidates: list[Candidate],
    *,
    include_uniqueness: bool,
) -> list[float]:
    # optimal_overlap is no longer a component: candidates are pre-filtered to
    # a low overlap ceiling, so it's near-constant across survivors and adds
    # only noise to the ranking.
    weights = {
        "heuristic_gap": 0.30,
        "turn_density": 0.20,
        "spatial_turn_entropy": 0.15,
        "bridges": 0.10,
        "choice_points": 0.10,
    }
    if include_uniqueness:
        weights["uniqueness"] = 0.15

    component_ranks: dict[str, list[float]] = {}
    for name in weights:
        if name == "uniqueness":
            alt_ranks = _percentile_rank(
                [c.metrics.get("alt_optima_fraction", 1.0) for c in candidates]
            )
            component_ranks[name] = [1.0 - r for r in alt_ranks]
        else:
            component_ranks[name] = _percentile_rank(
                [c.metrics.get(name, 0.0) for c in candidates]
            )

    total_weight = sum(weights.values())
    scores = []
    for idx in range(len(candidates)):
        s = sum(component_ranks[k][idx] * w for k, w in weights.items())
        scores.append(s / total_weight)
    return scores


def generate_candidates(seeds_per_combo: int) -> list[Candidate]:
    candidates: list[Candidate] = []
    for size in SIZES:
        for regime_name, regime in REGIMES.items():
            count_range = regime["cluster_count_range"]
            size_range = regime["cluster_size_range"]
            removed_range = regime["removed_fraction_range"]
            for seed in range(seeds_per_combo):
                grid = create_random_grid(
                    size,
                    seed,
                    removed_fraction_range=removed_range,
                    cluster_count_range=count_range,
                    cluster_size_range=size_range,
                )
                open_cells = sum(c for row in grid for c in row)
                if open_cells < 4:
                    continue
                resolved_count = count_range or (max(1, size // 3), max(2, size // 2))
                resolved_size = size_range or (max(2, size // 2), max(3, size))
                cand = Candidate(
                    regime=regime_name,
                    size=size,
                    seed=seed,
                    cluster_count_range=resolved_count,
                    cluster_size_range=resolved_size,
                    removed_fraction_range=removed_range,
                    grid=grid,
                    optimal_path={"start": None, "moves": []},
                    snake_moves=0,
                    spiral_moves=0,
                    open_cells=open_cells,
                )
                candidates.append(cand)
    return candidates


def solve_candidate(cand: Candidate) -> bool:
    """Solve a candidate. Returns False if concorde failed and the candidate
    should be dropped from analysis."""
    try:
        cand.optimal_path = concorde_solver(cand.grid)
    except Exception:
        return False
    snake = snake_solver(cand.grid, move_strategy="least_overlap")
    spiral = spiral_solver(cand.grid, move_strategy="least_overlap")
    cand.snake_moves = len(snake["moves"])
    cand.spiral_moves = len(spiral["moves"])
    if not cand.optimal_path["moves"]:
        return False
    return True


def write_csv(candidates: list[Candidate], scores: list[float], path: Path) -> None:
    fieldnames = [
        "rank",
        "score",
        "regime",
        "size",
        "seed",
        "cluster_count_min",
        "cluster_count_max",
        "cluster_size_min",
        "cluster_size_max",
        "removed_min",
        "removed_max",
        "open_cells",
        "optimal_moves",
        "optimal_overlap",
        "snake_moves",
        "spiral_moves",
        "heuristic_gap",
        "turn_count",
        "turn_density",
        "move_entropy",
        "spatial_turn_entropy",
        "bridges",
        "articulations",
        "dead_ends",
        "choice_points",
        "alt_optima_fraction",
        "alt_optima_probed",
    ]
    indexed = sorted(zip(candidates, scores), key=lambda kv: -kv[1])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, (cand, score) in enumerate(indexed, start=1):
            m = cand.metrics
            writer.writerow(
                {
                    "rank": rank,
                    "score": f"{score:.4f}",
                    "regime": cand.regime,
                    "size": cand.size,
                    "seed": cand.seed,
                    "cluster_count_min": cand.cluster_count_range[0],
                    "cluster_count_max": cand.cluster_count_range[1],
                    "cluster_size_min": cand.cluster_size_range[0],
                    "cluster_size_max": cand.cluster_size_range[1],
                    "removed_min": cand.removed_fraction_range[0],
                    "removed_max": cand.removed_fraction_range[1],
                    "open_cells": int(m.get("open_cells", 0)),
                    "optimal_moves": int(m.get("optimal_moves", 0)),
                    "optimal_overlap": int(m.get("optimal_overlap", 0)),
                    "snake_moves": int(m.get("snake_moves", 0)),
                    "spiral_moves": int(m.get("spiral_moves", 0)),
                    "heuristic_gap": f"{m.get('heuristic_gap', 0.0):.3f}",
                    "turn_count": int(m.get("turn_count", 0)),
                    "turn_density": f"{m.get('turn_density', 0.0):.3f}",
                    "move_entropy": f"{m.get('move_entropy', 0.0):.3f}",
                    "spatial_turn_entropy": f"{m.get('spatial_turn_entropy', 0.0):.3f}",
                    "bridges": int(m.get("bridges", 0)),
                    "articulations": int(m.get("articulations", 0)),
                    "dead_ends": int(m.get("dead_ends", 0)),
                    "choice_points": int(m.get("choice_points", 0)),
                    "alt_optima_fraction": (
                        f"{m['alt_optima_fraction']:.3f}"
                        if "alt_optima_fraction" in m
                        else ""
                    ),
                    "alt_optima_probed": int(m.get("alt_optima_probed", 0)),
                }
            )


def render_top(candidates: list[Candidate], scores: list[float], top_n: int) -> None:
    TOP_DIR.mkdir(parents=True, exist_ok=True)
    for existing in TOP_DIR.glob("*.svg"):
        existing.unlink()
    indexed = sorted(zip(candidates, scores), key=lambda kv: -kv[1])[:top_n]
    for rank, (cand, score) in enumerate(indexed, start=1):
        title = (
            f"#{rank} {cand.regime} {cand.size}x{cand.size} seed={cand.seed} "
            f"score={score:.3f}"
        )
        filename = (
            f"top_{rank:02d}_{cand.regime}_{cand.size}x{cand.size}_seed{cand.seed}.svg"
        )
        (TOP_DIR / filename).write_text(
            render_solution_svg(cand.grid, cand.optimal_path, title=title)
        )


def _format_row(rank: int, cand: Candidate, score: float) -> str:
    m = cand.metrics
    alt = m.get("alt_optima_fraction")
    alt_str = f"{alt * 100:5.1f}" if alt is not None else "   -- "
    return (
        f"{rank:>4} {cand.regime:>15} {cand.size:>4} {cand.seed:>5} "
        f"{score:>6.3f} {int(m['optimal_moves']):>5} {int(m['optimal_overlap']):>4} "
        f"{m['heuristic_gap']:>5.2f} {m['turn_density'] * 100:>5.1f} "
        f"{int(m['bridges']):>6} {int(m['choice_points']):>6} {alt_str}"
    )


def print_summary(candidates: list[Candidate], scores: list[float], top_n: int) -> None:
    indexed = sorted(zip(candidates, scores), key=lambda kv: -kv[1])
    header = (
        f"{'rank':>4} {'regime':>15} {'size':>4} {'seed':>5} {'score':>6} "
        f"{'moves':>5} {'over':>4} {'gap':>5} {'turn%':>6} {'bridge':>6} "
        f"{'choice':>6} {'alt%':>5}"
    )

    print()
    print(f"Top {top_n} hard-level candidates (overall):")
    print("-" * 110)
    print(header)
    for rank, (cand, score) in enumerate(indexed[:top_n], start=1):
        print(_format_row(rank, cand, score))

    print()
    print("Top 3 by regime (for cross-regime comparison):")
    print("-" * 110)
    print(header)
    seen_by_regime: dict[str, int] = defaultdict(int)
    for cand, score in indexed:
        if seen_by_regime[cand.regime] >= 3:
            continue
        seen_by_regime[cand.regime] += 1
        rank = sum(seen_by_regime.values())
        print(_format_row(rank, cand, score))
        if all(seen_by_regime.get(r, 0) >= 3 for r in REGIMES):
            break

    print()
    print("Reproduction params for overall top (paste into the UI):")
    for rank, (cand, _) in enumerate(indexed[:top_n], start=1):
        print(
            f"  #{rank}: size={cand.size}, seed={cand.seed}, "
            f"clusterCount=[{cand.cluster_count_range[0]},{cand.cluster_count_range[1]}], "
            f"clusterSize=[{cand.cluster_size_range[0]},{cand.cluster_size_range[1]}], "
            f"removed=[{cand.removed_fraction_range[0]:.2f},{cand.removed_fraction_range[1]:.2f}]"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds-per-combo",
        type=int,
        default=40,
        help="Seeds to sweep per (size, regime) combo",
    )
    parser.add_argument(
        "--stage2-top",
        type=int,
        default=40,
        help="Number of top stage-1 candidates to run the alt-optima probe on",
    )
    parser.add_argument(
        "--probe-samples",
        type=int,
        default=8,
        help="Random edges to probe per candidate in stage 2",
    )
    parser.add_argument(
        "--final-top",
        type=int,
        default=10,
        help="Number of candidates to render and summarize",
    )
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help="Skip the expensive alt-optima probe (faster, less informative)",
    )
    parser.add_argument(
        "--max-overlap",
        type=int,
        default=3,
        help=(
            "Drop candidates whose optimal path has more than this many "
            "forced overlaps. Filters out maze-y grids in favor of puzzle-y "
            "ones with elegant optima."
        ),
    )
    args = parser.parse_args()

    rng = random.Random(0)
    candidates = generate_candidates(args.seeds_per_combo)
    print(f"Generated {len(candidates)} candidates.", flush=True)

    t0 = time.time()
    surviving: list[Candidate] = []
    failed = 0
    for idx, cand in enumerate(candidates):
        if not solve_candidate(cand):
            failed += 1
            continue
        _compute_cheap_metrics(cand)
        surviving.append(cand)
        if (idx + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(
                f"  solved {idx + 1}/{len(candidates)} "
                f"({elapsed:.1f}s, {elapsed / (idx + 1):.2f}s per grid, "
                f"{failed} failed)",
                flush=True,
            )
    candidates = surviving
    print(f"Stage 1 complete: {len(candidates)} usable, {failed} skipped.", flush=True)

    pre_filter_count = len(candidates)
    candidates = [
        c for c in candidates if c.metrics["optimal_overlap"] <= args.max_overlap
    ]
    print(
        f"Overlap filter (<= {args.max_overlap}): "
        f"{len(candidates)}/{pre_filter_count} candidates kept.",
        flush=True,
    )
    if not candidates:
        print("No candidates survived the overlap filter. Try raising --max-overlap.")
        return

    scores = _composite_score(candidates, include_uniqueness=False)

    if not args.no_probe:
        ranked = sorted(zip(candidates, scores), key=lambda kv: -kv[1])
        top_for_probe = [c for c, _ in ranked[: args.stage2_top]]
        print(f"Running alt-optima probe on top {len(top_for_probe)} candidates...", flush=True)
        for idx, cand in enumerate(top_for_probe):
            try:
                frac, count, probed = _alt_optima_fraction(
                    cand.grid, samples=args.probe_samples, rng=rng
                )
            except Exception as exc:
                print(f"  probe failed for seed={cand.seed} ({exc}); skipping", flush=True)
                continue
            cand.metrics["alt_optima_fraction"] = float(frac)
            cand.metrics["alt_optima_count"] = float(count)
            cand.metrics["alt_optima_probed"] = float(probed)
            if (idx + 1) % 5 == 0:
                print(f"  probed {idx + 1}/{len(top_for_probe)}", flush=True)
        scores = _composite_score(candidates, include_uniqueness=True)

    write_csv(candidates, scores, CSV_PATH)
    render_top(candidates, scores, args.final_top)
    print_summary(candidates, scores, args.final_top)
    print(f"\nFull results: {CSV_PATH}")
    print(f"Top SVGs:     {TOP_DIR}")


if __name__ == "__main__":
    main()
