from __future__ import annotations

from statistics import mean
from typing import Any

from src.shared_types import Grid, Path
from src.visualize import calculate_visit_counts, trace_path_points


SummaryRow = dict[str, int | float | bool]
SummaryStats = dict[str, float]
RolloutResult = dict[str, Any]


def path_metrics(grid: Grid, path: Path) -> SummaryRow:
    open_cells = {
        (row_index, col_index)
        for row_index, row in enumerate(grid)
        for col_index, cell in enumerate(row)
        if cell == 1
    }
    points = trace_path_points(path)
    visited_open = {point for point in points if point in open_cells}
    visit_counts = calculate_visit_counts(points)
    overlaps = sum(
        count - 1
        for point, count in visit_counts.items()
        if point in open_cells and count > 1
    )
    open_count = len(open_cells)
    coverage = len(visited_open)

    return {
        "completed": coverage == open_count,
        "open_cells": open_count,
        "covered_cells": coverage,
        "coverage_ratio": (coverage / open_count) if open_count else 0.0,
        "moves": len(path["moves"]),
        "overlaps": overlaps,
    }


def summarize_metric_rows(rows: list[SummaryRow]) -> SummaryStats:
    return {
        "success_rate": mean(float(row["completed"]) for row in rows),
        "avg_coverage": mean(float(row["coverage_ratio"]) for row in rows),
        "avg_moves": mean(float(row["moves"]) for row in rows),
        "avg_overlaps": mean(float(row["overlaps"]) for row in rows),
    }


def summary_sort_key(summary: SummaryStats) -> tuple[float, float, float, float]:
    return (
        summary["success_rate"],
        summary["avg_coverage"],
        -summary["avg_overlaps"],
        -summary["avg_moves"],
    )


def format_summary(summary: SummaryStats, *, prefix: str = "") -> str:
    start = f"{prefix} " if prefix else ""
    return (
        f"{start}success={summary['success_rate']:.3f} "
        f"coverage={summary['avg_coverage']:.3f} "
        f"moves={summary['avg_moves']:.2f} "
        f"overlaps={summary['avg_overlaps']:.2f}"
    )


def summarize_rollout_results(results: list[RolloutResult]) -> SummaryStats:
    return summarize_metric_rows([result["metrics"] for result in results])


def format_rollout_result(result: RolloutResult, *, prefix: str = "") -> str:
    metrics = result["metrics"]
    start = f"{prefix}: " if prefix else ""
    return (
        f"{start}completed={int(bool(metrics['completed']))} "
        f"coverage={float(metrics['coverage_ratio']):.3f} "
        f"moves={int(metrics['moves'])} "
        f"overlaps={int(metrics['overlaps'])} "
        f"reward={float(result['total_reward']):.2f} "
        f"invalid_actions={int(result['invalid_actions'])} "
        f"edge_reuses={int(result['edge_reuses'])} "
        f"loop_oscillations={int(result['loop_oscillations'])} "
        f"max_oscillation_streak={int(result['max_oscillation_streak'])} "
        f"max_cell_overlap_count={int(result['max_cell_overlap_count'])} "
        f"overlap_limit_hit={int(bool(result['overlap_limit_hit']))} "
        f"terminated={int(bool(result['terminated']))} "
        f"truncated={int(bool(result['truncated']))}"
    )
