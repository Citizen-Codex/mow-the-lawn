import argparse
import csv
import json
from pathlib import Path
import random
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.concorde import concorde_solver
from src.grid import create_random_grid
from src.shared_types import Grid, Path as SolverPath, Point
from src.visualize import calculate_visit_counts, trace_path_points


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "library"
DEFAULT_IMAGE_DIRNAME = "images"
DEFAULT_COUNT = 1000
DEFAULT_GRID_SIZE = 7
DEFAULT_MIN_SIZE = DEFAULT_GRID_SIZE
DEFAULT_MAX_SIZE = DEFAULT_GRID_SIZE
DEFAULT_BASE_SEED = 0
DEFAULT_CELL_SIZE = 42
DEFAULT_MARGIN = 24
DEFAULT_TEXT_PANEL = 64


def _serialize_grid(grid: Grid) -> str:
    return "/".join("".join(str(cell) for cell in row) for row in grid)


def _path_string(path: SolverPath) -> str:
    return "".join(path["moves"])


def _overlap_count(path: SolverPath) -> int:
    visit_counts = calculate_visit_counts(trace_path_points(path))
    return sum(count - 1 for count in visit_counts.values() if count > 1)


def _open_cell_count(grid: Grid) -> int:
    return sum(cell for row in grid for cell in row)


def _svg_cell_bbox(
    row: int, col: int, *, cell_size: int, margin: int
) -> tuple[int, int, int, int]:
    x0 = margin + col * cell_size
    y0 = margin + row * cell_size
    x1 = x0 + cell_size
    y1 = y0 + cell_size
    return x0, y0, x1, y1


def _escape_xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_solution_svg(
    grid: Grid,
    path: SolverPath,
    *,
    title: str,
    cell_size: int = DEFAULT_CELL_SIZE,
    margin: int = DEFAULT_MARGIN,
    text_panel: int = DEFAULT_TEXT_PANEL,
) -> str:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    width = cols * cell_size + margin * 2
    height = rows * cell_size + margin * 2 + text_panel
    points = trace_path_points(path)
    visit_counts = calculate_visit_counts(points)
    overlaps = _overlap_count(path)

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        "<defs>",
        '<marker id="arrow" markerWidth="7" markerHeight="7" refX="5.5" refY="2.5" orient="auto" markerUnits="strokeWidth">',
        '<path d="M0,0 L0,5 L5.5,2.5 z" fill="#0f766e" />',
        "</marker>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#f8fafc" />',
        f'<text x="{margin}" y="18" fill="#0f172a" font-size="14" font-weight="700">{_escape_xml(title)}</text>',
    ]

    for row in range(rows):
        for col in range(cols):
            x0, y0, x1, y1 = _svg_cell_bbox(
                row, col, cell_size=cell_size, margin=margin
            )
            if grid[row][col] == 0:
                fill = "#334155"
                stroke = "#1e293b"
            else:
                fill = "#ffffff"
                stroke = "#cbd5e1"
            parts.append(
                f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" fill="{fill}" stroke="{stroke}" stroke-width="1" />'
            )

    for start_point, end_point in zip(points, points[1:]):
        start_row, start_col = start_point
        end_row, end_col = end_point
        if not (0 <= start_row < rows and 0 <= start_col < cols):
            continue
        if not (0 <= end_row < rows and 0 <= end_col < cols):
            continue
        sx0, sy0, sx1, sy1 = _svg_cell_bbox(
            start_row, start_col, cell_size=cell_size, margin=margin
        )
        ex0, ey0, ex1, ey1 = _svg_cell_bbox(
            end_row, end_col, cell_size=cell_size, margin=margin
        )
        parts.append(
            (
                f'<line x1="{(sx0 + sx1) / 2}" y1="{(sy0 + sy1) / 2}" '
                f'x2="{(ex0 + ex1) / 2}" y2="{(ey0 + ey1) / 2}" '
                'stroke="#0f766e" stroke-width="3" marker-end="url(#arrow)" />'
            )
        )

    for (row, col), count in visit_counts.items():
        if (
            count <= 1
            or not (0 <= row < rows and 0 <= col < cols)
            or grid[row][col] == 0
        ):
            continue
        x0, y0, x1, y1 = _svg_cell_bbox(row, col, cell_size=cell_size, margin=margin)
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        radius = cell_size * 0.18
        parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="#fb7185" stroke="#be123c" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{cx}" y="{cy + 3}" text-anchor="middle" fill="#ffffff" font-size="10" font-weight="700">{count}</text>'
        )

    if path["start"] is not None:
        start_row, start_col = path["start"]
        x0, y0, x1, y1 = _svg_cell_bbox(
            start_row, start_col, cell_size=cell_size, margin=margin
        )
        parts.append(
            f'<rect x="{x0 + 5}" y="{y0 + 5}" width="{x1 - x0 - 10}" height="{y1 - y0 - 10}" fill="none" stroke="#16a34a" stroke-width="3" />'
        )
        parts.append(
            f'<text x="{x0 + 12}" y="{y0 + 15}" fill="#15803d" font-size="10" font-weight="700">S</text>'
        )

    if points:
        end_row, end_col = points[-1]
        x0, y0, x1, y1 = _svg_cell_bbox(
            end_row, end_col, cell_size=cell_size, margin=margin
        )
        parts.append(
            f'<rect x="{x0 + 5}" y="{y0 + 5}" width="{x1 - x0 - 10}" height="{y1 - y0 - 10}" fill="none" stroke="#dc2626" stroke-width="3" />'
        )
        parts.append(
            f'<text x="{x0 + 12}" y="{y0 + 15}" fill="#b91c1c" font-size="10" font-weight="700">E</text>'
        )

    stats_y = margin + rows * cell_size + 24
    parts.append(
        f'<text x="{margin}" y="{stats_y}" fill="#334155" font-size="12">Moves: {len(path["moves"])} | Overlaps: {overlaps} | Open cells: {_open_cell_count(grid)}</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def generate_optimal_library(
    count: int = DEFAULT_COUNT,
    *,
    min_size: int = DEFAULT_MIN_SIZE,
    max_size: int = DEFAULT_MAX_SIZE,
    base_seed: int = DEFAULT_BASE_SEED,
    output_dir: Path | None = None,
) -> list[dict[str, str | int]]:
    if count <= 0:
        raise ValueError("count must be a positive integer")
    if min_size <= 0 or max_size <= 0:
        raise ValueError("grid sizes must be positive integers")
    if min_size > max_size:
        raise ValueError("min_size must be less than or equal to max_size")

    output_root = output_dir or DEFAULT_OUTPUT_DIR
    image_dir = output_root / DEFAULT_IMAGE_DIRNAME
    output_root.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    for existing_image in image_dir.glob("*.svg"):
        existing_image.unlink()

    size_rng = random.Random(base_seed)
    rows: list[dict[str, str | int]] = []

    for index in range(count):
        if index > 0 and index % 100 == 0:
            print(f"Generated {index}/{count} optimal paths...")

        size = size_rng.randint(min_size, max_size)
        seed = base_seed + index
        grid = create_random_grid(size, seed)
        path = concorde_solver(grid)
        points = trace_path_points(path)
        start: Point | None = path["start"]

        image_name = f"grid_{index:04d}_n{size}_seed{seed}.svg"
        image_path = image_dir / image_name
        image_path.write_text(
            render_solution_svg(
                grid,
                path,
                title=f"Optimal Path {index:04d} | {size}x{size} | seed={seed}",
            ),
            encoding="utf-8",
        )

        rows.append(
            {
                "index": index,
                "size": size,
                "seed": seed,
                "open_cells": _open_cell_count(grid),
                "start_row": -1 if start is None else start[0],
                "start_col": -1 if start is None else start[1],
                "end_row": -1 if not points else points[-1][0],
                "end_col": -1 if not points else points[-1][1],
                "moves": len(path["moves"]),
                "overlaps": _overlap_count(path),
                "path": _path_string(path),
                "grid": _serialize_grid(grid),
                "image": str(image_path.relative_to(output_root)),
            }
        )

    csv_path = output_root / "optimal_paths.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    average_moves = sum(int(row["moves"]) for row in rows) / len(rows)
    average_overlaps = sum(int(row["overlaps"]) for row in rows) / len(rows)
    size_counts: dict[str, int] = {}
    for row in rows:
        size_key = str(row["size"])
        size_counts[size_key] = size_counts.get(size_key, 0) + 1

    summary = {
        "count": count,
        "min_size": min_size,
        "max_size": max_size,
        "base_seed": base_seed,
        "csv": csv_path.name,
        "images": DEFAULT_IMAGE_DIRNAME,
        "average_moves": average_moves,
        "average_overlaps": average_overlaps,
        "size_counts": size_counts,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a library of optimal paths and SVG images"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=f"Number of random grids to generate (default: {DEFAULT_COUNT})",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=DEFAULT_MIN_SIZE,
        help=f"Minimum grid size to sample (default: {DEFAULT_MIN_SIZE})",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=DEFAULT_MAX_SIZE,
        help=f"Maximum grid size to sample (default: {DEFAULT_MAX_SIZE})",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=DEFAULT_BASE_SEED,
        help=f"Base seed for reproducible sampling (default: {DEFAULT_BASE_SEED})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for the CSV, summary, and SVG images",
    )
    args = parser.parse_args()

    rows = generate_optimal_library(
        args.count,
        min_size=args.min_size,
        max_size=args.max_size,
        base_seed=args.base_seed,
        output_dir=Path(args.output_dir),
    )
    print(f"Generated {len(rows)} optimal paths -> {args.output_dir}")


if __name__ == "__main__":
    main()
