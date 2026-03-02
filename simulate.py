import argparse

import pandas as pd

from src.grid import create_random_grid
from src.solvers import snake_solver, spiral_solver


def create_labelled_paths(
    count: int, size: int, output_path: str = "data/labelled_paths.csv"
) -> pd.DataFrame:
    if count <= 0:
        raise ValueError("count must be a positive integer")
    if size <= 0:
        raise ValueError("size must be a positive integer")

    rows: list[dict[str, str | int]] = []

    for index in range(count):
        grid = create_random_grid(size, index)
        snake_path = snake_solver(grid)
        spiral_path = spiral_solver(grid)

        rows.append(
            {
                "snake_path": "".join(snake_path["moves"]),
                "spiral_path": "".join(spiral_path["moves"]),
            }
        )

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(output_path, index=False)
    return dataframe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate labelled snake/spiral path data as CSV"
    )
    parser.add_argument("count", type=int, help="Number of random grids to generate")
    parser.add_argument("size", type=int, help="Grid size (size x size)")
    parser.add_argument(
        "--output",
        default="data/labelled_paths.csv",
        help="Output CSV path (default: labelled_paths.csv)",
    )
    args = parser.parse_args()

    dataframe = create_labelled_paths(args.count, args.size, args.output)
    print(f"Generated {len(dataframe)} rows -> {args.output}")


if __name__ == "__main__":
    main()
