import argparse
import json
import random
from collections import deque

try:
    from src.shared_types import Grid, Point, MOVE_DELTAS
except ModuleNotFoundError:  # Support direct script execution: uv run src/grid.py
    from shared_types import Grid, Point, MOVE_DELTAS


def can_remove(grid: Grid, x: int, y: int) -> bool:
    n = len(grid)
    if n == 0 or not (0 <= x < n and 0 <= y < n):
        return False
    if grid[x][y] == 0:
        return False

    neighbors: list[Point] = []
    for dx, dy in MOVE_DELTAS.values():
        nx, ny = x + dx, y + dy
        if 0 <= nx < n and 0 <= ny < n and grid[nx][ny] == 1:
            neighbors.append((nx, ny))

    if len(neighbors) <= 1:
        return True

    start = neighbors[0]
    visited = {start}
    queue = deque([start])

    while queue:
        cx, cy = queue.popleft()
        for dx, dy in MOVE_DELTAS.values():
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < n and 0 <= ny < n):
                continue
            if (nx, ny) == (x, y):
                continue
            if grid[nx][ny] == 0 or (nx, ny) in visited:
                continue
            visited.add((nx, ny))
            queue.append((nx, ny))

    return all(neighbor in visited for neighbor in neighbors)


def default_cluster_count_range(n: int) -> tuple[int, int]:
    return (max(1, n // 3), max(2, n // 2))


def default_cluster_size_range(n: int) -> tuple[int, int]:
    return (max(2, n // 2), max(3, n))


def _validate_int_range(name: str, value: tuple[int, int]) -> None:
    lo, hi = value
    if not (isinstance(lo, int) and isinstance(hi, int)):
        raise ValueError(f"{name} must be a pair of integers")
    if lo < 1 or hi < lo:
        raise ValueError(f"{name} must satisfy 1 <= min <= max")


def create_random_grid(
    n: int,
    s: int,
    *,
    removed_fraction_range: tuple[float, float] = (0.18, 0.42),
    cluster_count_range: tuple[int, int] | None = None,
    cluster_size_range: tuple[int, int] | None = None,
) -> Grid:
    if n <= 0:
        raise ValueError("n must be a positive integer")

    min_removed_fraction, max_removed_fraction = removed_fraction_range
    if not (0.0 <= min_removed_fraction <= max_removed_fraction < 1.0):
        raise ValueError("removed_fraction_range must satisfy 0.0 <= min <= max < 1.0")

    if cluster_count_range is None:
        cluster_count_range = default_cluster_count_range(n)
    if cluster_size_range is None:
        cluster_size_range = default_cluster_size_range(n)
    _validate_int_range("cluster_count_range", cluster_count_range)
    _validate_int_range("cluster_size_range", cluster_size_range)

    rng = random.Random(s)
    grid = [[1 for _ in range(n)] for _ in range(n)]
    cluster_id_grid = [[-1 for _ in range(n)] for _ in range(n)]

    def touches_other_cluster(x: int, y: int, current_id: int) -> bool:
        for dx, dy in MOVE_DELTAS.values():
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n:
                nid = cluster_id_grid[nx][ny]
                if nid != -1 and nid != current_id:
                    return True
        return False

    max_removed = n * n * rng.uniform(min_removed_fraction, max_removed_fraction)
    removed = 0
    cluster_count = rng.randint(*cluster_count_range)

    for cluster_id in range(cluster_count):
        if removed >= max_removed:
            break

        seeded = False
        for _ in range(20):
            x = rng.randrange(n)
            y = rng.randrange(n)
            if grid[x][y] == 1 and not touches_other_cluster(x, y, cluster_id):
                seeded = True
                break
        if not seeded:
            continue

        cluster_size = rng.randint(*cluster_size_range)

        for _ in range(cluster_size):
            if can_remove(grid, x, y) and not touches_other_cluster(x, y, cluster_id):
                grid[x][y] = 0
                cluster_id_grid[x][y] = cluster_id
                removed += 1
                if removed >= max_removed:
                    break

            dx, dy = MOVE_DELTAS[rng.choice(list(MOVE_DELTAS.keys()))]
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n:
                x, y = nx, ny

    return grid


def obstacle_coordinates(grid: Grid) -> list[dict[str, int]]:
    return [
        {"x": x, "y": y}
        for x, row in enumerate(grid)
        for y, cell in enumerate(row)
        if cell == 0
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and print a random grid")
    parser.add_argument("size", type=int, nargs="?", default=10, help="Grid size")
    parser.add_argument("seed", type=int, nargs="?", default=0, help="Random seed")
    parser.add_argument(
        "--removed-min",
        type=float,
        default=0.18,
        help="Minimum removed-cell fraction",
    )
    parser.add_argument(
        "--removed-max",
        type=float,
        default=0.42,
        help="Maximum removed-cell fraction",
    )
    parser.add_argument(
        "--cluster-count-min",
        type=int,
        default=None,
        help="Minimum number of obstacle clusters (default: scales with size)",
    )
    parser.add_argument(
        "--cluster-count-max",
        type=int,
        default=None,
        help="Maximum number of obstacle clusters (default: scales with size)",
    )
    parser.add_argument(
        "--cluster-size-min",
        type=int,
        default=None,
        help="Minimum cells per cluster (default: scales with size)",
    )
    parser.add_argument(
        "--cluster-size-max",
        type=int,
        default=None,
        help="Maximum cells per cluster (default: scales with size)",
    )
    args = parser.parse_args()

    cluster_count_range = None
    if args.cluster_count_min is not None or args.cluster_count_max is not None:
        default_lo, default_hi = default_cluster_count_range(args.size)
        cluster_count_range = (
            args.cluster_count_min if args.cluster_count_min is not None else default_lo,
            args.cluster_count_max if args.cluster_count_max is not None else default_hi,
        )

    cluster_size_range = None
    if args.cluster_size_min is not None or args.cluster_size_max is not None:
        default_lo, default_hi = default_cluster_size_range(args.size)
        cluster_size_range = (
            args.cluster_size_min if args.cluster_size_min is not None else default_lo,
            args.cluster_size_max if args.cluster_size_max is not None else default_hi,
        )

    grid = create_random_grid(
        args.size,
        args.seed,
        removed_fraction_range=(args.removed_min, args.removed_max),
        cluster_count_range=cluster_count_range,
        cluster_size_range=cluster_size_range,
    )
    payload = {
        "size": args.size,
        "seed": args.seed,
        "grid": grid,
        "obstacles": obstacle_coordinates(grid),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
