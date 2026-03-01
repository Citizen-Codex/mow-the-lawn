import random
from collections import deque

from grid_types import Point, Grid, Move, Path, MOVE_DELTAS
from visualize import path_stats, show_grid_path_tk, show_grid_tk


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


def create_random_grid(n: int, s: int) -> Grid:
    if n <= 0:
        raise ValueError("n must be a positive integer")

    rng = random.Random(s)
    grid = [[1 for _ in range(n)] for _ in range(n)]

    max_removed = n * n * rng.uniform(0.18, 0.42)
    removed = 0
    cluster_count = rng.randint(max(1, n // 3), max(2, n // 2))

    for _ in range(cluster_count):
        if removed >= max_removed:
            break

        x = rng.randrange(n)
        y = rng.randrange(n)
        cluster_size = rng.randint(max(2, n // 2), max(3, n))

        for _ in range(cluster_size):
            if can_remove(grid, x, y):
                grid[x][y] = 0
                removed += 1
                if removed >= max_removed:
                    break

            dx, dy = MOVE_DELTAS[rng.choice(list(MOVE_DELTAS.keys()))]
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n:
                x, y = nx, ny

    return grid


def find_start(grid: Grid) -> Point | None:
    if not grid or not grid[0]:
        return None

    rows = len(grid)
    cols = len(grid[0])
    for col_index in range(cols):
        for row_index in range(rows):
            if grid[row_index][col_index] == 1:
                return row_index, col_index
    return None


def shortest_path_moves(grid: Grid, start: Point, target: Point) -> list[Move] | None:
    if start == target:
        return []

    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    queue = deque([start])
    parents: dict[Point, tuple[Point, Move]] = {}
    visited = {start}

    ordered_moves: list[Move] = ["u", "d", "l", "r"]

    while queue:
        cx, cy = queue.popleft()
        for move_name in ordered_moves:
            dx, dy = MOVE_DELTAS[move_name]
            nx, ny = cx + dx, cy + dy

            if not (0 <= nx < rows and 0 <= ny < cols):
                continue
            if grid[nx][ny] == 0 or (nx, ny) in visited:
                continue

            parents[(nx, ny)] = ((cx, cy), move_name)
            if (nx, ny) == target:
                path: list[Move] = []
                node = (nx, ny)
                while node != start:
                    previous, step = parents[node]
                    path.append(step)
                    node = previous
                path.reverse()
                return path

            visited.add((nx, ny))
            queue.append((nx, ny))

    return None


def snake_solver(
    grid: Grid,
) -> Path:
    if not grid or not grid[0]:
        return {"start": None, "moves": []}

    start = find_start(grid)
    if start is None:
        return {"start": None, "moves": []}

    rows = len(grid)
    cols = len(grid[0])
    start_col = start[1]

    targets: list[Point] = []
    for col in range(start_col, cols):
        row_range = (
            range(rows) if (col - start_col) % 2 == 0 else range(rows - 1, -1, -1)
        )
        for row in row_range:
            if grid[row][col] == 1:
                targets.append((row, col))

    current = start
    moves: list[Move] = []

    for target in targets:
        if target == current:
            continue

        path_moves = shortest_path_moves(grid, current, target)
        if path_moves is None:
            continue

        for move in path_moves:
            dx, dy = MOVE_DELTAS[move]
            current = (current[0] + dx, current[1] + dy)
            moves.append(move)

    return {"start": start, "moves": moves}


if __name__ == "__main__":
    example_n = 12
    example_seed = 0
    demo_grid = create_random_grid(example_n, example_seed)

    print(f"Simulated {example_n}x{example_n} grid (seed={example_seed})")
    print("Launching empty-grid visualizer...")

    launched_grid = show_grid_tk(demo_grid, title="Grid Visualizer")
    if not launched_grid:
        print("Grid visualizer could not start (likely no display environment).")

    path = snake_solver(demo_grid)
    print(path_stats(path))
    print("Launching path visualizer...")

    launched = show_grid_path_tk(demo_grid, path, title="Snake Path Visualizer")
    if not launched:
        print("Path visualizer could not start (likely no display environment).")
