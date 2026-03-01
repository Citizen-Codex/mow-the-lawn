import random
from collections import deque
from typing import Literal, TypeAlias, TypedDict

from visualize import render_grid, render_grid_with_path, render_path

directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
Move: TypeAlias = Literal["up", "down", "left", "right"]
Point: TypeAlias = tuple[int, int]
Grid: TypeAlias = list[list[int]]

move_deltas = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


class Path(TypedDict):
    start: Point | None
    moves: list[Move]


def can_remove(grid: Grid, x: int, y: int) -> bool:
    n = len(grid)
    if n == 0 or not (0 <= x < n and 0 <= y < n):
        return False
    if grid[x][y] == 0:
        return False

    neighbors: list[Point] = []
    for dx, dy in directions:
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
        for dx, dy in directions:
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

            dx, dy = directions[rng.randrange(len(directions))]
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

    ordered_moves: list[Move] = ["up", "down", "left", "right"]

    while queue:
        cx, cy = queue.popleft()
        for move_name in ordered_moves:
            dx, dy = move_deltas[move_name]
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
            dx, dy = move_deltas[move]
            current = (current[0] + dx, current[1] + dy)
            moves.append(move)

    return {"start": start, "moves": moves}


if __name__ == "__main__":
    example_n = 12
    example_seed = 0
    demo_grid = create_random_grid(example_n, example_seed)

    print(f"Simulated {example_n}x{example_n} grid (seed={example_seed})")
    print("1 = present, 0 = removed\n")
    print(render_grid(demo_grid))

    path = snake_solver(demo_grid)
    print("\nZig-zag path")
    print(render_path(path))

    print("\nGrid with path overlay")
    print("Legend: S=start, E=end, *=visited, .=open, 0=removed\n")
    print(render_grid_with_path(demo_grid, path))
