import random
from collections import deque

from src.shared_types import Grid, Point, MOVE_DELTAS


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
