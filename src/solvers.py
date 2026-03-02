from collections import deque

from src.shared_types import Grid, Move, Path, Point, MOVE_DELTAS


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


def snake_solver(grid: Grid) -> Path:
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

    remaining_targets = dict.fromkeys(targets)
    remaining_targets.pop(start, None)

    current = start
    moves: list[Move] = []

    while remaining_targets:
        target = next(iter(remaining_targets))

        path_moves = shortest_path_moves(grid, current, target)
        if path_moves is None:
            remaining_targets.pop(target, None)
            continue

        for move in path_moves:
            dx, dy = MOVE_DELTAS[move]
            current = (current[0] + dx, current[1] + dy)
            moves.append(move)
            remaining_targets.pop(current, None)

    return {"start": start, "moves": moves}


def spiral_solver(grid: Grid) -> Path:
    if not grid or not grid[0]:
        return {"start": None, "moves": []}

    start = find_start(grid)
    if start is None:
        return {"start": None, "moves": []}

    rows = len(grid)
    cols = len(grid[0])

    all_targets: set[Point] = {
        (row, col) for row in range(rows) for col in range(cols) if grid[row][col] == 1
    }

    targets: list[Point] = [start]
    remaining_targets = set(all_targets)
    remaining_targets.discard(start)

    while remaining_targets:
        min_row = min(row for row, _ in remaining_targets)
        max_row = max(row for row, _ in remaining_targets)
        min_col = min(col for _, col in remaining_targets)
        max_col = max(col for _, col in remaining_targets)

        down_targets = sorted(
            (point for point in remaining_targets if point[1] == min_col),
            key=lambda point: point[0],
        )
        for point in down_targets:
            targets.append(point)
            remaining_targets.remove(point)

        right_targets = sorted(
            (point for point in remaining_targets if point[0] == max_row),
            key=lambda point: point[1],
        )
        for point in right_targets:
            targets.append(point)
            remaining_targets.remove(point)

        up_targets = sorted(
            (point for point in remaining_targets if point[1] == max_col),
            key=lambda point: point[0],
            reverse=True,
        )
        for point in up_targets:
            targets.append(point)
            remaining_targets.remove(point)

        left_targets = sorted(
            (point for point in remaining_targets if point[0] == min_row),
            key=lambda point: point[1],
            reverse=True,
        )
        for point in left_targets:
            targets.append(point)
            remaining_targets.remove(point)

    remaining_targets = dict.fromkeys(targets)
    remaining_targets.pop(start, None)

    current = start
    moves: list[Move] = []

    while remaining_targets:
        target = next(iter(remaining_targets))

        path_moves = shortest_path_moves(grid, current, target)
        if path_moves is None:
            remaining_targets.pop(target, None)
            continue

        for move in path_moves:
            dx, dy = MOVE_DELTAS[move]
            current = (current[0] + dx, current[1] + dy)
            moves.append(move)
            remaining_targets.pop(current, None)

    return {"start": start, "moves": moves}
