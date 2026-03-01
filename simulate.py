import random


def simulate(n: int, s: int) -> list[list[int]]:
    if n <= 0:
        raise ValueError("n must be a positive integer")

    rng = random.Random(s)
    grid = [[1 for _ in range(n)] for _ in range(n)]

    max_removed = n * n * rng.uniform(0.18, 0.42)
    removed = 0
    cluster_count = rng.randint(max(1, n // 3), max(2, n // 2))
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for _ in range(cluster_count):
        if removed >= max_removed:
            break

        x = rng.randrange(n)
        y = rng.randrange(n)
        cluster_size = rng.randint(max(2, n // 2), max(3, n))

        for _ in range(cluster_size):
            if grid[x][y] == 1:
                grid[x][y] = 0
                removed += 1
                if removed >= max_removed:
                    break

            dx, dy = directions[rng.randrange(len(directions))]
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n:
                x, y = nx, ny

    return grid


if __name__ == "__main__":
    example_n = 12
    example_seed = 0
    demo_grid = simulate(example_n, example_seed)

    print(f"Simulated {example_n}x{example_n} grid (seed={example_seed})")
    print("1 = present, 0 = removed\n")
    for row in demo_grid:
        print(" ".join(str(cell) for cell in row))
