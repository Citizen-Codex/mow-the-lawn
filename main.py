from src.grid import create_random_grid
from src.solvers import snake_solver, spiral_solver
from src.visualize import path_stats, show_grid_path_tk, show_grid_tk


def main(n: int, seed: int):
    demo_grid = create_random_grid(n, seed)

    print(f"Simulated {n}x{n} grid (seed={seed})")
    print("Launching empty-grid visualizer...")

    launched_grid = show_grid_tk(demo_grid, title="Grid Visualizer")
    if not launched_grid:
        print("Grid visualizer could not start (likely no display environment).")

    snake_path = snake_solver(demo_grid)
    print(path_stats(snake_path))
    print("Launching snake path visualizer...")

    launched_snake = show_grid_path_tk(
        demo_grid, snake_path, title="Snake Path Visualizer"
    )
    if not launched_snake:
        print("Snake path visualizer could not start (likely no display environment).")

    spiral_path = spiral_solver(demo_grid)
    print(path_stats(spiral_path))
    print("Launching spiral path visualizer...")

    launched_spiral = show_grid_path_tk(
        demo_grid, spiral_path, title="Spiral Path Visualizer"
    )
    if not launched_spiral:
        print("Spiral path visualizer could not start (likely no display environment).")


if __name__ == "__main__":
    main(12, 0)
