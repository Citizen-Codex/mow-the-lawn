import argparse

from src.grid import create_random_grid
from src.rl_solver.config import load_run_config
from src.rl_solver.metrics import format_rollout_result
from src.rl_solver.rollout import rollout_model_on_grid
from src.visualize import path_stats, show_grid_path_tk, show_grid_tk


def _print_result(label: str, result: dict) -> None:
    print(format_rollout_result(result, prefix=label))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a LawnMowingEnv grid and the trained model solution"
    )
    parser.add_argument(
        "--model", required=True, help="Path to a saved MaskablePPO model"
    )
    parser.add_argument("--seed", type=int, default=0, help="Grid seed to visualize")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy sampling instead of deterministic inference",
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Skip the base grid window and only show solution path windows",
    )
    args = parser.parse_args()

    config = load_run_config(args.model)
    size = int(config["env"]["size"])
    grid = create_random_grid(size, args.seed)
    model_result = rollout_model_on_grid(
        args.model,
        grid,
        deterministic=not args.stochastic,
    )

    print(f"Visualizing seed={args.seed} size={size}x{size}")
    _print_result("model", model_result)
    print(path_stats(model_result["path"]))

    if not args.no_grid:
        launched_grid = show_grid_tk(
            grid, title=f"LawnMowingEnv Grid (seed={args.seed})"
        )
        if not launched_grid:
            print("Grid visualizer could not start (likely no display environment).")

    launched_model = show_grid_path_tk(
        grid,
        model_result["path"],
        title=f"RL Model Path (seed={args.seed})",
    )
    if not launched_model:
        print("Model path visualizer could not start (likely no display environment).")


if __name__ == "__main__":
    main()
