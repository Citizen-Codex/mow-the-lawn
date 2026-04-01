import argparse

from src.rl_solver.evaluate import evaluate_model, summarize_results
from src.rl_solver.metrics import format_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an RL solver on generated grids using the model's saved density range"
    )
    parser.add_argument(
        "--model", required=True, help="Path to a saved MaskablePPO model"
    )
    parser.add_argument(
        "--size",
        type=int,
        required=True,
        help="Grid size used for evaluation (must not exceed the model max size)",
    )
    parser.add_argument(
        "--seeds", type=int, default=50, help="Number of evaluation seeds"
    )
    parser.add_argument(
        "--start-seed", type=int, default=0, help="First evaluation seed"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy sampling instead of deterministic inference",
    )
    args = parser.parse_args()

    results = evaluate_model(
        args.model,
        size=args.size,
        seeds=range(args.start_seed, args.start_seed + args.seeds),
        deterministic=not args.stochastic,
    )
    summary = summarize_results(results)

    print(f"Evaluated {args.seeds} seeds at size {args.size}x{args.size}")
    print(format_summary(summary, prefix="rl"))


if __name__ == "__main__":
    main()
