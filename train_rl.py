import argparse

from src.rl_solver.config import EnvConfig, TrainConfig, resolve_train_config
from src.rl_solver.model import train_maskable_ppo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a MaskablePPO solver for the lawn grid"
    )
    parser.add_argument(
        "--size", type=int, required=True, help="Grid size (size x size)"
    )
    parser.add_argument(
        "--timesteps", type=int, default=1_000_000, help="Training timesteps"
    )
    parser.add_argument(
        "--output", default="data/rl/maskable_ppo.zip", help="Model output path"
    )
    parser.add_argument(
        "--resume",
        help="Resume training from an existing MaskablePPO checkpoint",
    )
    parser.add_argument("--n-envs", type=int, default=16, help="Parallel environments")
    parser.add_argument("--n-steps", type=int, help="PPO rollout steps per environment")
    parser.add_argument("--batch-size", type=int, help="PPO minibatch size")
    parser.add_argument(
        "--n-epochs", type=int, help="PPO optimization epochs per update"
    )
    parser.add_argument(
        "--no-auto-tune",
        action="store_true",
        help="Disable rollout/update auto-tuning for the chosen env count",
    )
    parser.add_argument(
        "--base-seed", type=int, default=0, help="Base seed for generated grids"
    )
    parser.add_argument("--seed", type=int, default=0, help="Training RNG seed")
    args = parser.parse_args()

    env_config = EnvConfig(size=args.size, base_seed=args.base_seed)
    train_config = TrainConfig(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        seed=args.seed,
        auto_tune=not args.no_auto_tune,
    )
    resolved_train_config = resolve_train_config(train_config)
    print(
        "Training config: "
        f"timesteps={resolved_train_config.total_timesteps} "
        f"resume={args.resume or 'none'} "
        f"max_overlaps={env_config.max_overlaps} "
        f"n_envs={resolved_train_config.n_envs} "
        f"n_steps={resolved_train_config.n_steps} "
        f"batch_size={resolved_train_config.batch_size} "
        f"n_epochs={resolved_train_config.n_epochs} "
        f"gamma={resolved_train_config.gamma} "
        f"ent_coef={resolved_train_config.ent_coef} "
        f"eval_freq={resolved_train_config.eval_freq_timesteps} "
        f"eval_seeds={resolved_train_config.eval_seed_count} "
        f"save_best={resolved_train_config.save_best_checkpoint} "
        f"auto_tune={resolved_train_config.auto_tune}"
    )
    model_path = train_maskable_ppo(
        env_config,
        resolved_train_config,
        args.output,
        resume_from=args.resume,
    )
    print(f"Saved RL model to {model_path}")


if __name__ == "__main__":
    main()
