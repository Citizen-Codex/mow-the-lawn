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
        "--learning-rate",
        type=float,
        default=3e-4,
        help="PPO optimizer learning rate",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.995,
        help="PPO discount factor",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="PPO GAE lambda",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.005,
        help="PPO entropy coefficient",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.2,
        help="PPO clip range",
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
    parser.add_argument(
        "--curriculum-min-size",
        type=int,
        help="Smallest grid size to sample during curriculum training",
    )
    parser.add_argument(
        "--curriculum-warmup-episodes",
        type=int,
        default=0,
        help="Episodes per environment used to ramp curriculum up to --size",
    )
    parser.add_argument(
        "--curriculum-target-size-prob",
        type=float,
        default=0.5,
        help="Probability of sampling the max grid size once it is available",
    )
    parser.add_argument(
        "--curriculum-size-bias-power",
        type=float,
        default=2.0,
        help="Bias power used to favor larger unlocked curriculum sizes",
    )
    parser.add_argument(
        "--final-target-only-timesteps",
        type=int,
        default=0,
        help="Optional final fine-tuning timesteps run only at the max grid size",
    )
    parser.add_argument(
        "--removed-fraction-min",
        type=float,
        default=0.18,
        help="Minimum removed-cell fraction for generated training grids",
    )
    parser.add_argument(
        "--removed-fraction-max",
        type=float,
        default=0.42,
        help="Maximum removed-cell fraction for generated training grids",
    )
    parser.add_argument(
        "--max-overlaps",
        type=int,
        default=2,
        help="Per-cell overlap threshold before the overlap-limit penalty applies",
    )
    parser.add_argument(
        "--terminate-on-overlap-limit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Terminate immediately when a cell exceeds --max-overlaps",
    )
    parser.add_argument(
        "--max-steps-factor",
        type=float,
        default=2.0,
        help="Episode step budget as a multiple of open-cell count",
    )
    parser.add_argument(
        "--bc-epochs",
        type=int,
        default=0,
        help="Behavior cloning epochs before PPO (0 disables pretraining)",
    )
    parser.add_argument(
        "--bc-grid-count",
        type=int,
        default=0,
        help="Number of optimal-path library rows to use for behavior cloning",
    )
    parser.add_argument(
        "--bc-batch-size",
        type=int,
        default=1024,
        help="Behavior cloning minibatch size",
    )
    parser.add_argument(
        "--bc-learning-rate",
        type=float,
        default=3e-4,
        help="Behavior cloning optimizer learning rate",
    )
    parser.add_argument(
        "--bc-seed-start",
        type=int,
        default=0,
        help="Starting offset into the filtered optimal-path library rows",
    )
    parser.add_argument(
        "--bc-size-min",
        type=int,
        help="Optional minimum expert-library grid size for behavior cloning",
    )
    parser.add_argument(
        "--bc-size-max",
        type=int,
        help="Optional maximum expert-library grid size for behavior cloning",
    )
    parser.add_argument(
        "--bc-removed-fraction-min",
        type=float,
        help="Optional min removed-cell fraction for behavior cloning demo grids",
    )
    parser.add_argument(
        "--bc-removed-fraction-max",
        type=float,
        help="Optional max removed-cell fraction for behavior cloning demo grids",
    )
    parser.add_argument(
        "--eval-freq-timesteps",
        type=int,
        default=20_000,
        help="Held-out evaluation frequency in environment timesteps",
    )
    parser.add_argument(
        "--eval-seed-count",
        type=int,
        default=50,
        help="Number of held-out seeds used per evaluation sweep",
    )
    parser.add_argument(
        "--save-best-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save a *.best.zip checkpoint when held-out metrics improve",
    )
    args = parser.parse_args()

    env_config = EnvConfig(
        size=args.size,
        curriculum_min_size=args.curriculum_min_size,
        curriculum_warmup_episodes=args.curriculum_warmup_episodes,
        curriculum_target_size_probability=args.curriculum_target_size_prob,
        curriculum_size_bias_power=args.curriculum_size_bias_power,
        removed_fraction_min=args.removed_fraction_min,
        removed_fraction_max=args.removed_fraction_max,
        max_overlaps=args.max_overlaps,
        terminate_on_overlap_limit=args.terminate_on_overlap_limit,
        max_steps_factor=args.max_steps_factor,
        base_seed=args.base_seed,
    )
    train_config = TrainConfig(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        seed=args.seed,
        auto_tune=not args.no_auto_tune,
        bc_epochs=args.bc_epochs,
        bc_grid_count=args.bc_grid_count,
        bc_batch_size=args.bc_batch_size,
        bc_learning_rate=args.bc_learning_rate,
        bc_seed_start=args.bc_seed_start,
        bc_size_min=args.bc_size_min,
        bc_size_max=args.bc_size_max,
        bc_removed_fraction_min=args.bc_removed_fraction_min,
        bc_removed_fraction_max=args.bc_removed_fraction_max,
        final_target_only_timesteps=args.final_target_only_timesteps,
        eval_freq_timesteps=args.eval_freq_timesteps,
        eval_seed_count=args.eval_seed_count,
        save_best_checkpoint=args.save_best_checkpoint,
    )
    resolved_train_config = resolve_train_config(train_config)
    print(
        "Training config: "
        f"timesteps={resolved_train_config.total_timesteps} "
        f"resume={args.resume or 'none'} "
        f"curriculum_min_size={env_config.curriculum_min_size or env_config.size} "
        f"curriculum_warmup={env_config.curriculum_warmup_episodes} "
        f"curriculum_target_prob={env_config.curriculum_target_size_probability} "
        f"curriculum_size_bias={env_config.curriculum_size_bias_power} "
        f"removed_fraction_min={env_config.removed_fraction_min} "
        f"removed_fraction_max={env_config.removed_fraction_max} "
        f"max_overlaps={env_config.max_overlaps} "
        f"terminate_on_overlap_limit={env_config.terminate_on_overlap_limit} "
        f"max_steps_factor={env_config.max_steps_factor} "
        f"n_envs={resolved_train_config.n_envs} "
        f"n_steps={resolved_train_config.n_steps} "
        f"batch_size={resolved_train_config.batch_size} "
        f"n_epochs={resolved_train_config.n_epochs} "
        f"lr={resolved_train_config.learning_rate} "
        f"gamma={resolved_train_config.gamma} "
        f"gae_lambda={resolved_train_config.gae_lambda} "
        f"ent_coef={resolved_train_config.ent_coef} "
        f"clip_range={resolved_train_config.clip_range} "
        f"bc_epochs={resolved_train_config.bc_epochs} "
        f"bc_grids={resolved_train_config.bc_grid_count} "
        f"bc_size_min={resolved_train_config.bc_size_min or env_config.curriculum_min_size or 1} "
        f"bc_size_max={resolved_train_config.bc_size_max or env_config.size} "
        f"bc_batch_size={resolved_train_config.bc_batch_size} "
        f"bc_lr={resolved_train_config.bc_learning_rate} "
        f"final_target_only_timesteps={resolved_train_config.final_target_only_timesteps} "
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
