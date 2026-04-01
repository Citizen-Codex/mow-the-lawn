from __future__ import annotations

from dataclasses import replace
from pathlib import Path as FilePath
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from src.grid import create_random_grid
from src.rl_solver.config import (
    EnvConfig,
    TrainConfig,
    env_configs_resume_compatible,
    load_run_config,
    make_env_config,
    make_train_config,
    resolve_train_config,
    save_run_config,
)
from src.rl_solver.env import LawnMowingEnv
from src.rl_solver.expert import (
    build_optimal_expert_dataset,
    run_behavior_cloning_pretrain,
)
from src.rl_solver.features import LawnMowingFeaturesExtractor
from src.rl_solver.metrics import (
    SummaryStats,
    path_metrics,
    summarize_metric_rows,
    summary_sort_key,
    format_summary,
)
from src.shared_types import Grid, Path


def _mask_fn(env: LawnMowingEnv):
    return env.unwrapped.action_masks()


def _changed_env_fields(saved: EnvConfig, current: EnvConfig) -> list[str]:
    changed: list[str] = []
    for field_name in saved.__dataclass_fields__:
        if getattr(saved, field_name) != getattr(current, field_name):
            changed.append(
                f"{field_name}={getattr(saved, field_name)}->{getattr(current, field_name)}"
            )
    return changed


def make_masked_env(env_config: EnvConfig, *, seed_offset: int = 0) -> ActionMasker:
    config = replace(env_config, base_seed=env_config.base_seed + seed_offset)
    env = LawnMowingEnv(config)
    monitored_env = Monitor(env)
    return ActionMasker(monitored_env, _mask_fn)


def build_vec_env(env_config: EnvConfig, train_config: TrainConfig) -> VecEnv:
    env_fns = [
        (lambda offset=offset: make_masked_env(env_config, seed_offset=offset))
        for offset in range(train_config.n_envs)
    ]
    return DummyVecEnv(env_fns)


def _policy_kwargs() -> dict:
    return {
        "features_extractor_class": LawnMowingFeaturesExtractor,
        "features_extractor_kwargs": {
            "features_dim": 512,
            "scalar_hidden_dim": 128,
        },
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
    }


def _rollout_env_with_model(
    model: MaskablePPO,
    env: LawnMowingEnv,
    obs: dict,
    *,
    deterministic: bool = True,
) -> Path:
    while True:
        action, _ = model.predict(
            obs,
            action_masks=env.action_masks(),
            deterministic=deterministic,
        )
        obs, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            break
    return env.get_path()


def solve_grid_with_loaded_model(
    model: MaskablePPO,
    env_config: EnvConfig,
    grid: Grid,
    *,
    deterministic: bool = True,
) -> Path:
    if len(grid) > env_config.size:
        raise ValueError(
            f"Model supports grids up to size {env_config.size}, got {len(grid)}"
        )

    env = LawnMowingEnv(env_config)
    obs, _ = env.reset(options={"grid": grid})
    return _rollout_env_with_model(model, env, obs, deterministic=deterministic)


def _evaluate_model_on_seeds(
    model: MaskablePPO,
    env_config: EnvConfig,
    seeds: range,
    *,
    deterministic: bool = True,
) -> SummaryStats:
    rows: list[dict[str, int | float | bool]] = []
    for seed in seeds:
        grid = create_random_grid(
            env_config.size,
            seed,
            removed_fraction_range=(
                env_config.removed_fraction_min,
                env_config.removed_fraction_max,
            ),
        )
        env = LawnMowingEnv(env_config)
        obs, _ = env.reset(options={"grid": grid, "grid_seed": seed})
        path = _rollout_env_with_model(model, env, obs, deterministic=deterministic)
        rows.append(path_metrics(grid, path))
    return summarize_metric_rows(rows)


class CoverageEvalCallback(BaseCallback):
    def __init__(
        self,
        *,
        env_config: EnvConfig,
        train_config: TrainConfig,
        model_path: str | FilePath,
        best_score: tuple[float, float, float, float] | None = None,
        best_summary: SummaryStats | None = None,
    ):
        super().__init__()
        self.env_config = env_config
        self.train_config = train_config
        self.eval_freq_steps = max(
            1,
            train_config.eval_freq_timesteps // max(1, train_config.n_envs),
        )
        eval_seed_start = train_config.seed + 10_000
        self.eval_seeds = range(
            eval_seed_start,
            eval_seed_start + max(1, train_config.eval_seed_count),
        )
        self.best_score = best_score or (-1.0, -1.0, float("-inf"), float("-inf"))
        self.best_summary = best_summary
        output_path = FilePath(model_path)
        self.best_model_path = output_path.with_name(
            f"{output_path.stem}.best{output_path.suffix}"
        )

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq_steps != 0:
            return True

        summary = _evaluate_model_on_seeds(
            self.model,
            self.env_config,
            self.eval_seeds,
            deterministic=True,
        )
        score = summary_sort_key(summary)
        improved = score > self.best_score
        if improved:
            self.best_score = score
            self.best_summary = summary
            if self.train_config.save_best_checkpoint:
                self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(self.best_model_path)
                save_run_config(
                    self.best_model_path,
                    self.env_config,
                    self.train_config,
                )

        print(
            "Held-out eval: "
            f"timesteps={self.num_timesteps} "
            f"{format_summary(summary)} "
            f"best={'yes' if improved else 'no'}"
        )
        return True


def _run_training_stage(
    model: MaskablePPO,
    vec_env: VecEnv,
    env_config: EnvConfig,
    train_config: TrainConfig,
    model_path: str | FilePath,
    *,
    total_timesteps: int,
    reset_num_timesteps: bool,
    best_score: tuple[float, float, float, float] | None,
    best_summary: SummaryStats | None,
    label: str,
) -> tuple[tuple[float, float, float, float] | None, SummaryStats | None]:
    if total_timesteps <= 0:
        return best_score, best_summary

    model.set_env(vec_env)
    eval_callback = CoverageEvalCallback(
        env_config=env_config,
        train_config=train_config,
        model_path=model_path,
        best_score=best_score,
        best_summary=best_summary,
    )
    print(
        "Training stage: "
        f"{label} "
        f"timesteps={total_timesteps} "
        f"grid_size=[{env_config.curriculum_min_size or env_config.size}, "
        f"{env_config.size}]"
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=False,
        reset_num_timesteps=reset_num_timesteps,
    )
    return eval_callback.best_score, eval_callback.best_summary


def train_maskable_ppo(
    env_config: EnvConfig,
    train_config: TrainConfig,
    model_path: str | FilePath,
    *,
    resume_from: str | FilePath | None = None,
) -> FilePath:
    train_config = resolve_train_config(train_config)
    policy_kwargs = _policy_kwargs()
    if resume_from is not None:
        resume_path = FilePath(resume_from)
        resume_config = load_run_config(resume_path)
        resume_env_config = make_env_config(resume_config["env"])
        if not env_configs_resume_compatible(resume_env_config, env_config):
            raise ValueError(
                "Resume checkpoint is only compatible with the same grid size"
            )

        resume_train_config = resolve_train_config(
            make_train_config(resume_config["train"])
        )
        print(
            "Resuming from checkpoint: "
            f"{resume_path} "
            f"saved_size={resume_env_config.size} "
            f"saved_n_steps={resume_train_config.n_steps} "
            f"saved_batch_size={resume_train_config.batch_size} "
            f"saved_n_epochs={resume_train_config.n_epochs}"
        )
        changed_fields = _changed_env_fields(resume_env_config, env_config)
        if changed_fields:
            print(
                "Applying updated env config while resuming: "
                + ", ".join(changed_fields)
            )
        model = MaskablePPO.load(resume_path, device=train_config.device)
    else:
        model = None

    print(
        "Training distribution: "
        f"grid_size=[{env_config.curriculum_min_size or env_config.size}, "
        f"{env_config.size}] "
        f"curriculum_warmup={env_config.curriculum_warmup_episodes} "
        f"target_size_prob={env_config.curriculum_target_size_probability:.2f} "
        f"size_bias_power={env_config.curriculum_size_bias_power:.2f} "
        f"removed_fraction=[{env_config.removed_fraction_min:.2f}, "
        f"{env_config.removed_fraction_max:.2f}] "
        f"max_overlaps={env_config.max_overlaps} "
        f"terminate_on_overlap_limit={env_config.terminate_on_overlap_limit} "
        f"final_target_only_timesteps={min(train_config.total_timesteps, train_config.final_target_only_timesteps)} "
        f"timesteps={train_config.total_timesteps}"
    )

    vec_env = build_vec_env(env_config, train_config)
    if model is None:
        model = MaskablePPO(
            "MultiInputPolicy",
            vec_env,
            policy_kwargs=policy_kwargs,
            learning_rate=train_config.learning_rate,
            n_steps=train_config.n_steps,
            batch_size=train_config.batch_size,
            n_epochs=train_config.n_epochs,
            gamma=train_config.gamma,
            gae_lambda=train_config.gae_lambda,
            ent_coef=train_config.ent_coef,
            clip_range=train_config.clip_range,
            seed=train_config.seed,
            device=train_config.device,
            verbose=1,
        )
        reset_num_timesteps = True
    else:
        model.set_env(vec_env)
        reset_num_timesteps = False

    expert_dataset = build_optimal_expert_dataset(env_config, train_config)
    if expert_dataset is not None:
        if expert_dataset.avg_grid_size < env_config.size:
            print(
                "Behavior cloning note: "
                f"avg_expert_grid_size={expert_dataset.avg_grid_size:.1f} "
                f"model_max_size={env_config.size} "
                "consider fewer BC epochs or a larger expert library if PPO stalls on large grids"
            )
        run_behavior_cloning_pretrain(model, expert_dataset, train_config)

    initial_timesteps = max(
        0,
        train_config.total_timesteps
        - min(train_config.total_timesteps, train_config.final_target_only_timesteps),
    )
    final_target_only_timesteps = min(
        train_config.total_timesteps,
        train_config.final_target_only_timesteps,
    )
    best_score: tuple[float, float, float, float] | None = None
    best_summary: SummaryStats | None = None
    best_model_path = FilePath(model_path).with_name(
        f"{FilePath(model_path).stem}.best{FilePath(model_path).suffix}"
    )

    try:
        best_score, best_summary = _run_training_stage(
            model,
            vec_env,
            env_config,
            train_config,
            model_path,
            total_timesteps=initial_timesteps,
            reset_num_timesteps=reset_num_timesteps,
            best_score=best_score,
            best_summary=best_summary,
            label="curriculum/main",
        )

        if final_target_only_timesteps > 0:
            vec_env.close()
            target_only_env_config = replace(
                env_config,
                curriculum_min_size=env_config.size,
                curriculum_warmup_episodes=0,
                curriculum_target_size_probability=1.0,
            )
            vec_env = build_vec_env(target_only_env_config, train_config)
            best_score, best_summary = _run_training_stage(
                model,
                vec_env,
                target_only_env_config,
                train_config,
                model_path,
                total_timesteps=final_target_only_timesteps,
                reset_num_timesteps=False,
                best_score=best_score,
                best_summary=best_summary,
                label="target-size-only",
            )
    finally:
        vec_env.close()

    if model is None:
        raise RuntimeError("Failed to initialize PPO model")

    output_path = FilePath(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    save_run_config(output_path, env_config, train_config)
    if (
        best_model_path is not None
        and best_model_path.exists()
        and best_summary is not None
    ):
        print(
            "Saved best held-out checkpoint to "
            f"{best_model_path} "
            f"(success={best_summary['success_rate']:.3f} "
            f"coverage={best_summary['avg_coverage']:.3f})"
        )
    return output_path


def load_maskable_ppo(model_path: str | FilePath) -> tuple[MaskablePPO, dict]:
    model = MaskablePPO.load(model_path)
    config = load_run_config(model_path)
    return model, config


def solve_grid_with_model(
    model_path: str | FilePath,
    grid: Grid,
    *,
    deterministic: bool = True,
) -> Path:
    model, config = load_maskable_ppo(model_path)
    return solve_grid_with_loaded_model(
        model,
        make_env_config(config["env"]),
        grid,
        deterministic=deterministic,
    )
