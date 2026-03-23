from __future__ import annotations

from dataclasses import replace
from pathlib import Path as FilePath
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

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
    if len(grid) != env_config.size:
        raise ValueError(f"Model expects grid size {env_config.size}, got {len(grid)}")

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
        env = LawnMowingEnv(env_config)
        obs, _ = env.reset(seed=seed)
        path = _rollout_env_with_model(model, env, obs, deterministic=deterministic)
        rows.append(path_metrics(env.grid, path))
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


def train_maskable_ppo(
    env_config: EnvConfig,
    train_config: TrainConfig,
    model_path: str | FilePath,
    *,
    resume_from: str | FilePath | None = None,
) -> FilePath:
    train_config = resolve_train_config(train_config)
    policy_kwargs = {"net_arch": {"pi": [1024, 1024], "vf": [1024, 1024]}}
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

    vec_env = build_vec_env(env_config, train_config)
    eval_callback = CoverageEvalCallback(
        env_config=env_config,
        train_config=train_config,
        model_path=model_path,
    )
    print(
        "Training distribution: "
        f"removed_fraction=[{env_config.removed_fraction_min:.2f}, "
        f"{env_config.removed_fraction_max:.2f}] "
        f"max_overlaps={env_config.max_overlaps} "
        f"timesteps={train_config.total_timesteps}"
    )

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

    model.learn(
        total_timesteps=train_config.total_timesteps,
        callback=eval_callback,
        progress_bar=False,
        reset_num_timesteps=reset_num_timesteps,
    )
    best_summary = eval_callback.best_summary
    best_model_path = eval_callback.best_model_path
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
