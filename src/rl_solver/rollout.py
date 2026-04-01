from __future__ import annotations

from pathlib import Path
from typing import Any

from src.grid import create_random_grid
from src.rl_solver.config import load_run_config, make_env_config
from src.rl_solver.env import LawnMowingEnv
from src.rl_solver.metrics import RolloutResult, path_metrics
from src.rl_solver.model import load_maskable_ppo
from sb3_contrib import MaskablePPO
from src.rl_solver.config import EnvConfig
from src.shared_types import Grid, Path


def rollout_loaded_model_on_grid(
    model: MaskablePPO,
    env_config: EnvConfig,
    grid: Grid,
    *,
    deterministic: bool = True,
) -> RolloutResult:
    if len(grid) > env_config.size:
        raise ValueError(
            f"Model supports grids up to size {env_config.size}, got {len(grid)}"
        )

    env = LawnMowingEnv(env_config)
    obs, info = env.reset(options={"grid": grid})
    total_reward = 0.0
    invalid_actions = 0
    loop_oscillations = 0
    max_oscillation_streak = 0
    edge_reuses = 0
    max_cell_overlap_count = 0
    overlap_limit_hit = False
    terminated = False
    truncated = False

    while True:
        action, _ = model.predict(
            obs,
            action_masks=env.action_masks(),
            deterministic=deterministic,
        )
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += float(reward)
        invalid_actions += int(info["invalid_action"])
        loop_oscillations += int(info.get("loop_oscillation", False))
        max_oscillation_streak = max(
            max_oscillation_streak,
            int(info.get("oscillation_streak", 0)),
        )
        edge_reuses = int(info.get("edge_reuse_count", edge_reuses))
        max_cell_overlap_count = int(
            info.get("max_cell_overlap_count", max_cell_overlap_count)
        )
        overlap_limit_hit = overlap_limit_hit or bool(
            info.get("overlap_limit_hit", False)
        )
        if terminated or truncated:
            break

    path = env.get_path()
    return {
        "path": path,
        "metrics": path_metrics(grid, path),
        "total_reward": total_reward,
        "invalid_actions": invalid_actions,
        "loop_oscillations": loop_oscillations,
        "max_oscillation_streak": max_oscillation_streak,
        "edge_reuses": edge_reuses,
        "max_cell_overlap_count": max_cell_overlap_count,
        "overlap_limit_hit": overlap_limit_hit,
        "terminated": terminated,
        "truncated": truncated,
        "info": info,
    }


def rollout_model_on_grid(
    model_path: str | Path,
    grid: Grid,
    *,
    deterministic: bool = True,
) -> RolloutResult:
    model, config = load_maskable_ppo(model_path)
    return rollout_loaded_model_on_grid(
        model,
        make_env_config(config["env"]),
        grid,
        deterministic=deterministic,
    )


def rollout_model_on_seed(
    model_path: str | Path,
    seed: int,
    *,
    size: int | None = None,
    deterministic: bool = True,
) -> dict[str, Any]:
    config = load_run_config(model_path)
    env_config = make_env_config(config["env"])
    grid_size = int(env_config.size if size is None else size)
    grid = create_random_grid(
        grid_size,
        seed,
        removed_fraction_range=(
            env_config.removed_fraction_min,
            env_config.removed_fraction_max,
        ),
    )
    result = rollout_model_on_grid(model_path, grid, deterministic=deterministic)
    result["grid"] = grid
    result["seed"] = seed
    return result


def extract_path(result: dict[str, Any]) -> Path:
    return result["path"]
