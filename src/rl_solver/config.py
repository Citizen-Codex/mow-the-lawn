from dataclasses import asdict, dataclass, fields, replace
import json
import math
from pathlib import Path


ACTION_ORDER = ("u", "d", "l", "r")


@dataclass(slots=True)
class EnvConfig:
    size: int
    removed_fraction_min: float = 0.18
    removed_fraction_max: float = 0.42
    reward_new_cell: float = 10
    # reward_revisit: float = -0.02
    reward_revisit: float = -0.2
    reward_revisit_growth: float = 1.25
    # reward_revisit_max_penalty: float = 0.5
    reward_revisit_max_penalty: float = 2.5
    reward_edge_reuse: float = -0.03
    reward_edge_reuse_growth: float = 1.4
    reward_edge_reuse_max_penalty: float = 0.6
    reward_loop: float = -0.15
    reward_loop_growth: float = 1.5
    reward_loop_max_penalty: float = 1.0
    # reward_frontier_progress: float = 0.08
    reward_frontier_progress: float = 0.0
    # reward_frontier_regress: float = -0.02
    reward_frontier_regress: float = -0.0
    reward_stall_step: float = -0.01
    stall_grace_steps: int = 8
    reward_complete: float = 1000.0
    reward_invalid: float = -1.0
    max_overlaps: int = 2
    reward_overlap_limit: float = -20.0
    reward_timeout: float = -5.0
    reward_timeout_uncovered_scale: float = -15.0
    max_steps_factor: float = 2.0
    base_seed: int = 0


@dataclass(slots=True)
class TrainConfig:
    total_timesteps: int = 1_000_000
    n_envs: int = 16
    learning_rate: float = 3e-4
    n_steps: int | None = None
    batch_size: int | None = None
    n_epochs: int | None = None
    gamma: float = 0.995
    gae_lambda: float = 0.95
    ent_coef: float = 0.005
    clip_range: float = 0.2
    seed: int = 0
    device: str = "auto"
    auto_tune: bool = True
    eval_freq_timesteps: int = 20_000
    eval_seed_count: int = 10
    save_best_checkpoint: bool = True


def _filter_config_payload(data: dict, config_type: type) -> dict:
    allowed = {field.name for field in fields(config_type)}
    return {key: value for key, value in data.items() if key in allowed}


def make_env_config(data: dict) -> EnvConfig:
    return EnvConfig(**_filter_config_payload(data, EnvConfig))


def make_train_config(data: dict) -> TrainConfig:
    return TrainConfig(**_filter_config_payload(data, TrainConfig))


def env_configs_resume_compatible(saved: EnvConfig, current: EnvConfig) -> bool:
    return saved.size == current.size


def _pick_batch_size(total_batch_size: int, limit: int = 2048) -> int:
    for candidate in (2048, 1024, 512, 256, 128, 64, 32):
        if (
            candidate <= limit
            and candidate <= total_batch_size
            and total_batch_size % candidate == 0
        ):
            return candidate
    return min(total_batch_size, limit)


def _default_n_steps(n_envs: int) -> int:
    if n_envs <= 8:
        return 512
    if n_envs <= 32:
        return 256
    return 128


def _default_n_epochs(rollout_batch: int) -> int:
    if rollout_batch >= 8192:
        return 4
    if rollout_batch >= 4096:
        return 6
    return 8


def resolve_train_config(train_config: TrainConfig) -> TrainConfig:
    max_n_steps = max(
        1, math.ceil(train_config.total_timesteps / max(1, train_config.n_envs))
    )
    if not train_config.auto_tune:
        n_steps = min(
            train_config.n_steps or _default_n_steps(train_config.n_envs), max_n_steps
        )
        rollout_batch = n_steps * max(1, train_config.n_envs)
        return replace(
            train_config,
            n_steps=n_steps,
            batch_size=min(
                train_config.batch_size or _pick_batch_size(rollout_batch),
                rollout_batch,
            ),
            n_epochs=train_config.n_epochs or _default_n_epochs(rollout_batch),
        )

    n_steps = train_config.n_steps or _default_n_steps(train_config.n_envs)
    n_steps = min(n_steps, max_n_steps)
    rollout_batch = n_steps * max(1, train_config.n_envs)
    batch_size = train_config.batch_size or _pick_batch_size(rollout_batch)
    n_epochs = train_config.n_epochs or _default_n_epochs(rollout_batch)
    return replace(
        train_config, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs
    )


def save_run_config(
    model_path: str | Path, env_config: EnvConfig, train_config: TrainConfig
) -> Path:
    output_path = Path(model_path)
    config_path = output_path.with_suffix(".config.json")
    payload = {
        "env": asdict(env_config),
        "train": asdict(train_config),
        "action_order": list(ACTION_ORDER),
    }
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return config_path


def load_run_config(model_path: str | Path) -> dict:
    output_path = Path(model_path)
    config_path = output_path.with_suffix(".config.json")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing model config: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))
