from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from src.rl_solver.config import ACTION_ORDER, EnvConfig, TrainConfig
from src.rl_solver.env import LawnMowingEnv
from src.shared_types import Grid, Path as SolverPath


ACTION_TO_INDEX = {move: index for index, move in enumerate(ACTION_ORDER)}
DEFAULT_OPTIMAL_LIBRARY_PATH = (
    Path(__file__).resolve().parents[1] / "optimal" / "library" / "optimal_paths.csv"
)


@dataclass(slots=True)
class ExpertDataset:
    observations: dict[str, np.ndarray]
    action_masks: np.ndarray
    actions: np.ndarray
    grid_count: int
    transition_count: int
    avg_grid_size: float
    avg_path_length: float
    removed_fraction_range: tuple[float, float]


def _behavior_cloning_removed_fraction_range(
    env_config: EnvConfig, train_config: TrainConfig
) -> tuple[float, float]:
    min_fraction = (
        env_config.removed_fraction_min
        if train_config.bc_removed_fraction_min is None
        else train_config.bc_removed_fraction_min
    )
    max_fraction = (
        env_config.removed_fraction_max
        if train_config.bc_removed_fraction_max is None
        else train_config.bc_removed_fraction_max
    )
    if min_fraction < 0 or max_fraction > 1 or min_fraction > max_fraction:
        raise ValueError(
            "Behavior cloning removed-fraction range must satisfy 0 <= min <= max <= 1"
        )
    return (min_fraction, max_fraction)


def _behavior_cloning_size_range(
    env_config: EnvConfig,
    train_config: TrainConfig,
) -> tuple[int, int]:
    min_size = (
        env_config.curriculum_min_size or 1
        if train_config.bc_size_min is None
        else train_config.bc_size_min
    )
    max_size = (
        env_config.size
        if train_config.bc_size_max is None
        else train_config.bc_size_max
    )
    if min_size <= 0 or max_size < min_size or max_size > env_config.size:
        raise ValueError(
            "Behavior cloning size range must satisfy 1 <= min <= max <= env size"
        )
    return (min_size, max_size)


def _deserialize_grid(grid_text: str) -> Grid:
    rows = [row for row in grid_text.split("/") if row]
    return [[int(cell) for cell in row] for row in rows]


def _deserialize_path(row: dict[str, str]) -> SolverPath:
    start_row = int(row["start_row"])
    start_col = int(row["start_col"])
    start = None if start_row < 0 or start_col < 0 else (start_row, start_col)
    return {
        "start": start,
        "moves": list(row["path"]),
    }


def _load_optimal_library_rows(
    env_config: EnvConfig,
    train_config: TrainConfig,
) -> list[dict[str, str]]:
    library_path = DEFAULT_OPTIMAL_LIBRARY_PATH
    if not library_path.exists():
        raise FileNotFoundError(f"Missing optimal-path library: {library_path}")

    size_range = _behavior_cloning_size_range(env_config, train_config)
    removed_fraction_range = _behavior_cloning_removed_fraction_range(
        env_config,
        train_config,
    )
    rows: list[dict[str, str]] = []
    with library_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            size = int(row["size"])
            if size < size_range[0] or size > size_range[1]:
                continue

            open_cells = int(row["open_cells"])
            removed_fraction = 1.0 - (open_cells / max(1, size * size))
            if (
                removed_fraction < removed_fraction_range[0]
                or removed_fraction > removed_fraction_range[1]
            ):
                continue

            rows.append(row)

    if not rows:
        raise ValueError(
            "No optimal-library rows matched the requested behavior-cloning filters: "
            f"sizes=[{size_range[0]}, {size_range[1]}] "
            f"removed_fraction=[{removed_fraction_range[0]:.2f}, {removed_fraction_range[1]:.2f}]"
        )
    return rows


def build_optimal_expert_dataset(
    env_config: EnvConfig,
    train_config: TrainConfig,
) -> ExpertDataset | None:
    if train_config.bc_epochs <= 0 or train_config.bc_grid_count <= 0:
        return None

    removed_fraction_range = _behavior_cloning_removed_fraction_range(
        env_config, train_config
    )
    library_rows = _load_optimal_library_rows(env_config, train_config)
    library_offset = max(0, train_config.bc_seed_start)
    if library_offset >= len(library_rows):
        raise ValueError(
            "Behavior-cloning library offset exceeds the number of matching rows: "
            f"offset={library_offset} matches={len(library_rows)}"
        )
    eligible_rows = library_rows[library_offset:]
    if train_config.bc_grid_count > len(eligible_rows):
        raise ValueError(
            "Not enough optimal-library rows for behavior cloning: "
            f"requested={train_config.bc_grid_count} available={len(eligible_rows)}"
        )

    action_masks: list[np.ndarray] = []
    actions: list[int] = []
    observation_rows: dict[str, list[np.ndarray]] | None = None
    observation_template: dict[str, np.ndarray] | None = None
    path_lengths: list[int] = []
    grid_sizes: list[int] = []
    size_rng = np.random.default_rng(train_config.seed)

    row_indices = size_rng.permutation(len(eligible_rows))[: train_config.bc_grid_count]
    selected_rows = [eligible_rows[int(index)] for index in row_indices]

    for row in selected_rows:
        seed = int(row["seed"])
        grid = _deserialize_grid(row["grid"])
        expert_path = _deserialize_path(row)
        grid_sizes.append(len(grid))
        open_cell_count = sum(cell == 1 for row in grid for cell in row)
        required_max_steps_factor = (len(expert_path["moves"]) + 1) / max(
            1, open_cell_count
        )
        expert_env_config = replace(
            env_config,
            max_overlaps=max(env_config.max_overlaps, len(expert_path["moves"]) + 1),
            max_steps_factor=max(
                env_config.max_steps_factor, required_max_steps_factor
            ),
        )
        env = LawnMowingEnv(expert_env_config)
        observation, _ = env.reset(options={"grid": grid, "grid_seed": seed})
        if observation_rows is None:
            observation_rows = {key: [] for key in observation}
            observation_template = {
                key: value.copy() for key, value in observation.items()
            }

        if len(expert_path["moves"]) != int(row["moves"]):
            raise RuntimeError(
                "Optimal-library path length does not match moves column: "
                f"seed={seed} expected={row['moves']} actual={len(expert_path['moves'])}"
            )

        path_lengths.append(len(expert_path["moves"]))
        for move_index, move in enumerate(expert_path["moves"]):
            action = ACTION_TO_INDEX[move]
            action_mask = env.action_masks()
            if not action_mask[action]:
                raise RuntimeError(
                    f"Optimal expert emitted invalid action {move!r} on seed {seed}"
                )

            for key, value in observation.items():
                observation_rows[key].append(value.copy())
            action_masks.append(action_mask.copy())
            actions.append(action)

            observation, _, terminated, truncated, _ = env.step(action)
            if truncated:
                raise RuntimeError(
                    f"Optimal expert rollout truncated on seed {seed} after {move_index + 1} moves"
                )
            if terminated and move_index != len(expert_path["moves"]) - 1:
                raise RuntimeError(
                    f"Optimal expert terminated early on seed {seed} after {move_index + 1} moves"
                )

        if env.covered_cell_count != env.open_cell_count:
            raise RuntimeError(
                f"Optimal expert failed to cover all open cells on seed {seed}"
            )

    if observation_rows is None or observation_template is None:
        return None

    stacked_observations = {
        key: (
            np.stack(values, axis=0)
            if values
            else np.empty(
                (0, *observation_template[key].shape),
                dtype=observation_template[key].dtype,
            )
        )
        for key, values in observation_rows.items()
    }
    stacked_action_masks = (
        np.stack(action_masks, axis=0)
        if action_masks
        else np.empty((0, len(ACTION_ORDER)), dtype=bool)
    )
    stacked_actions = np.asarray(actions, dtype=np.int64)
    avg_grid_size = sum(grid_sizes) / max(1, len(grid_sizes))
    avg_path_length = sum(path_lengths) / max(1, len(path_lengths))
    return ExpertDataset(
        observations=stacked_observations,
        action_masks=stacked_action_masks,
        actions=stacked_actions,
        grid_count=train_config.bc_grid_count,
        transition_count=int(stacked_actions.shape[0]),
        avg_grid_size=avg_grid_size,
        avg_path_length=avg_path_length,
        removed_fraction_range=removed_fraction_range,
    )


def run_behavior_cloning_pretrain(
    model: MaskablePPO,
    dataset: ExpertDataset,
    train_config: TrainConfig,
) -> None:
    if dataset.transition_count <= 0:
        print(
            "Skipping behavior cloning pretrain: no expert transitions were collected"
        )
        return

    policy = model.policy
    was_training = policy.training
    policy.set_training_mode(True)
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=train_config.bc_learning_rate,
    )
    batch_size = max(1, min(train_config.bc_batch_size, dataset.transition_count))
    rng = np.random.default_rng(train_config.seed)

    print(
        "Behavior cloning pretrain: "
        f"grids={dataset.grid_count} "
        f"samples={dataset.transition_count} "
        f"epochs={train_config.bc_epochs} "
        f"batch_size={batch_size} "
        f"lr={train_config.bc_learning_rate} "
        f"avg_grid_size={dataset.avg_grid_size:.1f} "
        f"avg_path_length={dataset.avg_path_length:.1f} "
        f"removed_fraction=[{dataset.removed_fraction_range[0]:.2f}, "
        f"{dataset.removed_fraction_range[1]:.2f}]"
    )

    for epoch in range(1, train_config.bc_epochs + 1):
        permutation = rng.permutation(dataset.transition_count)
        epoch_loss = 0.0
        epoch_correct = 0

        for batch_start in range(0, dataset.transition_count, batch_size):
            batch_indices = permutation[batch_start : batch_start + batch_size]
            batch_observations = {
                key: torch.as_tensor(values[batch_indices], device=policy.device)
                for key, values in dataset.observations.items()
            }
            batch_action_masks = torch.as_tensor(
                dataset.action_masks[batch_indices],
                device=policy.device,
                dtype=torch.bool,
            )
            batch_actions = torch.as_tensor(
                dataset.actions[batch_indices],
                device=policy.device,
                dtype=torch.long,
            )

            distribution = policy.get_distribution(
                batch_observations,
                action_masks=batch_action_masks,
            )
            loss = -distribution.log_prob(batch_actions).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()

            batch_count = int(batch_actions.shape[0])
            epoch_loss += float(loss.detach().item()) * batch_count
            predicted_actions = distribution.distribution.probs.argmax(dim=1)
            epoch_correct += int((predicted_actions == batch_actions).sum().item())

        avg_loss = epoch_loss / dataset.transition_count
        accuracy = epoch_correct / dataset.transition_count
        print(
            "Behavior cloning epoch: "
            f"{epoch}/{train_config.bc_epochs} "
            f"loss={avg_loss:.4f} "
            f"accuracy={accuracy:.3f}"
        )

    policy.set_training_mode(was_training)
