from __future__ import annotations

import math
from collections import deque
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.grid import create_random_grid
from src.rl_solver.config import ACTION_ORDER, EnvConfig
from src.shared_types import Grid, MOVE_DELTAS, Move, Path, Point
from src.solvers import find_start


class LawnMowingEnv(gym.Env[dict[str, np.ndarray], int]):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig):
        super().__init__()
        if config.size <= 0:
            raise ValueError("size must be positive")

        self.config = config
        self.action_space = spaces.Discrete(len(ACTION_ORDER))
        cell_shape = (config.size, config.size)
        self.observation_space = spaces.Dict(
            {
                "open_cells": spaces.Box(0, 1, shape=cell_shape, dtype=np.int8),
                "visited_cells": spaces.Box(0, 1, shape=cell_shape, dtype=np.int8),
                "visit_intensity": spaces.Box(
                    0.0, 1.0, shape=cell_shape, dtype=np.float32
                ),
                "agent_position": spaces.Box(0, 1, shape=cell_shape, dtype=np.int8),
                "previous_position": spaces.Box(0, 1, shape=cell_shape, dtype=np.int8),
                "progress_features": spaces.Box(0.0, 1.0, shape=(5,), dtype=np.float32),
                "action_features": spaces.Box(
                    0.0,
                    1.0,
                    shape=(len(ACTION_ORDER), 5),
                    dtype=np.float32,
                ),
            }
        )

        self.grid: Grid = []
        self.grid_seed = config.base_seed
        self.start: Point | None = None
        self.position: Point | None = None
        self.previous_position: Point | None = None
        self.open_cell_count = 0
        self.covered_cell_count = 0
        self.steps_taken = 0
        self.max_steps = 0
        self.moves: list[Move] = []
        self.visited_counts = np.zeros(cell_shape, dtype=np.int16)
        self.visited_mask = np.zeros(cell_shape, dtype=np.int8)
        self.max_cell_overlap_count = 0
        self.steps_since_last_new_cell = 0
        self.oscillation_streak = 0
        self.visit_count_log_denominator = 1.0
        self.position_history: deque[Point] = deque(maxlen=5)
        self.edge_traversal_counts: dict[tuple[Point, Point], int] = {}
        self.edge_reuse_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        provided_grid = options.get("grid")
        if provided_grid is not None:
            self.grid = [list(row) for row in provided_grid]
            self.grid_seed = int(options.get("grid_seed", self.config.base_seed))
        else:
            if seed is not None:
                self.grid_seed = int(seed)
            else:
                self.grid_seed = int(self.np_random.integers(0, 2**31 - 1))
            self.grid = create_random_grid(
                self.config.size,
                self.grid_seed,
                removed_fraction_range=(
                    self.config.removed_fraction_min,
                    self.config.removed_fraction_max,
                ),
            )

        self.start = find_start(self.grid)
        self.position = self.start
        self.previous_position = None
        self.open_cell_count = sum(cell == 1 for row in self.grid for cell in row)
        self.covered_cell_count = 0
        self.steps_taken = 0
        self.max_steps = max(
            1, math.ceil(self.open_cell_count * self.config.max_steps_factor)
        )
        self.moves = []
        self.visited_counts.fill(0)
        self.visited_mask.fill(0)
        self.max_cell_overlap_count = 0
        self.steps_since_last_new_cell = 0
        self.oscillation_streak = 0
        self.visit_count_log_denominator = float(math.log1p(max(1, self.max_steps + 1)))
        self.position_history.clear()
        self.edge_traversal_counts.clear()
        self.edge_reuse_count = 0

        if self.start is not None:
            row, col = self.start
            self.visited_counts[row, col] = 1
            self.visited_mask[row, col] = 1
            self.covered_cell_count = 1
            self.position_history.append(self.start)

        return self._get_obs(), self._get_info(invalid_action=False)

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self.position is None:
            raise RuntimeError("Environment must be reset before stepping")
        if self.covered_cell_count >= self.open_cell_count and self.open_cell_count > 0:
            return (
                self._get_obs(),
                self.config.reward_complete,
                True,
                False,
                self._get_info(invalid_action=False),
            )

        reward = 0.0
        terminated = False
        truncated = False
        invalid_action = False
        loop_oscillation = False
        reused_edge = False
        overlap_limit_hit = False
        frontier_distance_before = self._distance_to_nearest_unvisited(self.position)
        move = ACTION_ORDER[action]
        row, col = self.position
        d_row, d_col = MOVE_DELTAS[move]
        next_row = row + d_row
        next_col = col + d_col

        if self._is_valid_cell(next_row, next_col):
            next_position = (next_row, next_col)
            self.position = next_position
            self.previous_position = (row, col)
            self.moves.append(move)
            edge_key = self._get_edge_key((row, col), next_position)
            prior_edge_traversals = self.edge_traversal_counts.get(edge_key, 0)
            self.edge_traversal_counts[edge_key] = prior_edge_traversals + 1
            reused_edge = prior_edge_traversals > 0
            if reused_edge:
                self.edge_reuse_count += 1
                reward -= self._get_edge_reuse_penalty(prior_edge_traversals)
            prior_visits = int(self.visited_counts[next_row, next_col])
            revisit = prior_visits > 0
            self.visited_counts[next_row, next_col] += 1
            if revisit:
                cell_overlap_count = int(self.visited_counts[next_row, next_col] - 1)
                self.max_cell_overlap_count = max(
                    self.max_cell_overlap_count,
                    cell_overlap_count,
                )
                reward -= self._get_revisit_penalty(prior_visits)
                self.steps_since_last_new_cell += 1
            else:
                self.visited_mask[next_row, next_col] = 1
                self.covered_cell_count += 1
                reward += self.config.reward_new_cell
                self.steps_since_last_new_cell = 0

            self.position_history.append(next_position)
            loop_oscillation = self._update_oscillation_streak()
            if loop_oscillation:
                reward -= self._get_loop_penalty(self.oscillation_streak)

            frontier_distance_after = self._distance_to_nearest_unvisited(self.position)
            reward += self._get_frontier_progress_reward(
                frontier_distance_before,
                frontier_distance_after,
            )
            reward += self._get_stall_penalty()
        else:
            invalid_action = True
            reward += self.config.reward_invalid
            self.steps_since_last_new_cell += 1
            self.oscillation_streak = 0
            reward += self._get_stall_penalty()

        self.steps_taken += 1
        if self.covered_cell_count >= self.open_cell_count and self.open_cell_count > 0:
            reward += self.config.reward_complete
            terminated = True
        elif self.max_cell_overlap_count > self.config.max_overlaps:
            reward += self.config.reward_overlap_limit
            terminated = True
            overlap_limit_hit = True
        elif self.steps_taken >= self.max_steps:
            reward += self._get_timeout_penalty()
            truncated = True

        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            self._get_info(
                invalid_action=invalid_action,
                loop_oscillation=loop_oscillation,
                reused_edge=reused_edge,
                overlap_limit_hit=overlap_limit_hit,
            ),
        )

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(len(ACTION_ORDER), dtype=bool)
        if self.position is None:
            return mask
        if self.covered_cell_count >= self.open_cell_count and self.open_cell_count > 0:
            mask[:] = True
            return mask

        row, col = self.position
        for index, move in enumerate(ACTION_ORDER):
            d_row, d_col = MOVE_DELTAS[move]
            mask[index] = self._is_valid_cell(row + d_row, col + d_col)
        return mask

    def get_path(self) -> Path:
        return {"start": self.start, "moves": list(self.moves)}

    def _get_obs(self) -> dict[str, np.ndarray]:
        open_cells = np.asarray(self.grid, dtype=np.int8)
        agent_position = np.zeros_like(open_cells)
        previous_position = np.zeros_like(open_cells)
        if self.position is not None:
            row, col = self.position
            agent_position[row, col] = 1
        if self.previous_position is not None:
            row, col = self.previous_position
            previous_position[row, col] = 1

        visit_intensity = np.log1p(self.visited_counts.astype(np.float32))
        visit_intensity = np.clip(
            visit_intensity / self.visit_count_log_denominator,
            0.0,
            1.0,
        )
        progress_features = np.asarray(
            [
                self._coverage_ratio(),
                max(0, self.max_steps - self.steps_taken) / max(1, self.max_steps),
                min(1.0, self.steps_since_last_new_cell / max(1, self.max_steps)),
                self._frontier_distance_feature(),
                min(1.0, self.oscillation_streak / 4.0),
            ],
            dtype=np.float32,
        )
        action_features = self._get_action_features()

        return {
            "open_cells": open_cells,
            "visited_cells": self.visited_mask.copy(),
            "visit_intensity": visit_intensity,
            "agent_position": agent_position,
            "previous_position": previous_position,
            "progress_features": progress_features,
            "action_features": action_features,
        }

    def _get_info(
        self,
        *,
        invalid_action: bool,
        loop_oscillation: bool = False,
        reused_edge: bool = False,
        overlap_limit_hit: bool = False,
    ) -> dict[str, Any]:
        return {
            "grid_seed": self.grid_seed,
            "covered_cells": self.covered_cell_count,
            "open_cells": self.open_cell_count,
            "coverage_ratio": self._coverage_ratio(),
            "max_cell_overlap_count": self.max_cell_overlap_count,
            "max_overlaps": self.config.max_overlaps,
            "frontier_distance": self._distance_to_nearest_unvisited(self.position),
            "steps_taken": self.steps_taken,
            "steps_remaining": max(0, self.max_steps - self.steps_taken),
            "steps_since_last_new_cell": self.steps_since_last_new_cell,
            "oscillation_streak": self.oscillation_streak,
            "loop_oscillation": loop_oscillation,
            "edge_reuse_count": self.edge_reuse_count,
            "reused_edge": reused_edge,
            "overlap_limit_hit": overlap_limit_hit,
            "invalid_action": invalid_action,
        }

    def _get_revisit_penalty(self, prior_visits: int) -> float:
        revisit_penalty = abs(self.config.reward_revisit) * (
            self.config.reward_revisit_growth ** max(0, prior_visits - 1)
        )
        return min(revisit_penalty, self.config.reward_revisit_max_penalty)

    def _get_edge_reuse_penalty(self, prior_edge_traversals: int) -> float:
        edge_reuse_penalty = abs(self.config.reward_edge_reuse) * (
            self.config.reward_edge_reuse_growth ** max(0, prior_edge_traversals - 1)
        )
        return min(edge_reuse_penalty, self.config.reward_edge_reuse_max_penalty)

    def _get_timeout_penalty(self) -> float:
        if self.open_cell_count <= 0:
            return 0.0

        uncovered_ratio = 1.0 - self._coverage_ratio()
        return self.config.reward_timeout + (
            uncovered_ratio * self.config.reward_timeout_uncovered_scale
        )

    def _get_loop_penalty(self, oscillation_streak: int) -> float:
        loop_penalty = abs(self.config.reward_loop) * (
            self.config.reward_loop_growth ** max(0, oscillation_streak - 1)
        )
        return min(loop_penalty, self.config.reward_loop_max_penalty)

    def _get_frontier_progress_reward(
        self,
        frontier_distance_before: int | None,
        frontier_distance_after: int | None,
    ) -> float:
        if frontier_distance_before is None or frontier_distance_after is None:
            return 0.0

        distance_improvement = frontier_distance_before - frontier_distance_after
        if distance_improvement > 0:
            return self.config.reward_frontier_progress * distance_improvement
        if distance_improvement < 0:
            return abs(self.config.reward_frontier_regress) * distance_improvement
        return 0.0

    def _get_stall_penalty(self) -> float:
        excess_no_progress = (
            self.steps_since_last_new_cell - self.config.stall_grace_steps
        )
        if excess_no_progress <= 0:
            return 0.0
        return self.config.reward_stall_step * excess_no_progress

    def _update_oscillation_streak(self) -> bool:
        if len(self.position_history) < 5:
            self.oscillation_streak = 0
            return False

        latest_positions = list(self.position_history)
        first_point = latest_positions[-1]
        second_point = latest_positions[-2]
        alternating = (
            first_point != second_point
            and latest_positions[-1] == latest_positions[-3] == latest_positions[-5]
            and latest_positions[-2] == latest_positions[-4]
        )
        if not alternating:
            self.oscillation_streak = 0
            return False

        self.oscillation_streak += 1
        return True

    def _coverage_ratio(self) -> float:
        if self.open_cell_count <= 0:
            return 0.0
        return self.covered_cell_count / self.open_cell_count

    def _frontier_distance_feature(self) -> float:
        frontier_distance = self._distance_to_nearest_unvisited(self.position)
        if frontier_distance is None:
            return 0.0
        return min(1.0, frontier_distance / max(1, self.open_cell_count))

    def _get_action_features(self) -> np.ndarray:
        features = np.zeros((len(ACTION_ORDER), 5), dtype=np.float32)
        if self.position is None:
            return features

        row, col = self.position
        for index, move in enumerate(ACTION_ORDER):
            d_row, d_col = MOVE_DELTAS[move]
            next_row = row + d_row
            next_col = col + d_col
            if not self._is_valid_cell(next_row, next_col):
                continue

            next_point = (next_row, next_col)
            edge_key = self._get_edge_key((row, col), next_point)
            features[index, 0] = 1.0
            features[index, 1] = float(self.visited_mask[next_row, next_col] == 0)
            features[index, 2] = self._normalize_visit_count(
                int(self.visited_counts[next_row, next_col])
            )
            features[index, 3] = self._normalize_frontier_distance(
                self._distance_to_nearest_unvisited(next_point)
            )
            features[index, 4] = self._normalize_edge_count(
                self.edge_traversal_counts.get(edge_key, 0)
            )
        return features

    def _normalize_visit_count(self, visit_count: int) -> float:
        if visit_count <= 0:
            return 0.0
        return min(1.0, math.log1p(visit_count) / self.visit_count_log_denominator)

    def _normalize_frontier_distance(self, frontier_distance: int | None) -> float:
        if frontier_distance is None:
            return 0.0
        return min(1.0, frontier_distance / max(1, self.open_cell_count))

    def _normalize_edge_count(self, edge_count: int) -> float:
        if edge_count <= 0:
            return 0.0
        return min(1.0, math.log1p(edge_count) / math.log1p(4.0))

    def _get_edge_key(self, start: Point, end: Point) -> tuple[Point, Point]:
        return (start, end) if start <= end else (end, start)

    def _is_valid_cell(self, row: int, col: int) -> bool:
        size = len(self.grid)
        return 0 <= row < size and 0 <= col < size and self.grid[row][col] == 1

    def _distance_to_nearest_unvisited(self, start: Point | None) -> int | None:
        if start is None:
            return None
        if self.covered_cell_count >= self.open_cell_count:
            return None

        rows = len(self.grid)
        cols = len(self.grid[0]) if rows else 0
        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            (row, col), distance = queue.popleft()
            if self.grid[row][col] == 1 and self.visited_mask[row, col] == 0:
                return distance

            for d_row, d_col in MOVE_DELTAS.values():
                next_row = row + d_row
                next_col = col + d_col
                next_point = (next_row, next_col)
                if not (0 <= next_row < rows and 0 <= next_col < cols):
                    continue
                if self.grid[next_row][next_col] == 0 or next_point in visited:
                    continue
                visited.add(next_point)
                queue.append((next_point, distance + 1))

        return None
