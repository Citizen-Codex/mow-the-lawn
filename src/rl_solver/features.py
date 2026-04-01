from __future__ import annotations

from gymnasium import spaces
import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


GRID_OBSERVATION_KEYS = (
    "open_cells",
    "visited_cells",
    "visit_intensity",
    "agent_position",
    "previous_position",
)


class LawnMowingFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        *,
        features_dim: int = 512,
        scalar_hidden_dim: int = 128,
    ):
        super().__init__(observation_space, features_dim)

        progress_shape = observation_space["progress_features"].shape
        action_shape = observation_space["action_features"].shape
        scalar_input_dim = int(progress_shape[0] + action_shape[0] * action_shape[1])

        self.grid_encoder = nn.Sequential(
            nn.Conv2d(len(GRID_OBSERVATION_KEYS), 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
        )
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_input_dim, scalar_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(scalar_hidden_dim),
        )
        self.combined_encoder = nn.Sequential(
            nn.Linear((64 * 3 * 3) + scalar_hidden_dim, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim),
        )

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        grid_tensor = th.stack(
            [observations[key].float() for key in GRID_OBSERVATION_KEYS],
            dim=1,
        )
        grid_features = self.grid_encoder(grid_tensor)
        scalar_features = th.cat(
            [
                observations["progress_features"].float(),
                observations["action_features"].float().flatten(start_dim=1),
            ],
            dim=1,
        )
        encoded_scalar_features = self.scalar_encoder(scalar_features)
        return self.combined_encoder(
            th.cat([grid_features, encoded_scalar_features], dim=1)
        )
