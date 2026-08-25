"""PyTorch Q-network for Independent DQN with parameter sharing."""

from __future__ import annotations

import torch
from torch import nn


class DQNPolicyNetwork(nn.Module):
    """Multi-layer perceptron that maps state observations to Q-values."""

    def __init__(self, observation_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return raw Q-values; no softmax is used for DQN."""
        return self.net(observations.float())
