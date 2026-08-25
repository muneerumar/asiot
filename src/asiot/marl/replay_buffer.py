"""Replay buffer for masked DQN training."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np


@dataclass(frozen=True)
class Transition:
    """One Independent-DQN transition."""

    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool
    action_mask: np.ndarray
    next_action_mask: np.ndarray


class ReplayBuffer:
    """Fixed-capacity FIFO replay memory."""

    def __init__(self, capacity: int, seed: int | None = None) -> None:
        self.capacity = int(capacity)
        self.memory: deque[Transition] = deque(maxlen=self.capacity)
        self.rng = random.Random(seed)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
        next_action_mask: np.ndarray,
    ) -> None:
        """Store a transition."""
        self.memory.append(
            Transition(
                np.asarray(obs, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_obs, dtype=np.float32),
                bool(done),
                np.asarray(action_mask, dtype=np.float32),
                np.asarray(next_action_mask, dtype=np.float32),
            )
        )

    def sample(self, batch_size: int) -> list[Transition]:
        """Sample a random mini-batch."""
        if batch_size > len(self.memory):
            raise ValueError("batch_size cannot exceed replay buffer length.")
        return self.rng.sample(list(self.memory), batch_size)

    def __len__(self) -> int:
        return len(self.memory)
