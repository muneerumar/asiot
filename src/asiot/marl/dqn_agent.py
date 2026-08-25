"""Independent DQN agent with shared parameters and action masking."""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from asiot.marl.policy_network import DQNPolicyNetwork
from asiot.marl.replay_buffer import ReplayBuffer


def resolve_device(device: str | None = None) -> torch.device:
    """Resolve the compute device for DQN training.

    Preference order: explicit request, then CUDA, then Apple MPS, then CPU.
    The original code only tried CUDA and fell back to CPU, so on Apple Silicon
    the GPU sat idle while six CPU cores ran at 100%.
    """
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DQNAgent:
    """DQN learner used by neural MARL SIoT policies."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.95,
        replay_buffer_size: int = 50_000,
        seed: int | None = None,
        device: str | None = None,
    ) -> None:
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.device = resolve_device(device)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.policy_net = DQNPolicyNetwork(observation_dim, action_dim).to(self.device)
        self.target_net = DQNPolicyNetwork(observation_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(replay_buffer_size, seed=seed)
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        epsilon: float = 0.0,
    ) -> int:
        """Select an epsilon-greedy action while preventing invalid actions."""
        mask = np.asarray(action_mask, dtype=np.float32)
        valid_actions = np.flatnonzero(mask > 0.0)
        if len(valid_actions) == 0:
            raise ValueError("action_mask must contain at least one valid action.")
        if random.random() < epsilon:
            return int(random.choice(valid_actions.tolist()))
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(obs_tensor).squeeze(0)
            mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            q_values = q_values.masked_fill(~mask_tensor, -1e9)
            return int(torch.argmax(q_values).item())

    def update(self, batch_size: int) -> float | None:
        """Run one DQN gradient step and return the loss."""
        if len(self.replay_buffer) < batch_size:
            return None
        transitions = self.replay_buffer.sample(batch_size)
        obs = torch.as_tensor(
            np.stack([transition.obs for transition in transitions]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            [transition.action for transition in transitions],
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(1)
        rewards = torch.as_tensor(
            [transition.reward for transition in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        next_obs = torch.as_tensor(
            np.stack([transition.next_obs for transition in transitions]),
            dtype=torch.float32,
            device=self.device,
        )
        done = torch.as_tensor(
            [transition.done for transition in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        next_masks = torch.as_tensor(
            np.stack([transition.next_action_mask for transition in transitions]),
            dtype=torch.bool,
            device=self.device,
        )

        q_values = self.policy_net(obs).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_obs).masked_fill(~next_masks, -1e9)
            max_next_q = next_q.max(dim=1).values
            target = rewards + self.gamma * max_next_q * (1.0 - done)
        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()
        return float(loss.item())

    def update_target_network(self) -> None:
        """Synchronize target network parameters."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str | Path) -> Path:
        """Save model and optimizer state."""
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "observation_dim": self.observation_dim,
                "action_dim": self.action_dim,
                "gamma": self.gamma,
                "policy_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )
        return checkpoint_path

    @classmethod
    def load(cls, path: str | Path, learning_rate: float = 1e-3, device: str | None = None) -> "DQNAgent":
        """Load a DQN checkpoint."""
        checkpoint = torch.load(Path(path), map_location=device or "cpu")
        agent = cls(
            int(checkpoint["observation_dim"]),
            int(checkpoint["action_dim"]),
            learning_rate=learning_rate,
            gamma=float(checkpoint.get("gamma", 0.95)),
            device=device,
        )
        agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        agent.target_net.load_state_dict(checkpoint["target_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return agent
