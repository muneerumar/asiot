"""Training loop for Independent DQN MARL policies."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from asiot.config import load_config
from asiot.datatypes import SimulationConfig
from asiot.environment import ASIoTEnvironment
from asiot.logger import INTERACTION_COLUMNS, write_csv
from asiot.marl.dqn_agent import DQNAgent, resolve_device
from asiot.marl.observations import OBSERVATION_DIM
from asiot.metrics import STEP_METRIC_COLUMNS


class MARLTrainer:
    """Train and evaluate a parameter-shared Independent DQN policy."""

    def __init__(
        self,
        config: SimulationConfig | None = None,
        baseline_name: str = "neural_marl_social",
        include_social_features: bool = True,
        episodes: int = 500,
        steps_per_episode: int = 500,
        seed_start: int = 3000,
        output_dir: str | Path = "outputs/marl_training",
        checkpoint_dir: str | Path = "outputs/marl_checkpoints",
        top_k_candidates: int = 8,
        learning_rate: float = 1e-3,
        gamma: float = 0.95,
        batch_size: int = 64,
        replay_buffer_size: int = 50_000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_fraction: float = 0.65,
        target_update_interval: int = 250,
        eval_interval: int = 250,
        checkpoint_interval: int = 250,
        eval_seed_start: int = 900_000,
        eval_episodes: int = 3,
        attack_type: str = "none",
        attacker_fraction: float = 0.0,
        device: str | None = "cpu",
    ) -> None:
        self.config = config or load_config("config/default.yaml")
        self.baseline_name = baseline_name
        self.include_social_features = include_social_features
        self.episodes = int(episodes)
        self.steps_per_episode = int(steps_per_episode)
        self.seed_start = int(seed_start)
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.top_k_candidates = int(top_k_candidates)
        self.batch_size = int(batch_size)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        # Epsilon is scheduled in EPISODES, not transitions. The original code
        # decayed over 20,000 transitions; at ~6.7 transitions per step and 500
        # steps per episode that is ~6 episodes, so a 3,000-episode run spent
        # 99.8% of its time at the exploration floor, exploiting a network that
        # was still essentially random. Anchoring the schedule to the episode
        # budget makes exploration scale with training length and keeps the
        # schedule identical across variants and seeds.
        self.epsilon_decay_fraction = float(epsilon_decay_fraction)
        self.target_update_interval = int(target_update_interval)
        self.eval_interval = int(eval_interval)
        self.checkpoint_interval = int(checkpoint_interval)
        # Frozen-evaluation seeds are held out from training by construction:
        # training uses seed_start + episode, evaluation starts five orders of
        # magnitude higher. Reported results must never come from training-time
        # exploratory behaviour.
        self.eval_seed_start = int(eval_seed_start)
        self.eval_episodes = int(eval_episodes)
        self.attack_type = str(attack_type)
        self.attacker_fraction = float(attacker_fraction)
        self.device = resolve_device(device)
        self.observation_dim = OBSERVATION_DIM * self.top_k_candidates
        self.action_dim = self.top_k_candidates + 1
        self.agent = DQNAgent(
            self.observation_dim,
            self.action_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            replay_buffer_size=replay_buffer_size,
            seed=seed_start,
            device=self.device,
        )
        self.training_log: list[dict[str, Any]] = []

    def train(self) -> dict[str, Any]:
        """Run DQN training episodes and save a checkpoint."""
        global_step = 0
        for episode in range(self.episodes):
            seed = self.seed_start + episode
            env = ASIoTEnvironment(
                replace(self.config, random_seed=seed),
                seed=seed,
                baseline_name=self.baseline_name,
                run_id=episode,
            )
            episode_reward = 0.0
            losses: list[float] = []
            epsilon = self._epsilon(episode)
            for _ in range(self.steps_per_episode):
                transitions = env.step_with_neural_policy(
                    self.agent,
                    include_social_features=self.include_social_features,
                    top_k_candidates=self.top_k_candidates,
                    training=True,
                    epsilon=epsilon,
                )
                for transition in transitions:
                    self.agent.replay_buffer.push(
                        transition["obs"],
                        transition["action"],
                        transition["reward"],
                        transition["next_obs"],
                        transition["done"],
                        transition["action_mask"],
                        transition["next_action_mask"],
                    )
                    episode_reward += float(transition["reward"])
                    loss = self.agent.update(self.batch_size)
                    if loss is not None:
                        losses.append(loss)
                    global_step += 1
                    if global_step % self.target_update_interval == 0:
                        self.agent.update_target_network()

            record = {
                "episode": episode,
                "seed": seed,
                "reward": episode_reward,
                "epsilon": epsilon,
                "loss": sum(losses) / len(losses) if losses else 0.0,
                "eval_reward": "",
                "eval_cooperation_rate": "",
            }

            # Periodic frozen evaluation and checkpoint. The learning curve is
            # plotted from these points, never from `reward` above, which is
            # confounded by exploration and by how many tasks a seed generated.
            due = (episode + 1) % self.eval_interval == 0 or episode == self.episodes - 1
            if due:
                evaluation = self.frozen_evaluation()
                record.update(evaluation)
            if (episode + 1) % self.checkpoint_interval == 0:
                self.agent.save(self.checkpoint_dir /
                                f"{self.baseline_name}_{self.config.load_level}"
                                f"_seed{self.seed_start}_ep{episode + 1}.pt")

            self.training_log.append(record)
            self._write_training_log()  # crash-safe: the log survives a long run
            suffix = ""
            if due:
                suffix = (f" eval_reward={record['eval_reward']:.4f}"
                          f" eval_coop={record['eval_cooperation_rate']:.4f}")
            print(
                "episode="
                f"{episode} reward={episode_reward:.4f} "
                f"epsilon={epsilon:.4f} "
                f"loss={record['loss']:.6f}{suffix}",
                flush=True,
            )

        checkpoint_path = self.agent.save(self.checkpoint_path)
        self._write_training_log()
        return {
            "checkpoint": checkpoint_path,
            "training_log": self.output_dir / "training_log.csv",
            "episodes": self.episodes,
            "steps_per_episode": self.steps_per_episode,
        }

    def evaluate(
        self,
        runs: int,
        steps: int,
        seed_start: int,
        output_dir: str | Path,
        load_level: str | None = None,
        attack_type: str | None = None,
        attacker_fraction: float | None = None,
    ) -> dict[str, Path]:
        """Evaluate the trained agent without exploration and save raw CSVs."""
        output_root = Path(output_dir)
        completed = 0
        for run_id in range(runs):
            seed = seed_start + run_id
            config = replace(
                self.config,
                random_seed=seed,
                load_level=load_level or self.config.load_level,
            )
            # Attacker identity is drawn from the seed, so the learned policy
            # and the heuristic face the SAME attacker set at the same seed.
            env = ASIoTEnvironment(
                config,
                seed=seed,
                baseline_name=self.baseline_name,
                run_id=run_id,
                attack_type=self.attack_type if attack_type is None else attack_type,
                attacker_fraction=(self.attacker_fraction if attacker_fraction is None
                                   else float(attacker_fraction)),
            )
            for _ in range(steps):
                env.step_with_neural_policy(
                    self.agent,
                    include_social_features=self.include_social_features,
                    top_k_candidates=self.top_k_candidates,
                    training=False,
                    epsilon=0.0,
                )
            run_dir = output_root / (load_level or self.config.load_level) / self.baseline_name / f"run_{run_id}"
            write_csv(run_dir / "interactions.csv", env.logger.interactions, INTERACTION_COLUMNS)
            write_csv(run_dir / "steps.csv", env.logger.steps, STEP_METRIC_COLUMNS)
            completed += 1
        return {"output_dir": output_root, "runs_completed": Path(str(completed))}

    @property
    def checkpoint_path(self) -> Path:
        """Final checkpoint path, namespaced by seed so parallel seeds coexist."""
        return (self.checkpoint_dir /
                f"{self.baseline_name}_{self.config.load_level}_seed{self.seed_start}.pt")

    def _epsilon(self, episode: int) -> float:
        """Linear decay across the first ``epsilon_decay_fraction`` of episodes."""
        decay_episodes = max(1.0, self.epsilon_decay_fraction * self.episodes)
        fraction = min(episode / decay_episodes, 1.0)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def _write_training_log(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "training_log.csv"
        fieldnames = ("episode", "seed", "reward", "epsilon", "loss",
                      "eval_reward", "eval_cooperation_rate")
        write_csv(path, self.training_log, fieldnames)

    def frozen_evaluation(
        self,
        episodes: int | None = None,
        steps: int | None = None,
        seed_start: int | None = None,
        load_level: str | None = None,
        attack_type: str | None = None,
        attacker_fraction: float | None = None,
    ) -> dict[str, float]:
        """Evaluate the current policy greedily (epsilon = 0) on held-out seeds.

        Nothing here touches the replay buffer or the optimizer, so calling it
        mid-training cannot influence what is learned. This is the only source
        of reportable performance numbers: training reward is confounded by the
        exploration schedule and must never appear as a result.
        """
        episodes = self.eval_episodes if episodes is None else int(episodes)
        steps = self.steps_per_episode if steps is None else int(steps)
        seed_start = self.eval_seed_start if seed_start is None else int(seed_start)
        rewards: list[float] = []
        cooperation: list[float] = []
        for index in range(episodes):
            seed = seed_start + index
            env = ASIoTEnvironment(
                replace(self.config, random_seed=seed,
                        load_level=load_level or self.config.load_level),
                seed=seed,
                baseline_name=self.baseline_name,
                run_id=index,
                attack_type=self.attack_type if attack_type is None else attack_type,
                attacker_fraction=(self.attacker_fraction if attacker_fraction is None
                                   else float(attacker_fraction)),
            )
            total = 0.0
            for _ in range(steps):
                for transition in env.step_with_neural_policy(
                    self.agent,
                    include_social_features=self.include_social_features,
                    top_k_candidates=self.top_k_candidates,
                    training=False,
                    epsilon=0.0,
                ):
                    total += float(transition["reward"])
            rewards.append(total)
            rows = env.logger.steps
            if rows:
                cooperation.append(
                    sum(float(r["cooperation_rate"]) for r in rows) / len(rows)
                )
        return {
            "eval_reward": sum(rewards) / len(rewards) if rewards else float("nan"),
            "eval_cooperation_rate": (sum(cooperation) / len(cooperation)
                                      if cooperation else float("nan")),
        }
