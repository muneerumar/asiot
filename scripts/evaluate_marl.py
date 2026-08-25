"""Evaluate a trained neural MARL checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asiot.config import load_config
from asiot.environment import ASIoTEnvironment
from asiot.logger import INTERACTION_COLUMNS, write_csv
from asiot.marl.dqn_agent import DQNAgent
from asiot.metrics import STEP_METRIC_COLUMNS
from scripts.aggregate_results import aggregate_outputs

POLICY_TYPES = ("neural_marl_social", "neural_marl_no_social")
DEFAULT_LOAD_LEVELS = ("low", "medium", "high", "extreme")


def main() -> None:
    """Run checkpoint evaluation with epsilon-free neural action selection."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--policy-type", choices=POLICY_TYPES, required=True)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=5000)
    parser.add_argument("--load-levels", default=",".join(DEFAULT_LOAD_LEVELS))
    parser.add_argument("--output-dir", default="outputs/marl_eval")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    agent = DQNAgent.load(args.checkpoint)
    include_social = args.policy_type == "neural_marl_social"
    base_config = load_config("config/default.yaml")
    output_root = Path(args.output_dir)
    completed = 0
    for load_level in _parse_load_levels(args.load_levels):
        for run_id in range(args.runs):
            seed = args.seed_start + run_id
            run_dir = output_root / load_level / args.policy_type / f"run_{run_id}"
            config = replace(
                base_config,
                load_level=load_level,
                steps=args.steps,
                random_seed=seed,
                output_dir=run_dir,
            )
            env = ASIoTEnvironment(
                config,
                seed=seed,
                baseline_name=args.policy_type,
                run_id=run_id,
            )
            for _ in range(args.steps):
                env.step_with_neural_policy(
                    agent,
                    include_social_features=include_social,
                    top_k_candidates=args.top_k,
                    training=False,
                    epsilon=0.0,
                )
            write_csv(run_dir / "interactions.csv", env.logger.interactions, INTERACTION_COLUMNS)
            write_csv(run_dir / "steps.csv", env.logger.steps, STEP_METRIC_COLUMNS)
            completed += 1

    paths = aggregate_outputs(output_root, output_root / "aggregated")
    print(f"runs_completed={completed}")
    for name, path in paths.items():
        print(f"{name}={path}")


def _parse_load_levels(raw: str) -> list[str]:
    load_levels = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in load_levels if item not in DEFAULT_LOAD_LEVELS]
    if unknown:
        raise ValueError(f"Unknown load levels: {', '.join(unknown)}")
    return load_levels


if __name__ == "__main__":
    main()
