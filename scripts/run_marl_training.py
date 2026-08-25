"""Train and smoke-evaluate neural MARL DQN policies."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asiot.config import load_config
from asiot.marl.trainer import MARLTrainer
from scripts.aggregate_results import aggregate_outputs

POLICY_TYPES = ("neural_marl_no_social", "neural_marl_social")
DEFAULT_LOAD_LEVELS = ("low", "medium", "high", "extreme")


def main() -> None:
    """Train a neural MARL policy and write evaluation CSV summaries."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=3000)
    parser.add_argument("--load-level", default="medium")
    parser.add_argument("--load-levels", default=None)
    parser.add_argument("--policy-type", choices=POLICY_TYPES, default="neural_marl_social")
    parser.add_argument("--output-dir", default="outputs/marl_training")
    parser.add_argument("--checkpoint-dir", default="outputs/marl_checkpoints")
    parser.add_argument("--eval-runs", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epsilon-decay-fraction", type=float, default=0.65,
                        help="Fraction of episodes over which epsilon decays to its floor.")
    parser.add_argument("--eval-interval", type=int, default=250,
                        help="Episodes between frozen (epsilon=0) evaluations.")
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--eval-episodes", type=int, default=3,
                        help="Held-out episodes per periodic frozen evaluation.")
    parser.add_argument("--eval-seed-start", type=int, default=900_000,
                        help="Evaluation seeds, disjoint from training seeds by construction.")
    parser.add_argument("--attack", default="none",
                        help="Attack type for the FINAL evaluation (not for training).")
    parser.add_argument("--attacker-fraction", type=float, default=0.0)
    parser.add_argument("--device", default="cpu",
                        help="torch device for DQN. Measured on this workload: "
                             "MPS is ~3x SLOWER than CPU (env stepping dominates, "
                             "the DQN is tiny), so the default is cpu; pass "
                             "'mps' only if you have a good reason.")
    args = parser.parse_args()

    load_levels = (
        _parse_load_levels(args.load_levels)
        if args.load_levels
        else [args.load_level]
    )
    include_social = args.policy_type == "neural_marl_social"
    base_config = load_config("config/default.yaml")
    all_checkpoints = []

    for load_level in load_levels:
        config = replace(
            base_config,
            load_level=load_level,
            steps=args.steps,
            random_seed=args.seed_start,
        )
        trainer = MARLTrainer(
            config=config,
            baseline_name=args.policy_type,
            include_social_features=include_social,
            episodes=args.episodes,
            steps_per_episode=args.steps,
            seed_start=args.seed_start,
            output_dir=Path(args.output_dir) / load_level / args.policy_type / "training",
            checkpoint_dir=args.checkpoint_dir,
            top_k_candidates=args.top_k,
            batch_size=args.batch_size,
            epsilon_decay_fraction=args.epsilon_decay_fraction,
            eval_interval=args.eval_interval,
            checkpoint_interval=args.checkpoint_interval,
            eval_episodes=args.eval_episodes,
            eval_seed_start=args.eval_seed_start,
            device=args.device,
        )
        train_info = trainer.train()
        eval_dir = Path(args.output_dir) / load_level / args.policy_type / "evaluation"
        # Final frozen evaluation. Seeds are disjoint from training
        # (seed_start + episode) and from the periodic in-training evaluations
        # (eval_seed_start), so no reported number comes from a seed the policy
        # was trained or model-selected on.
        trainer.evaluate(
            runs=args.eval_runs,
            steps=args.steps,
            seed_start=args.eval_seed_start + 500_000,
            output_dir=eval_dir,
            load_level=load_level,
            attack_type=args.attack,
            attacker_fraction=args.attacker_fraction,
        )
        all_checkpoints.append(train_info["checkpoint"])
        print(f"checkpoint={train_info['checkpoint']}")
        print(f"training_log={train_info['training_log']}")

    aggregated_dir = Path(args.output_dir) / "aggregated"
    paths = aggregate_outputs(args.output_dir, aggregated_dir)
    for name, path in paths.items():
        print(f"{name}={path}")
    print("checkpoints=" + ",".join(str(path) for path in all_checkpoints))


def _parse_load_levels(raw: str) -> list[str]:
    load_levels = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in load_levels if item not in DEFAULT_LOAD_LEVELS]
    if unknown:
        raise ValueError(f"Unknown load levels: {', '.join(unknown)}")
    return load_levels


if __name__ == "__main__":
    main()
