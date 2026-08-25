"""Why does the no_social MARL variant score higher reward than social?

Both neural variants are trained inside ``ASIoTEnvironment.step_with_neural_policy``,
where an accepted action is rewarded with ``score["total_utility"]`` -- but that
value is produced by the variant's own ``score_neighbors``. The two variants wire
DIFFERENT score functions:

  neural_marl_social    -> NeuralMARLSocialModel  (ProposedASIoTFramework)
                           total_utility = Eqs 39-46 weighted sum.
  neural_marl_no_social -> NeuralMARLNoSocialModel (StandardMARLNoSocialModel)
                           total_utility = expected_success - 0.2*delay_norm
                                           + 0.2*resource_utility  (legacy heuristic).

So the two variants optimize different reward functions on different scales, which
can explain (a) the higher frozen-eval reward of no_social and (b) its ~50x lower
TD loss (its target is a smoother function of fewer moving parts). This script
quantifies the difference at identical decision points: it walks the same tasks with
a fixed deterministic agent for both variants and compares per-transition rewards,
accept ratios, and the two total_utility formulas side by side.

Usage:
    python scripts/diagnose_marl_reward_scale.py --steps 200 --seed 777
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asiot.baselines import (  # noqa: E402
    _compute_weighted_total,
    ProposedASIoTFramework,
    StandardMARLNoSocialModel,
)
from asiot.config import load_config  # noqa: E402
from asiot.environment import ASIoTEnvironment  # noqa: E402
from asiot.social_cognition import clip01  # noqa: E402


class _FixedAgent:
    """Deterministic stub: always take the first valid (masked) action."""

    def __init__(self) -> None:
        self.accepted = 0
        self.rejected = 0

    def select_action(self, obs, action_mask, epsilon: float = 0.0) -> int:
        valid = [i for i, m in enumerate(action_mask) if m > 0.0]
        action = int(valid[0])
        return action


def _formula_no_social(score: dict) -> float:
    return clip01(
        float(score["expected_success"])
        - 0.2 * float(score["delay_norm"])
        + 0.2 * float(score["resource_utility"])
    )


def _collect(steps: int, load_level: str, seed: int) -> dict:
    """Run both variants over the SAME tasks and record their rewards."""
    config = replace(load_config("config/default.yaml"), load_level=load_level)
    rows: list[dict] = []

    # Pure formula-scale isolation: on the SAME score dict produced by the
    # proposed score function, compute both total_utility formulas. This shows
    # how much of the reward gap is just the reward definition, independent of
    # the scoring differences between the two variants.
    proposed = ProposedASIoTFramework()
    env0 = ASIoTEnvironment(replace(config, random_seed=seed), seed=seed,
                            baseline_name="proposed", run_id=0)
    f_social: list[float] = []
    f_no_social: list[float] = []
    for _ in range(steps):
        for task in env0.generate_tasks(env0.time_step):
            requester = env0.nodes[task.requester_id]
            neighbors = requester.observe_neighbors(env0.graph)
            if not neighbors:
                continue
            scores = proposed.score_neighbors(requester, neighbors, task,
                                              env0.graph, config, env0.nodes)
            for score in scores.values():
                f_social.append(_compute_weighted_total(score, config))
                f_no_social.append(_formula_no_social(score))
        env0.time_step += 1
    scale = statistics.mean(f_no_social) / statistics.mean(f_social)

    for variant, social in (("neural_marl_social", True),
                            ("neural_marl_no_social", False)):
        rewards: list[float] = []
        accept_rewards: list[float] = []
        formula_values: list[float] = []
        env = ASIoTEnvironment(replace(config, random_seed=seed), seed=seed,
                               baseline_name=variant, run_id=0)
        for _ in range(steps):
            for task in env.generate_tasks(env.time_step):
                requester = env.nodes[task.requester_id]
                neighbors = requester.observe_neighbors(env.graph)
                if not neighbors:
                    continue
                scores = env.policy.score_neighbors(requester, neighbors, task,
                                                    env.graph, config, env.nodes)
                for score in scores.values():
                    formula_values.append(
                        _compute_weighted_total(score, config)
                        if social else _formula_no_social(score)
                    )
            env.time_step += 1

        env = ASIoTEnvironment(replace(config, random_seed=seed), seed=seed,
                               baseline_name=variant, run_id=0)
        agent = _FixedAgent()
        for _ in range(steps):
            transitions = env.step_with_neural_policy(
                agent, include_social_features=social, top_k_candidates=8,
                training=False, epsilon=0.0,
            )
            for t in transitions:
                rewards.append(float(t["reward"]))
                if float(t["reward"]) > 0.0:
                    accept_rewards.append(float(t["reward"]))
            env.time_step += 1

        n = len(rewards)
        n_accept = len(accept_rewards)
        rows.append({
            "variant": variant,
            "decision_scores": len(formula_values),
            "formula_mean": statistics.mean(formula_values),
            "transitions": n,
            "accept_ratio": n_accept / n if n else 0.0,
            "mean_transition_reward": statistics.mean(rewards) if n else 0.0,
            "mean_accept_reward": statistics.mean(accept_rewards) if n_accept else 0.0,
            "accept_reward_std": statistics.stdev(accept_rewards) if n_accept > 1 else 0.0,
            "formula_scale_ratio_same_scores": scale,
            "sample_tasks": "same seed both variants",
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--load-level", default="high")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--output", default="outputs/marl_diagnostics/reward_scale.csv")
    args = ap.parse_args()

    rows = _collect(args.steps, args.load_level, args.seed)
    for row in rows:
        print(f"--- {row['variant']} ---")
        print(f"  total_utility formula mean (identical tasks) = {row['formula_mean']:.4f}")
        print(f"  transitions = {row['transitions']}  accept_ratio = {row['accept_ratio']:.3f}")
        print(f"  mean reward/transition = {row['mean_transition_reward']:.4f}"
              f"  mean reward/accept = {row['mean_accept_reward']:.4f}"
              f"  std = {row['accept_reward_std']:.4f}")

    if len(rows) == 2:
        ratio = rows[1]["mean_transition_reward"] / rows[0]["mean_transition_reward"]
        print(f"\nno_social/social mean reward ratio = {ratio:.3f}")
        print(f"pure formula scale ratio (same score dict) = "
              f"{rows[0]['formula_scale_ratio_same_scores']:.3f}")
        print(f"accept-reward std ratio (social/no_social) = "
              f"{rows[0]['accept_reward_std'] / rows[1]['accept_reward_std']:.1f}x")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
