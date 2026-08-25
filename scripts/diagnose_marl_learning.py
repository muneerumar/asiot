"""Is the DQN learning at all? Compare checkpoints against an untrained network.

A rising training reward proves nothing when epsilon is decaying at the same
time, and a converged TD loss only says the network fits its own moving target.
The decisive question is whether a frozen trained policy beats a frozen
RANDOM-INITIALISED one on held-out seeds. If it does not, more episodes will
not help, and that is the finding to report.

Every policy here is evaluated with epsilon = 0 on the same held-out seeds, so
the comparison is paired: identical worlds, identical attacker draws, identical
task streams. The heuristic is included as the reference the paper compares to.

Usage:
    python scripts/diagnose_marl_learning.py --checkpoint-dir <dir> --seeds 10
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asiot.config import load_config  # noqa: E402
from asiot.environment import ASIoTEnvironment  # noqa: E402
from asiot.marl.dqn_agent import DQNAgent, resolve_device  # noqa: E402
from asiot.marl.observations import OBSERVATION_DIM  # noqa: E402

TOP_K = 8


def evaluate_agent(agent, seeds, steps, load_level, config, include_social,
                   attack_type="none", attacker_fraction=0.0,
                   baseline_name="neural_marl_social"):
    """Frozen (epsilon = 0) evaluation; returns per-seed reward, cooperation
    and accept/reject counts.

    ``baseline_name`` sets the score function that builds the observation AND
    defines the reward (score["total_utility"]). It must match the variant the
    network was trained as: neural_marl_social uses Eqs 39-46 utility, the
    no_social legacy heuristic uses its own formula.
    """
    rewards, cooperation, n_accept, n_reject = [], [], [], []
    for seed in seeds:
        env = ASIoTEnvironment(
            replace(config, random_seed=seed, load_level=load_level),
            seed=seed, baseline_name=baseline_name, run_id=0,
            attack_type=attack_type, attacker_fraction=attacker_fraction,
        )
        total = 0.0
        for _ in range(steps):
            for transition in env.step_with_neural_policy(
                agent, include_social_features=include_social,
                top_k_candidates=TOP_K, training=False, epsilon=0.0,
            ):
                total += float(transition["reward"])
        rewards.append(total)
        rows = env.logger.steps
        cooperation.append(sum(float(r["cooperation_rate"]) for r in rows) / len(rows))
        n_accept.append(sum(1 for r in env.logger.interactions
                            if str(r["attempted"]) == "True"))
        n_reject.append(sum(1 for r in env.logger.interactions
                            if str(r["attempted"]) != "True"))
    return rewards, cooperation, n_accept, n_reject


def evaluate_heuristic(seeds, steps, load_level, config,
                       attack_type="none", attacker_fraction=0.0):
    """The heuristic on the same seeds. Its reward is the same utility sum."""
    rewards, cooperation, n_accept, n_reject = [], [], [], []
    for seed in seeds:
        env = ASIoTEnvironment(
            replace(config, random_seed=seed, load_level=load_level),
            seed=seed, baseline_name="proposed", run_id=0,
            attack_type=attack_type, attacker_fraction=attacker_fraction,
        )
        env.run(steps)
        rows = env.logger.steps
        cooperation.append(sum(float(r["cooperation_rate"]) for r in rows) / len(rows))
        rewards.append(sum(float(r.get("total_utility", 0.0) or 0.0)
                           for r in env.logger.interactions if r.get("attempted")))
        n_accept.append(sum(1 for r in env.logger.interactions
                            if str(r["attempted"]) == "True"))
        n_reject.append(sum(1 for r in env.logger.interactions
                            if str(r["attempted"]) != "True"))
    return rewards, cooperation, n_accept, n_reject


def summarize(label, rewards, cooperation):
    def ci(vals):
        m = statistics.mean(vals)
        if len(vals) < 2:
            return m, 0.0
        return m, 1.96 * statistics.stdev(vals) / math.sqrt(len(vals))
    rm, rh = ci(rewards)
    cm, ch = ci(cooperation)
    print(f"{label:34s} reward={rm:9.2f} +/-{rh:6.2f}   cooperation={cm:.4f} +/-{ch:.4f}")
    return rm, cm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--pattern", default="neural_marl_social_high_seed3000_ep*.pt")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=1_400_000)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--load-level", default="high")
    ap.add_argument("--include-social", action="store_true", default=True)
    ap.add_argument("--device", default="cpu",
                    help="torch device for DQN. MPS is ~3x slower than CPU on "
                         "this workload; default is cpu.")
    args = ap.parse_args()

    config = load_config("config/default.yaml")
    device = resolve_device(args.device)
    print(f"device={device}\n")
    seeds = [args.seed_start + i for i in range(args.seeds)]
    print(f"frozen evaluation, epsilon=0, {len(seeds)} held-out seeds, "
          f"{args.steps} steps, load={args.load_level}\n")

    # Control: an untrained network with the same architecture and seeding.
    untrained = DQNAgent(OBSERVATION_DIM * TOP_K, TOP_K + 1, seed=12345,
                         device=device)
    r, c, _, _ = evaluate_agent(untrained, seeds, args.steps, args.load_level,
                                config, args.include_social)
    summarize("UNTRAINED (random init)", r, c)

    for path in sorted(Path(args.checkpoint_dir).glob(args.pattern),
                       key=lambda p: int(p.stem.rsplit("ep", 1)[-1])):
        agent = DQNAgent.load(path, device=device)
        r, c, _, _ = evaluate_agent(agent, seeds, args.steps, args.load_level,
                                    config, args.include_social)
        summarize(f"trained {path.stem.rsplit('_', 1)[-1]}", r, c)

    r, c, _, _ = evaluate_heuristic(seeds, args.steps, args.load_level, config)
    summarize("HEURISTIC (proposed)", r, c)


if __name__ == "__main__":
    main()
