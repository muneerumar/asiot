"""Frozen MARL vs heuristic, benign and under attack, paired on held-out seeds.

Writes per-seed rows so paired statistics can be computed later without
recomputing the evaluation. Every policy sees the same seeds, and because
ASIoTEnvironment derives attacker identity from the seed, the learned policy and
the heuristic face the identical attacker set in every cell.

The UNTRAINED (random-initialised) network is evaluated alongside them as the
control that decides whether training contributed anything at all.

Usage:
    python scripts/evaluate_marl_under_attack.py --checkpoint <path> --seeds 10
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from asiot.config import load_config  # noqa: E402
from asiot.marl.dqn_agent import DQNAgent, resolve_device  # noqa: E402
from asiot.marl.observations import OBSERVATION_DIM  # noqa: E402
from diagnose_marl_learning import TOP_K, evaluate_agent, evaluate_heuristic  # noqa: E402

CONDITIONS = (("none", 0.0), ("selective", 0.3), ("collusion", 0.3))


def _ci(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, 0.0
    return mean, 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=1_400_000)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--load-level", default="high")
    ap.add_argument("--untrained-seed", type=int, default=12345)
    ap.add_argument("--device", default="cpu",
                    help="torch device for DQN. MPS is ~3x slower than CPU on "
                         "this workload; default is cpu.")
    ap.add_argument("--include-social", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Feed social-cognitive features to the network. Must "
                         "match how the checkpoint was trained: True for "
                         "neural_marl_social, False for neural_marl_no_social.")
    ap.add_argument("--baseline-name", default="neural_marl_social",
                    help="Score function used for observations AND reward. "
                         "Must match the checkpoint's variant.")
    ap.add_argument("--output", default="outputs/marl_diagnostics/marl_attack_eval.csv")
    args = ap.parse_args()

    config = load_config("config/default.yaml")
    device = resolve_device(args.device)
    print(f"device={device}", flush=True)
    seeds = [args.seed_start + i for i in range(args.seeds)]
    rows: list[dict[str, object]] = []

    for attack, fraction in CONDITIONS:
        print(f"\n=== attack={attack} f={fraction} | epsilon=0, {len(seeds)} held-out "
              f"seeds, {args.steps} steps, load={args.load_level} ===", flush=True)
        policies = {
            "untrained_random_init": DQNAgent(OBSERVATION_DIM * TOP_K, TOP_K + 1,
                                              seed=args.untrained_seed,
                                              device=device),
            "marl_trained": DQNAgent.load(args.checkpoint, device=device),
        }
        for label, agent in policies.items():
            rewards, cooperation, n_accept, n_reject = evaluate_agent(
                agent, seeds, args.steps, args.load_level, config,
                args.include_social,
                attack_type=attack, attacker_fraction=fraction,
                baseline_name=args.baseline_name,
            )
            _report(rows, label, attack, fraction, seeds, rewards,
                    cooperation, n_accept, n_reject)

        rewards, cooperation, n_accept, n_reject = evaluate_heuristic(
            seeds, args.steps, args.load_level, config,
            attack_type=attack, attacker_fraction=fraction,
        )
        _report(rows, "heuristic_proposed", attack, fraction, seeds, rewards,
                cooperation, n_accept, n_reject)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.output}")


def _report(rows, label, attack, fraction, seeds, rewards, cooperation,
            n_accept=None, n_reject=None) -> None:
    if n_accept is None:
        n_accept = [0] * len(seeds)
    if n_reject is None:
        n_reject = [0] * len(seeds)
    for seed, reward, coop, na, nr in zip(seeds, rewards, cooperation,
                                          n_accept, n_reject):
        rows.append({"policy": label, "attack": attack, "fraction": fraction,
                     "seed": seed, "reward": f"{reward:.6f}",
                     "cooperation_rate": f"{coop:.6f}",
                     "n_accept": na, "n_reject": nr})
    rm, rh = _ci(rewards)
    cm, ch = _ci(cooperation)
    ar = sum(na / (na + nr) for na, nr in zip(n_accept, n_reject)
             if na + nr > 0) / max(len(n_accept), 1)
    print(f"  {label:24s} reward={rm:9.2f}+/-{rh:6.2f}  "
          f"cooperation={cm:.4f}+/-{ch:.4f}  accept={ar:.4f}", flush=True)


if __name__ == "__main__":
    main()
