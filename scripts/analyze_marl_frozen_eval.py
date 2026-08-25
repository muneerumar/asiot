"""Frozen-policy table and reward-semantics analysis from the POST-TRAINING eval.

The paper's number for a trained policy is the post-training frozen evaluation:
epsilon=0, held-out seeds 1,400,000+, written per-run by run_marl_training.py as
interactions.csv/steps.csv under each cell's evaluation/ directory. It is NOT the
last training-log row (which mixes the epsilon=0.05 schedule).

This script reads those local raw CSVs and reports, for each variant:
  - reward semantics: is eval_reward a cumulative sum over the episode, or a
    per-decision mean? (It is a cumulative sum: the training loop sums each
    transition's reward over 500 steps. A policy that accepts more interactions
    accrues more reward regardless of quality.)
  - accept / reject ratio and per-decision total_utility (the reward basis)
  - benign frozen-policy numbers (cooperation rate, reward) with mean +/- CI

Cooperation is the run-level MEAN over all 500 steps. An earlier version read
only the final step, which at high load is 3-8 interactions; see the comment in
_read_cell. Any figure or manuscript number derived from the pre-fix
frozen_benign.csv is superseded.

Compact derived tables are tracked under `supplementary_results/marl/`.

Run this for benign rows (Task 1 + 2 of the frozen table). Attack rows are
produced by scripts/evaluate_marl_under_attack.py.

Usage:
    python scripts/analyze_marl_frozen_eval.py \
        --run-root outputs/marl_full_run2 --seeds 1_400_000 --runs 10
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

VARIANTS = (
    "neural_marl_social",
    "neural_marl_no_social",
)
SEED_STARTS = (3000, 6000, 9000)


def _ci(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, 0.0
    return mean, 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def _read_cell(run_root: Path, variant: str, seed_start: int, run_id: int) -> dict:
    cell = f"{variant}_seed{seed_start}"
    base = (
        run_root / cell / "high" / variant / "evaluation" / "high"
        / variant / f"run_{run_id}"
    )
    interactions = list(csv.DictReader(open(base / "interactions.csv")))
    steps = list(csv.DictReader(open(base / "steps.csv")))
    accepted = [r for r in interactions if r["attempted"] == "True"]
    rejected = [r for r in interactions if r["attempted"] == "False"]
    successful = [r for r in accepted if r["success"] == "True"]
    # Run-level cooperation is the MEAN over all steps, matching
    # run_stage0_sweep.one_run and diagnose_marl_learning.evaluate_agent.
    # This previously read steps[-1] -- a single final step, i.e. 3-8
    # interactions at high load -- which inflated the 95% CI about 19x and
    # made a p = 3e-17 difference between the variants look like p = 0.55.
    coop = statistics.mean(float(r["cooperation_rate"]) for r in steps)
    reward = sum(float(r["total_utility"]) for r in accepted) \
        + len(rejected) * -0.05
    return {
        "variant": variant,
        "seed_start": seed_start,
        "run_id": run_id,
        "n_interactions": len(interactions),
        "n_accept": len(accepted),
        "n_reject": len(rejected),
        "accept_ratio": len(accepted) / len(interactions),
        "n_success": len(successful),
        "success_ratio": len(successful) / len(accepted) if accepted else 0.0,
        "mean_total_utility": statistics.mean(
            float(r["total_utility"]) for r in accepted) if accepted else 0.0,
        "sum_reward": reward,
        "cooperation_rate": coop,
        "seed": interactions[0]["seed"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="outputs/marl_full_run2")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--output", default="outputs/marl_frozen_benign.csv")
    args = ap.parse_args()
    run_root = Path(args.run_root)

    rows: list[dict] = []
    for variant in VARIANTS:
        for seed_start in SEED_STARTS:
            for run_id in range(args.runs):
                rows.append(_read_cell(run_root, variant, seed_start, run_id))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("=== POST-TRAINING FROZEN EVAL (epsilon=0, held-out seeds) ===")
    print(f"  {len(rows)} runs total (3 seeds x {args.runs} runs x 2 variants)\n")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)

    print(f"{'variant':28s} {'accept%':>8s} {'reject%':>8s} {'succ/acc':>8s} "
          f"{'mean U':>7s} {'sumRw':>9s} {'coop':>7s} {'coop CI':>10s}")
    for variant, group in grouped.items():
        accepts = [r["accept_ratio"] for r in group]
        rejects = [r["n_reject"] / r["n_interactions"] for r in group]
        succ = [r["success_ratio"] for r in group]
        mean_u = [r["mean_total_utility"] for r in group]
        sum_reward = [r["sum_reward"] for r in group]
        coop = [r["cooperation_rate"] for r in group]
        cm, ch = _ci(coop)
        print(f"{variant:28s} {statistics.mean(accepts):8.4f} "
              f"{statistics.mean(rejects):8.4f} {statistics.mean(succ):8.4f} "
              f"{statistics.mean(mean_u):7.4f} {statistics.mean(sum_reward):9.1f} "
              f"{statistics.mean(coop):7.4f} +/-{ch:8.4f}")

    print("\nReward semantics: eval_reward = SUM over 500 steps of per-transition")
    print("  reward (accept: total_utility, reject: -0.05). It is a cumulative")
    print("  episode sum, NOT a per-decision mean. Accepting more interactions")
    print("  accrues more reward regardless of decision quality.")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
