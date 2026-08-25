"""Stress-test the collusion result (Task A follow-up).

The claim under test: "collusion f=0.3: trained social 0.6823 vs untrained
0.6679, p=0.0039, the one case training adds value."

This script applies the Holm-Bonferroni correction across ALL SIX
trained-vs-untrained comparisons (2 variants x 3 conditions) and reports
whether the collusion cell survives. It also reports Cohen's d_z (paired) as
the effect size, computed on the per-seed cooperation rates.

Input: per-seed CSVs produced by scripts/build_frozen_policy_table.py
       (one row per policy x condition x seed, cooperation_rate averaged over
       the three training seeds' checkpoints).

Usage:
    python scripts/stress_test_collusion.py \
        --social outputs/frozen_policy_table/per_seed_social.csv \
        --no-social outputs/frozen_policy_table/per_seed_no_social.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scipy import stats  # noqa: E402

CONDITIONS = ("none", "selective", "collusion")
FIXED_ALPHA = 0.05


def _load(path: Path) -> dict:
    """{(policy, condition, seed): cooperation_rate}."""
    data = {}
    with open(path) as handle:
        for row in csv.DictReader(handle):
            data[(row["policy"], row["condition"], int(row["seed"]))] = \
                float(row["cooperation_rate"])
    return data


def _cohen_dz(x: list[float], y: list[float]) -> float:
    diffs = [a - b for a, b in zip(x, y)]
    sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    return (statistics.mean(diffs) / sd) if sd > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--social", required=True, type=Path)
    ap.add_argument("--no-social", required=True, type=Path)
    args = ap.parse_args()

    # Six comparisons: {social, no_social} x {none, selective, collusion}
    comparisons = []
    for variant, path in (("social", args.social), ("no_social", args.no_social)):
        data = _load(path)
        seeds = sorted({s for (_, _, s) in data})
        for cond in CONDITIONS:
            trained = [data[("marl_trained", cond, s)] for s in seeds]
            untrained = [data[("untrained_random_init", cond, s)] for s in seeds]
            if len(trained) < 2:
                sys.exit(f"need >=2 seeds, got {len(trained)}")
            w, p = stats.wilcoxon(trained, untrained)
            dz = _cohen_dz(trained, untrained)
            comparisons.append({
                "variant": variant,
                "condition": cond,
                "n": len(seeds),
                "trained_mean": statistics.mean(trained),
                "untrained_mean": statistics.mean(untrained),
                "diff": statistics.mean(trained) - statistics.mean(untrained),
                "wilcoxon_w": float(w),
                "p": float(p),
                "cohen_dz": dz,
            })

    # Holm-Bonferroni across the six comparisons.
    ordered = sorted(comparisons, key=lambda c: c["p"])
    m = len(ordered)
    for i, c in enumerate(ordered):
        c["holm_alpha"] = FIXED_ALPHA / (m - i)
        c["survives"] = c["p"] <= c["holm_alpha"]

    print(f"{'variant':10s} {'cond':10s} {'n':>3s} {'trained':>8s} "
          f"{'untrained':>9s} {'diff':>7s} {'p':>9s} {'d_z':>7s} "
          f"{'holm_alp':>9s} {'survives':>8s}")
    print("-" * 96)
    for c in ordered:
        print(f"{c['variant']:10s} {c['condition']:10s} {c['n']:3d} "
              f"{c['trained_mean']:8.4f} {c['untrained_mean']:9.4f} "
              f"{c['diff']:+7.4f} {c['p']:9.4f} {c['cohen_dz']:+7.3f} "
              f"{c['holm_alpha']:9.4f} {str(c['survives']):>8s}")

    survived = [c for c in ordered if c["survives"]]
    print(f"\nHolm-Bonferroni (alpha={FIXED_ALPHA}, m={m}): "
          f"{len(survived)} of {m} comparisons survive.")
    collusion = [c for c in comparisons
                 if c["variant"] == "social" and c["condition"] == "collusion"][0]
    verdict = ("SURVIVES" if collusion["survives"] else "DOES NOT SURVIVE")
    print(f"social/collusion trained-vs-untrained: p={collusion['p']:.4f}, "
          f"d_z={collusion['cohen_dz']:+.3f} -> {verdict} Holm-Bonferroni.")
    print("Report plainly; do NOT call this 'the one case training adds value' "
          "until it has passed both checks at n=30.")


if __name__ == "__main__":
    main()