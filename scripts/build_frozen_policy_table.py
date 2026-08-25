"""Frozen-policy table (Task A deliverable).

Builds the paper's table from the POST-TRAINING frozen evaluations:

    policies    = untrained_random_init, marl_trained, heuristic_proposed
    conditions  = benign (f=0), selective (f=0.3), collusion (f=0.3)
    metrics     = reward, cooperation rate

Inputs (raw local run artifacts under `outputs/marl_attack_eval/`; compact
derived tables are tracked under `supplementary_results/marl/`):
  - 6 CSVs from scripts/evaluate_marl_under_attack.py, one per trained
    checkpoint (neural_marl_{social,no_social}_high_seed{3000,6000,9000}),
    each evaluated with its matching include-social/baseline config on the
    same 10 held-out seeds (1,400,000+), epsilon=0.
  - outputs/marl_frozen_benign.csv from scripts/analyze_marl_frozen_eval.py
    (benign rows recomputed from the run_dir interactions.csv/steps.csv).

The 3 trained checkpoints of a variant are averaged per seed, then paired
(by seed) against the untrained control and the heuristic with the Wilcoxon
signed-rank test. Raw reward is NOT comparable across variants (no_social
uses a different, higher-scale legacy heuristic objective, see scripts/
diagnose_marl_reward_scale.py); cooperation rate is the comparable metric.

Usage:
    python scripts/build_frozen_policy_table.py
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

import numpy as np
from scipy import stats

CONDITIONS = (("none", 0.0), ("selective", 0.3), ("collusion", 0.3))
SEED_STARTS = (3000, 6000, 9000)


def _ci(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def _load_attack_csvs(attack_root: Path, variant: str) -> dict[str, dict]:
    """Return {(policy, condition, seed): (reward, coop)} averaged over seeds."""
    data: dict[str, list] = defaultdict(list)
    for seed_start in SEED_STARTS:
        stem = f"{variant}_high_seed{seed_start}"
        with open(attack_root / f"{stem}.csv") as handle:
            for row in csv.DictReader(handle):
                key = (row["policy"], row["attack"], int(row["seed"]))
                data[key].append(
                    (float(row["reward"]), float(row["cooperation_rate"])))
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack-root", default="outputs/marl_attack_eval")
    ap.add_argument("--benign", default="outputs/marl_frozen_benign.csv")
    ap.add_argument("--out-dir", default="outputs/frozen_policy_table")
    args = ap.parse_args()

    attack_root = Path(args.attack_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = ("social", "no_social")
    eval_seeds: set[int] = set()
    for variant in variants:
        rows = _load_attack_csvs(attack_root, variant)
        for (_, _, seed) in rows:
            eval_seeds.add(seed)
    eval_seeds = sorted(eval_seeds)
    for variant in variants:
        rows = _load_attack_csvs(attack_root, variant)
        with open(out_dir / f"per_seed_{variant}.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["policy", "condition", "fraction", "seed",
                             "reward", "cooperation_rate"])
            for (pol, cond, seed), runs in sorted(rows.items()):
                rew = statistics.mean(r[0] for r in runs)
                coop = statistics.mean(r[1] for r in runs)
                frac = dict(CONDITIONS)[cond]
                writer.writerow([pol, cond, frac, seed, f"{rew:.6f}",
                                 f"{coop:.6f}"])

    # ---- summary table + paired tests -----------------------------------
    print(f"{'variant':10s} {'condition':10s} {'policy':22s} "
          f"{'reward':>8s} {'coop':>8s} {'coopCI':>8s} {'vsUntr p':>9s} "
          f"{'vsHeur p':>9s}")
    print("-" * 100)

    with open(out_dir / "summary.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        # Both tests are written out. The paper's prespecified inference is the
        # paired t test (as used for the benign, attack and ablation families),
        # so reporting only Wilcoxon here would apply an undeclared method to
        # one table. The two disagree on the social attack cells, and that
        # disagreement is itself reportable.
        writer.writerow(["variant", "condition", "fraction", "policy",
                         "reward_mean", "coop_mean", "coop_ci95",
                         "p_vs_untrained", "p_vs_heuristic",
                         "p_vs_untrained_ttest", "p_vs_heuristic_ttest",
                         "dz_vs_untrained", "dz_vs_heuristic"])
        for variant in variants:
            rows = _load_attack_csvs(attack_root, variant)
            for cond, frac in CONDITIONS:
                pols = ("untrained_random_init", "marl_trained",
                        "heuristic_proposed")
                stats_by_pol: dict[str, tuple[list, list]] = {}
                for pol in pols:
                    rew = [statistics.mean(r[0] for r in rows[(pol, cond, s)])
                           for s in eval_seeds]
                    coop = [statistics.mean(r[1] for r in rows[(pol, cond, s)])
                            for s in eval_seeds]
                    stats_by_pol[pol] = (rew, coop)
                untr_coop = stats_by_pol["untrained_random_init"][1]
                heur_coop = stats_by_pol["heuristic_proposed"][1]
                def _paired(a: list[float], b: list[float]) -> tuple[float, float]:
                    """Paired t p-value and Cohen's d_z, the declared method."""
                    diffs = [x - y for x, y in zip(a, b)]
                    sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
                    p = stats.ttest_rel(a, b).pvalue
                    return p, (statistics.mean(diffs) / sd if sd > 0 else float("nan"))

                for pol in pols:
                    rew, coop = stats_by_pol[pol]
                    is_untr = pol == "untrained_random_init"
                    is_heur = pol == "heuristic_proposed"
                    p_untr = stats.wilcoxon(coop, untr_coop).pvalue \
                        if not is_untr else float("nan")
                    p_heur = stats.wilcoxon(coop, heur_coop).pvalue \
                        if not is_heur else float("nan")
                    pt_untr, dz_untr = _paired(coop, untr_coop) if not is_untr \
                        else (float("nan"), float("nan"))
                    pt_heur, dz_heur = _paired(coop, heur_coop) if not is_heur \
                        else (float("nan"), float("nan"))
                    line = (f"{variant:10s} {cond:10s} {pol:22s} "
                            f"{statistics.mean(rew):8.1f} "
                            f"{statistics.mean(coop):8.4f} "
                            f"{_ci(coop):8.4f} "
                            f"{p_untr:9.3g} {p_heur:9.3g}")
                    print(line)
                    writer.writerow([variant, cond, frac, pol,
                                     f"{statistics.mean(rew):.6f}",
                                     f"{statistics.mean(coop):.6f}",
                                     f"{_ci(coop):.6f}",
                                     f"{p_untr}", f"{p_heur}",
                                     f"{pt_untr}", f"{pt_heur}",
                                     f"{dz_untr}", f"{dz_heur}"])

    print(f"\nwrote per-seed CSVs + summary.csv to {out_dir}")


if __name__ == "__main__":
    main()
