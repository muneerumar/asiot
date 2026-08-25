"""Always-accept finding: accept/reject rates per variant per condition (Task A).

Quantifies the degenerate-policy finding: the no_social learner accepts ~100%
of interactions (zero rejects) across all conditions, whereas the social
learner rejects ~4%. Combined with the higher-scale legacy heuristic objective this
explains (a) the inflated cumulative reward of no_social, (b) its lower
cooperation, and (c) its collapse under selective/collusion attack.

Input: the six per-checkpoint attack CSVs under a run root. When those CSVs
carry n_accept/n_reject columns (produced by the extended
scripts/evaluate_marl_under_attack.py) the accept rate is taken from them
directly for every policy x condition x seed. Otherwise accept rates are only
available for the benign rows via the post-training interaction CSVs.

Output: outputs/frozen_policy_table/always_accept_<suffix>.csv plus a table.

Usage:
    python scripts/analyze_always_accept.py \
        --attack-root outputs/marl_attack_eval_n30 --out-suffix n30
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

CONDITIONS = ("none", "selective", "collusion")
SEED_STARTS = (3000, 6000, 9000)
POLICIES = ("untrained_random_init", "marl_trained", "heuristic_proposed")


def _ci(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def _load_attack(root: Path, variant: str) -> tuple[dict, bool]:
    """{(policy, condition): [ (accept_ratio, coop) ]}; returns has_accept."""
    data: dict = defaultdict(list)
    has_accept = False
    for seed_start in SEED_STARTS:
        f = root / f"{variant}_high_seed{seed_start}.csv"
        if not f.exists():
            continue
        with open(f) as handle:
            for row in csv.DictReader(handle):
                if "n_accept" in row and "n_reject" in row:
                    has_accept = True
                    na = int(row["n_accept"])
                    nr = int(row["n_reject"])
                    accept = na / (na + nr) if na + nr > 0 else float("nan")
                else:
                    accept = float("nan")
                data[(row["policy"], row["attack"])].append(
                    (accept, float(row["cooperation_rate"])))
    return data, has_accept


def _benign_accept_counts() -> dict:
    """Fallback: exact accept/reject counts from the 60 benign eval CSVs."""
    run_root = Path("outputs/marl_full_run2")
    out = {}
    for variant in ("neural_marl_social", "neural_marl_no_social"):
        accept = reject = n = 0
        coop_vals = []
        for seed_start in SEED_STARTS:
            cell = f"{variant}_seed{seed_start}"
            base = (run_root / cell / "high" / variant / "evaluation"
                    / "high" / variant)
            for run_id in range(10):
                interactions = base / f"run_{run_id}" / "interactions.csv"
                if not interactions.exists():
                    continue
                n += 1
                with open(interactions) as handle:
                    for row in csv.DictReader(handle):
                        if str(row["attempted"]) == "True":
                            accept += 1
                        else:
                            reject += 1
                steps = base / f"run_{run_id}" / "steps.csv"
                with open(steps) as handle:
                    coop_vals.append(float(
                        [r for r in csv.DictReader(handle)][-1]
                        ["cooperation_rate"]))
        out[variant.replace("neural_marl_", "")] = {
            "accept": accept, "reject": reject,
            "n_interactions": accept + reject, "n_runs": n,
            "coop_mean": statistics.mean(coop_vals),
            "coop_ci": _ci(coop_vals),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack-root", default="outputs/marl_attack_eval_n30")
    ap.add_argument("--out-suffix", default="n30")
    args = ap.parse_args()

    root = Path(args.attack_root)
    benign = _benign_accept_counts()
    rows: list[list[str]] = []

    for variant in ("social", "no_social"):
        data, has_accept = _load_attack(root, variant)
        for cond in CONDITIONS:
            for pol in POLICIES:
                vals = data.get((pol, cond), [])
                if not vals:
                    continue
                accepts = [a for a, _ in vals if a == a]
                coops = [c for _, c in vals]
                if accepts:
                    accept_mean = statistics.mean(accepts)
                    accept_str = f"{accept_mean:.4f}"
                elif cond == "none" and pol == "marl_trained":
                    b = benign[variant]
                    accept_mean = b["accept"] / b["n_interactions"]
                    accept_str = f"{accept_mean:.4f} (from eval CSVs)"
                else:
                    accept_str = "n/a"
                rows.append([variant, cond, pol, f"{len(coops)}",
                             accept_str,
                             f"{statistics.mean(coops):.4f}",
                             f"{_ci(coops):.4f}"])

    out = Path("outputs/frozen_policy_table")
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / f"always_accept_{args.out_suffix}.csv"
    with open(out_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "condition", "policy", "n_seeds",
                         "accept_rate", "coop_mean", "coop_ci95"])
        writer.writerows(rows)

    print(f"{'variant':10s} {'cond':10s} {'policy':22s} {'n':>3s} "
          f"{'accept':>14s} {'coop':>8s} {'ci':>8s}")
    print("-" * 84)
    for r in rows:
        print(f"{r[0]:10s} {r[1]:10s} {r[2]:22s} {r[3]:>3s} "
              f"{r[4]:>14s} {r[5]:>8s} {r[6]:>8s}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
