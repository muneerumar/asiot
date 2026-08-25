"""Paired analysis of the adaptive-mu ablation (Task 1).

Reads outputs/<dir>/adaptive_mu/run_summary.csv, which contains one row per
(attack, fraction, adaptive_mu, run_id). Because the sweep derives its seed
from run_id alone, the adaptive and fixed arms of a given (attack, fraction,
run_id) triple face an identical world -- identical node placement, identical
attacker set, identical task stream. The comparison is therefore paired, and a
paired test is the correct one; treating the arms as independent would throw
away that structure and inflate the variance.

Reports, per cell and per metric: paired mean difference (fixed - adaptive,
so positive = adaptive reduces the quantity), 95% CI on the difference,
paired t, Cohen's d_z, and a Holm-corrected p-value across cells for the
primary metric. No cell is selected after seeing the results.

Usage:
    python scripts/analyze_adaptive_mu.py --input outputs/stage2b/adaptive_mu/run_summary.csv
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from scipy import stats

PRIMARY = "decision_distortion"
METRICS = [
    "decision_distortion",
    "decision_distortion_evidenced",
    "decision_distortion_counterfactual",
    "cooperation_rate",
    "task_completion_ratio",
    "reputation_separation",
]


def paired_stats(fixed: pd.Series, adaptive: pd.Series) -> dict[str, float]:
    """Paired difference statistics for fixed-mu minus adaptive-mu."""
    diff = (fixed - adaptive).dropna()
    n = len(diff)
    if n < 2:
        return {"n": n, "mean_diff": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "t": float("nan"), "p": float("nan"),
                "d_z": float("nan")}
    mean = diff.mean()
    sd = diff.std(ddof=1)
    se = sd / math.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    t_stat, p_val = stats.ttest_rel(fixed, adaptive, nan_policy="omit")
    return {"n": n, "mean_diff": mean, "ci_low": mean - tcrit * se,
            "ci_high": mean + tcrit * se, "t": float(t_stat), "p": float(p_val),
            "d_z": mean / sd if sd > 0 else float("nan")}


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adjusted = [0.0] * len(pvals)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (len(pvals) - rank) * pvals[idx]
        running = max(running, min(val, 1.0))
        adjusted[idx] = running
    return adjusted


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/stage2b/adaptive_mu/run_summary.csv")
    ap.add_argument("--output", default="outputs/stage2b/adaptive_mu/adaptive_mu_paired.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df["adaptive_mu"] = df["adaptive_mu"].astype(str).str.lower() == "true"

    rows = []
    for (attack, frac), cell in df.groupby(["attack", "fraction"], sort=True):
        adaptive = cell[cell.adaptive_mu].set_index("run_id").sort_index()
        fixed = cell[~cell.adaptive_mu].set_index("run_id").sort_index()
        shared = adaptive.index.intersection(fixed.index)
        for metric in METRICS:
            if metric not in cell.columns:
                continue
            res = paired_stats(fixed.loc[shared, metric], adaptive.loc[shared, metric])
            rows.append({"attack": attack, "fraction": frac, "metric": metric,
                         "mean_fixed": fixed.loc[shared, metric].mean(),
                         "mean_adaptive": adaptive.loc[shared, metric].mean(),
                         **res})

    # --- Decomposition of the distortion reduction -------------------------
    # Adaptive mu satisfies mu_eff >= base_mu pointwise and distortion is
    # (1 - mu) * |T - R|, so part of any reduction is guaranteed by the
    # definition rather than earned by a changed trajectory. Split it:
    #
    #   observed_i   = D_fixed(fixed state)      - D_adaptive(adaptive state)
    #   mechanical_i = D_fixed(fixed state)      - D_adaptive(fixed state)
    #   trajectory_i = D_adaptive(fixed state)   - D_adaptive(adaptive state)
    #
    # where D_adaptive(fixed state) is the counterfactual column recorded on
    # the fixed-mu arm. observed = mechanical + trajectory identically, so the
    # trajectory term is the only part that reflects the mechanism doing
    # something in the world rather than re-weighting the same numbers.
    decomp = []
    for (attack, frac), cell in df.groupby(["attack", "fraction"], sort=True):
        adaptive = cell[cell.adaptive_mu].set_index("run_id").sort_index()
        fixed = cell[~cell.adaptive_mu].set_index("run_id").sort_index()
        shared = adaptive.index.intersection(fixed.index)
        for suffix in ("", "_evidenced"):
            actual = f"decision_distortion{suffix}"
            cf = f"decision_distortion_counterfactual{suffix}"
            if cf not in cell.columns:
                continue
            observed = fixed.loc[shared, actual] - adaptive.loc[shared, actual]
            mechanical = fixed.loc[shared, actual] - fixed.loc[shared, cf]
            trajectory = fixed.loc[shared, cf] - adaptive.loc[shared, actual]
            traj_stats = paired_stats(fixed.loc[shared, cf], adaptive.loc[shared, actual])
            decomp.append({
                "attack": attack, "fraction": frac,
                "variant": "evidenced" if suffix else "all_pairs",
                "observed": observed.mean(), "mechanical": mechanical.mean(),
                "trajectory": trajectory.mean(),
                "mechanical_share_pct": 100.0 * mechanical.mean() / observed.mean()
                if observed.mean() else float("nan"),
                "traj_ci_low": traj_stats["ci_low"], "traj_ci_high": traj_stats["ci_high"],
                "traj_p": traj_stats["p"],
            })

    out = pd.DataFrame(rows)
    primary = out.metric == PRIMARY
    out.loc[primary, "p_holm"] = holm(out.loc[primary, "p"].tolist())
    out["pct_change"] = -100.0 * (out.mean_adaptive - out.mean_fixed) / out.mean_fixed

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    pd.set_option("display.width", 200, "display.max_rows", 200)
    for metric in METRICS:
        sub = out[out.metric == metric]
        if sub.empty:
            continue
        print(f"\n=== {metric} (positive mean_diff = adaptive mu REDUCES it) ===")
        cols = ["attack", "fraction", "n", "mean_fixed", "mean_adaptive", "mean_diff",
                "ci_low", "ci_high", "t", "d_z", "p", "pct_change"]
        if metric == PRIMARY:
            cols.append("p_holm")
        print(sub[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if decomp:
        dec = pd.DataFrame(decomp)
        dec_path = str(Path(args.output).with_name("adaptive_mu_decomposition.csv"))
        dec.to_csv(dec_path, index=False)
        print("\n=== distortion reduction: mechanical (identity) vs trajectory (real) ===")
        print(dec.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"wrote {dec_path}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
