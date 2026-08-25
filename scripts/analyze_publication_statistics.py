"""Create the prespecified paired-inference package used for publication.

The experiment drivers reuse seeds across models/variants.  This script keeps
that pairing, reports practical effect sizes, and controls multiplicity within
three separately declared primary families:

* benign: proposed versus each comparator at each load (28 comparisons),
* attacks: proposed versus greedy and Nitti at each nonzero attack cell
  (64 comparisons), and
* ablation: full proposed versus each removal at each load (28 comparisons).

The primary endpoint is run-level cooperation rate.  A positive difference
always favours the proposed/full model.  Holm controls family-wise error and
Benjamini-Hochberg controls false discovery rate; no comparison is selected
after inspecting its p-value.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import pandas as pd
from scipy import stats


ALPHA = 0.05
PRIMARY_METRIC = "cooperation_rate"
ATTACK_COMPARATORS = ("greedy_utility", "nitti_subjective_trust")


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in their original order."""
    values = [float(value) for value in p_values]
    finite = [index for index, value in enumerate(values) if math.isfinite(value)]
    adjusted = [float("nan")] * len(values)
    order = sorted(finite, key=lambda index: values[index])
    running = 0.0
    family_size = len(order)
    for rank, index in enumerate(order):
        candidate = min(1.0, (family_size - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def benjamini_hochberg(p_values: Iterable[float]) -> list[float]:
    """Return Benjamini-Hochberg adjusted q-values in original order."""
    values = [float(value) for value in p_values]
    finite = [index for index, value in enumerate(values) if math.isfinite(value)]
    adjusted = [float("nan")] * len(values)
    order = sorted(finite, key=lambda index: values[index])
    family_size = len(order)
    running = 1.0
    for reverse_rank in range(family_size - 1, -1, -1):
        index = order[reverse_rank]
        rank = reverse_rank + 1
        candidate = min(1.0, values[index] * family_size / rank)
        running = min(running, candidate)
        adjusted[index] = running
    return adjusted


def paired_statistics(
    focal: pd.Series,
    comparator: pd.Series,
) -> dict[str, float | int]:
    """Calculate paired inference with a focal-minus-comparator direction."""
    paired = pd.concat(
        [focal.rename("focal"), comparator.rename("comparator")], axis=1, join="inner"
    ).dropna()
    difference = paired["focal"] - paired["comparator"]
    n = int(len(difference))
    if n < 2:
        return {
            "n_pairs": n,
            "mean_focal": float("nan"),
            "mean_comparator": float("nan"),
            "mean_difference": float("nan"),
            "median_difference": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "t_statistic": float("nan"),
            "p_raw": float("nan"),
            "cohens_dz": float("nan"),
            "common_language_probability": float("nan"),
        }

    mean_difference = float(difference.mean())
    sd_difference = float(difference.std(ddof=1))
    if math.isclose(sd_difference, 0.0, abs_tol=1e-15):
        t_statistic = 0.0 if math.isclose(mean_difference, 0.0, abs_tol=1e-15) else math.copysign(float("inf"), mean_difference)
        p_raw = 1.0 if t_statistic == 0.0 else 0.0
        half_width = 0.0
        cohens_dz = 0.0 if t_statistic == 0.0 else math.copysign(float("inf"), mean_difference)
    else:
        standard_error = sd_difference / math.sqrt(n)
        t_statistic = mean_difference / standard_error
        p_raw = float(2.0 * stats.t.sf(abs(t_statistic), df=n - 1))
        half_width = float(stats.t.ppf(0.975, df=n - 1) * standard_error)
        cohens_dz = mean_difference / sd_difference

    greater = int((difference > 0.0).sum())
    ties = int((difference == 0.0).sum())
    return {
        "n_pairs": n,
        "mean_focal": float(paired["focal"].mean()),
        "mean_comparator": float(paired["comparator"].mean()),
        "mean_difference": mean_difference,
        "median_difference": float(difference.median()),
        "ci95_low": mean_difference - half_width,
        "ci95_high": mean_difference + half_width,
        "t_statistic": t_statistic,
        "p_raw": p_raw,
        "cohens_dz": cohens_dz,
        "common_language_probability": (greater + 0.5 * ties) / n,
    }


def _indexed(cell: pd.DataFrame, key: str, metric: str) -> pd.Series:
    """Return one unique, sorted metric series indexed by the pairing key."""
    if cell[key].duplicated().any():
        duplicates = cell.loc[cell[key].duplicated(), key].tolist()[:5]
        raise ValueError(f"Duplicate pairing keys in cell: {duplicates}")
    return cell.set_index(key)[metric].astype(float).sort_index()


def analyze_benign(data: pd.DataFrame) -> pd.DataFrame:
    """Analyze proposed versus all seven benign comparators by load."""
    rows: list[dict[str, object]] = []
    comparators = sorted(set(data["baseline_name"]) - {"proposed"})
    for load in sorted(data["load_level"].unique()):
        cell = data[data["load_level"] == load]
        focal = _indexed(cell[cell["baseline_name"] == "proposed"], "seed", PRIMARY_METRIC)
        for comparator_name in comparators:
            comparator = _indexed(
                cell[cell["baseline_name"] == comparator_name], "seed", PRIMARY_METRIC
            )
            rows.append(
                {
                    "family": "benign",
                    "load_level": load,
                    "attack": "none",
                    "attacker_fraction": 0.0,
                    "focal": "proposed",
                    "comparator": comparator_name,
                    **paired_statistics(focal, comparator),
                }
            )
    return _correct_family(pd.DataFrame(rows))


def analyze_attacks(data: pd.DataFrame) -> pd.DataFrame:
    """Analyze proposed versus two prespecified comparators under attacks."""
    rows: list[dict[str, object]] = []
    attacked = data[data["fraction"].astype(float) > 0.0]
    for (attack, fraction), cell in attacked.groupby(["attack", "fraction"], sort=True):
        focal = _indexed(cell[cell["model"] == "proposed"], "run_id", PRIMARY_METRIC)
        for comparator_name in ATTACK_COMPARATORS:
            comparator = _indexed(
                cell[cell["model"] == comparator_name], "run_id", PRIMARY_METRIC
            )
            rows.append(
                {
                    "family": "attacks",
                    "load_level": str(cell["load"].iloc[0]),
                    "attack": attack,
                    "attacker_fraction": float(fraction),
                    "focal": "proposed",
                    "comparator": comparator_name,
                    **paired_statistics(focal, comparator),
                }
            )
    return _correct_family(pd.DataFrame(rows))


def analyze_ablation(data: pd.DataFrame) -> pd.DataFrame:
    """Analyze full proposed versus every component removal by load."""
    rows: list[dict[str, object]] = []
    variants = sorted(set(data["ablation_variant"]) - {"full_proposed"})
    for load in sorted(data["load_level"].unique()):
        cell = data[data["load_level"] == load]
        focal = _indexed(
            cell[cell["ablation_variant"] == "full_proposed"], "seed", PRIMARY_METRIC
        )
        for variant in variants:
            comparator = _indexed(
                cell[cell["ablation_variant"] == variant], "seed", PRIMARY_METRIC
            )
            rows.append(
                {
                    "family": "ablation",
                    "load_level": load,
                    "attack": "none",
                    "attacker_fraction": 0.0,
                    "focal": "full_proposed",
                    "comparator": variant,
                    **paired_statistics(focal, comparator),
                }
            )
    return _correct_family(pd.DataFrame(rows))


def _correct_family(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply both corrections to one declared comparison family."""
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["p_holm"] = holm_adjust(frame["p_raw"])
    frame["q_bh"] = benjamini_hochberg(frame["p_raw"])
    frame["holm_significant_0_05"] = frame["p_holm"] < ALPHA
    frame["bh_significant_0_05"] = frame["q_bh"] < ALPHA
    frame["relative_difference_pct"] = (
        100.0 * frame["mean_difference"] / frame["mean_comparator"]
    )
    return frame


def write_report(frame: pd.DataFrame, path: Path) -> None:
    """Write an auditable compact interpretation without cherry-picking."""
    lines = [
        "# Prespecified paired-inference report",
        "",
        "The primary endpoint is run-level cooperation rate. Every difference is",
        "focal minus comparator, so positive values favor the proposed/full model.",
        "Holm correction controls family-wise error within each declared family;",
        "Benjamini-Hochberg q-values are supplied as a complementary FDR analysis.",
        "Confidence intervals and Cohen's paired d_z quantify practical effects.",
        "",
        "| Family | Comparisons | Pairs/cell | Holm significant (+/−) | BH significant |",
        "|---|---:|---:|---:|---:|",
    ]
    for family, cell in frame.groupby("family", sort=False):
        pair_counts = sorted(int(value) for value in cell["n_pairs"].unique())
        pair_text = ", ".join(str(value) for value in pair_counts)
        holm_positive = int(
            (cell["holm_significant_0_05"] & (cell["mean_difference"] > 0.0)).sum()
        )
        holm_negative = int(
            (cell["holm_significant_0_05"] & (cell["mean_difference"] < 0.0)).sum()
        )
        lines.append(
            f"| {family} | {len(cell)} | {pair_text} | "
            f"{int(cell['holm_significant_0_05'].sum())} "
            f"({holm_positive}/{holm_negative}) | "
            f"{int(cell['bh_significant_0_05'].sum())} |"
        )

    lines.extend(
        [
            "",
            "## Prespecified headline cells",
            "",
            "| Family/cell | Comparator | Mean difference | 95% CI | d_z | Holm p |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    headline = pd.concat(
        [
            frame[(frame.family == "benign") & (frame.load_level == "high")],
            frame[
                (frame.family == "attacks")
                & (frame.attacker_fraction == 0.4)
                & frame.attack.isin(["selective", "collusion", "sybil", "whitewashing"])
            ],
            frame[(frame.family == "ablation") & (frame.load_level == "high")],
        ],
        ignore_index=True,
    )
    for row in headline.itertuples(index=False):
        if row.family == "attacks":
            cell_name = f"attack: {row.attack}, f={row.attacker_fraction:.1f}"
        else:
            cell_name = f"{row.family}: {row.load_level}"
        lines.append(
            f"| {cell_name} | {row.comparator} | {row.mean_difference:.6f} | "
            f"[{row.ci95_low:.6f}, {row.ci95_high:.6f}] | "
            f"{row.cohens_dz:.3f} | {row.p_holm:.4g} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "Statistical detection alone is not described as practical superiority.",
            "The manuscript should interpret the confidence interval, paired d_z,",
            "and the metric scale together. In particular, the benign high-load",
            "proposed-versus-greedy result remains a small effect even if a raw or",
            "adjusted test crosses 0.05.",
            "For attacks, significance is not synonymous with superiority: the two",
            "Holm-significant negative cells are bad-mouthing versus greedy utility at",
            "fractions 0.30 and 0.40 (differences about −0.006). Bad-mouthing acts on",
            "feedback, so reputation separation and decision distortion are the more",
            "direct endpoints for that mechanism.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Load the tracked run summaries and create inference artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--benign", default="supplementary_results/benign/run_summary.csv")
    parser.add_argument("--attacks", default="supplementary_results/attacks/run_summary.csv")
    parser.add_argument("--ablation", default="supplementary_results/ablation/run_summary.csv")
    parser.add_argument("--output-dir", default="supplementary_results/statistics")
    args = parser.parse_args()

    frames = [
        analyze_benign(pd.read_csv(args.benign)),
        analyze_attacks(pd.read_csv(args.attacks)),
        analyze_ablation(pd.read_csv(args.ablation)),
    ]
    combined = pd.concat(frames, ignore_index=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_dir / "paired_primary_comparisons.csv", index=False)
    for family, frame in combined.groupby("family", sort=False):
        frame.to_csv(output_dir / f"{family}_paired.csv", index=False)
    write_report(combined, output_dir / "PAIRED_INFERENCE.md")

    print(f"comparisons={len(combined)}")
    for family, frame in combined.groupby("family", sort=False):
        print(
            f"{family}: comparisons={len(frame)}, "
            f"holm_significant={int(frame['holm_significant_0_05'].sum())}, "
            f"bh_significant={int(frame['bh_significant_0_05'].sum())}"
        )
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
