"""Learning curves for the MARL training runs (Task A deliverable).

Plots the periodic frozen (epsilon=0) evaluation from the training logs:
eval_reward and eval_cooperation_rate versus training episode, with the
epsilon decay schedule on a secondary axis. Mean +/- 95% CI across the three
training seeds per variant.

The curve is plotted ONLY from the frozen evaluation points (trainer.py:143-149
comment: "never from `reward` above, which is confounded by exploration and by
how many tasks a seed generated"). eval_reward is a cumulative episode sum over
transition rewards (trainer.py:275-285); eval_cooperation_rate is the
per-episode cooperation fraction of the frozen policy.

Input: the six training_log.csv files under outputs/marl_full_run2/
Output: outputs/marl_figures/marl_learning_curves.png
        outputs/marl_figures/marl_learning_curves.pdf
        outputs/marl_figures/marl_learning_curves.csv (plotted points)

Usage:
    python scripts/plot_marl_learning_curves.py
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED_STARTS = (3000, 6000, 9000)
VARIANTS = ("neural_marl_social", "neural_marl_no_social")


def _load_log(path: Path) -> list[dict]:
    with open(path) as handle:
        return [r for r in csv.DictReader(handle) if r["eval_reward"]]


def _mean_ci(values: list[float]) -> tuple[float, float]:
    m = statistics.mean(values)
    if len(values) < 2:
        return m, 0.0
    return m, 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="outputs/marl_full_run2")
    ap.add_argument("--out-dir", default="outputs/marl_figures")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    series: dict[str, list[dict]] = {}
    for variant in VARIANTS:
        rows_by_seed = []
        for seed_start in SEED_STARTS:
            log_path = (run_root / f"{variant}_seed{seed_start}"
                        / "high" / variant / "training" / "training_log.csv")
            rows_by_seed.append(_load_log(log_path))
        series[variant] = rows_by_seed

    for variant, color in (("neural_marl_social", "#1f77b4"),
                           ("neural_marl_no_social", "#d62728")):
        rows_by_seed = series[variant]
        episodes = [int(r["episode"]) for r in rows_by_seed[0]]

        for ax, metric in ((axes[0], "eval_reward"),
                           (axes[1], "eval_cooperation_rate")):
            means, cis = [], []
            all_rows = [row for seed_rows in rows_by_seed for row in seed_rows]
            for i, ep in enumerate(episodes):
                vals = [float(r[metric]) for r in all_rows if int(r["episode"]) == ep]
                m, ci = _mean_ci(vals)
                means.append(m)
                cis.append(ci)
            ax.plot(episodes, means, color=color, label=variant.replace(
                "neural_marl_", ""), lw=1.8)
            ax.fill_between(episodes,
                            [m - c for m, c in zip(means, cis)],
                            [m + c for m, c in zip(means, cis)],
                            color=color, alpha=0.18)

    # epsilon schedule on the secondary axis (same for all seeds)
    eps_by_ep = {int(r["episode"]): float(r["epsilon"]) for r in series[VARIANTS[0]][0]}
    for ax, label, ylabel in (
            (axes[0], "Frozen evaluation reward (sum over 500 steps)",
             "cumulative episode reward"),
            (axes[1], "Frozen evaluation cooperation rate",
             "cooperation rate")):
        ax2 = ax.twinx()
        ax2.plot(list(eps_by_ep), list(eps_by_ep.values()), color="#7f7f7f",
                 ls="--", lw=1.0, alpha=0.8, label=r"$\epsilon$ schedule")
        ax2.set_ylabel(r"$\epsilon$", fontsize=10)
        ax2.set_ylim(-0.05, 1.05)
        ax.set_xlabel("training episode")
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3)

    lines1, labels1 = axes[0].get_legend_handles_labels()
    lines2, labels2 = axes[0].get_legend_handles_labels()
    axes[0].legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")
    fig.tight_layout()

    fig.savefig(out_dir / "marl_learning_curves.png", dpi=150)
    fig.savefig(out_dir / "marl_learning_curves.pdf")

    # csv of plotted points
    with open(out_dir / "marl_learning_curves.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "episode", "metric", "mean", "ci95"])
        for variant, rows_by_seed in series.items():
            all_rows = [row for seed_rows in rows_by_seed for row in seed_rows]
            for metric in ("eval_reward", "eval_cooperation_rate"):
                for i, ep in enumerate(episodes):
                    vals = [float(r[metric]) for r in all_rows
                            if int(r["episode"]) == ep]
                    m, c = _mean_ci(vals)
                    writer.writerow([variant, ep, metric, f"{m:.6f}", f"{c:.6f}"])

    print(f"wrote {out_dir}/marl_learning_curves.{{png,pdf,csv}}")


if __name__ == "__main__":
    main()