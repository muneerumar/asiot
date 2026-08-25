"""Create the three compact figures used by the validated-results package."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "outputs/.matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOADS = ("low", "medium", "high", "extreme")
COLORS = {
    "proposed": "#176B87",
    "greedy_utility": "#D97706",
    "nitti_subjective_trust": "#6D597A",
    "standard_marl_no_social": "#64748B",
}
LABELS = {
    "proposed": "Proposed A-SIoT",
    "greedy_utility": "Greedy utility",
    "nitti_subjective_trust": "Nitti subjective trust",
    "standard_marl_no_social": "Non-social heuristic",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benign-summary",
        default="supplementary_results/benign/summary_by_load_baseline.csv",
    )
    parser.add_argument(
        "--attack-summary",
        default="supplementary_results/attacks/attack_summary.csv",
    )
    parser.add_argument(
        "--ablation-summary",
        default="supplementary_results/ablation/ablation_summary.csv",
    )
    parser.add_argument("--png-dir", default="supplementary_results/figures")
    parser.add_argument("--pdf-dir", default="output/pdf")
    args = parser.parse_args()

    png_dir = Path(args.png_dir)
    pdf_dir = Path(args.pdf_dir)
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    _style()
    _plot_benign(_read(args.benign_summary), png_dir, pdf_dir)
    _plot_attacks(_read(args.attack_summary), png_dir, pdf_dir)
    _plot_ablation(_read(args.ablation_summary), png_dir, pdf_dir)


def _plot_benign(rows: list[dict[str, str]], png_dir: Path, pdf_dir: Path) -> None:
    models = (
        "proposed",
        "greedy_utility",
        "nitti_subjective_trust",
        "standard_marl_no_social",
    )
    lookup = {
        (row["load_level"], row["baseline_name"]): row
        for row in rows
        if row["metric"] == "cooperation_rate"
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.25))
    for model in models:
        means = [float(lookup[(load, model)]["mean"]) for load in LOADS]
        low = [float(lookup[(load, model)]["ci95_low"]) for load in LOADS]
        high = [float(lookup[(load, model)]["ci95_high"]) for load in LOADS]
        x = list(range(len(LOADS)))
        ax.plot(x, means, marker="o", linewidth=2.0, label=LABELS[model], color=COLORS[model])
        ax.fill_between(x, low, high, alpha=0.12, color=COLORS[model])
    ax.set_xticks(range(len(LOADS)), [load.title() for load in LOADS])
    ax.set_ylim(0.35, 0.92)
    ax.set_xlabel("Workload")
    ax.set_ylabel("Cooperation rate")
    ax.set_title("Benign cooperation across paired workloads (100 seeds)")
    ax.legend(frameon=False, ncol=2, loc="lower left")
    _finish(fig, "benign_cooperation", png_dir, pdf_dir)


def _plot_attacks(rows: list[dict[str, str]], png_dir: Path, pdf_dir: Path) -> None:
    attacks = ("selective", "collusion", "whitewashing")
    models = ("proposed", "greedy_utility", "nitti_subjective_trust")
    lookup = {
        (row["attack"], row["model"], float(row["fraction"])): row
        for row in rows
        if row["metric"] == "cooperation_rate"
    }
    fractions = (0.1, 0.2, 0.3, 0.4)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.7), sharey=True)
    for ax, attack in zip(axes, attacks):
        for model in models:
            selected = [lookup[(attack, model, fraction)] for fraction in fractions]
            means = [float(row["mean"]) for row in selected]
            low = [float(row["ci95_low"]) for row in selected]
            high = [float(row["ci95_high"]) for row in selected]
            ax.plot(
                fractions,
                means,
                marker="o",
                linewidth=1.9,
                label=LABELS[model],
                color=COLORS[model],
            )
            ax.fill_between(fractions, low, high, alpha=0.12, color=COLORS[model])
        ax.set_title(attack.replace("_", " ").title())
        ax.set_xlabel("Attacker fraction")
        ax.set_xticks(fractions)
        ax.set_ylim(0.35, 0.78)
    axes[0].set_ylabel("Cooperation rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center")
    fig.suptitle("Resilience at high load (50 paired seeds)", y=0.99)
    fig.subplots_adjust(bottom=0.23, top=0.82, wspace=0.12)
    _finish(fig, "attack_resilience", png_dir, pdf_dir, tight=False)


def _plot_ablation(rows: list[dict[str, str]], png_dir: Path, pdf_dir: Path) -> None:
    selected = [
        row
        for row in rows
        if row["load_level"] == "high" and row["metric"] == "cooperation_rate"
    ]
    order = (
        "full_proposed",
        "without_trust",
        "without_preference",
        "without_reciprocity",
        "without_privacy_gate",
        "without_resource_awareness",
        "without_incentive",
        "without_social_graph_adaptation",
    )
    lookup = {row["ablation_variant"]: row for row in selected}
    values = [float(lookup[name]["mean"]) for name in order]
    errors = [
        float(lookup[name]["ci95_high"]) - float(lookup[name]["mean"]) for name in order
    ]
    labels = [
        "Full",
        "No trust",
        "No preference",
        "No reciprocity",
        "No privacy gate",
        "No resources",
        "No incentive",
        "Static social graph",
    ]
    colors = ["#176B87"] + ["#94A3B8"] * (len(order) - 1)
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    ax.bar(range(len(order)), values, yerr=errors, capsize=3, color=colors)
    ax.set_xticks(range(len(order)), labels, rotation=25, ha="right")
    ax.set_ylim(0.45, 0.84)
    ax.set_ylabel("Cooperation rate")
    ax.set_title("Behavioral ablation at high load (50 paired seeds)")
    _finish(fig, "ablation_high_load", png_dir, pdf_dir)


def _finish(
    fig,
    name: str,
    png_dir: Path,
    pdf_dir: Path,
    *,
    tight: bool = True,
) -> None:
    if tight:
        fig.tight_layout()
    fig.savefig(png_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(pdf_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(png_dir / f"{name}.png")
    print(pdf_dir / f"{name}.pdf")


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
        }
    )


def _read(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
