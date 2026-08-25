"""Run proposed-framework ablation experiments."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asiot.ablation import ABLATION_VARIANTS
from asiot.config import load_config
from asiot.environment import ASIoTEnvironment
from asiot.logger import INTERACTION_COLUMNS, write_csv
from asiot.metrics import STEP_METRIC_COLUMNS
from scripts.aggregate_results import aggregate_ablation_outputs

DEFAULT_VARIANTS = ",".join(ABLATION_VARIANTS)


def main() -> None:
    """Run selected ablation variants and aggregate their CSV outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=2000)
    parser.add_argument("--load-level", default="high")
    parser.add_argument("--load-levels", default=None)
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--output-dir", default="outputs/ablation")
    parser.add_argument("--privacy-threshold-min", type=float, default=None)
    parser.add_argument("--privacy-threshold-max", type=float, default=None)
    parser.add_argument("--data-sensitivity-min", type=float, default=None)
    parser.add_argument("--data-sensitivity-max", type=float, default=None)
    args = parser.parse_args()

    load_levels = _parse_load_levels(args.load_levels or args.load_level)
    variants = _parse_variants(args.variants)
    output_root = Path(args.output_dir)
    if output_root.exists():
        shutil.rmtree(output_root, ignore_errors=True)
    base_config = load_config("config/default.yaml")
    completed = 0

    for load_level in load_levels:
        for variant in variants:
            for run_id in range(args.runs):
                seed = args.seed_start + run_id
                run_dir = output_root / load_level / variant / f"run_{run_id}"
                config = replace(
                    base_config,
                    steps=args.steps,
                    random_seed=seed,
                    load_level=load_level,
                    output_dir=run_dir,
                    privacy_threshold_min=(
                        base_config.privacy_threshold_min
                        if args.privacy_threshold_min is None
                        else args.privacy_threshold_min
                    ),
                    privacy_threshold_max=(
                        base_config.privacy_threshold_max
                        if args.privacy_threshold_max is None
                        else args.privacy_threshold_max
                    ),
                    data_sensitivity_min=(
                        base_config.data_sensitivity_min
                        if args.data_sensitivity_min is None
                        else args.data_sensitivity_min
                    ),
                    data_sensitivity_max=(
                        base_config.data_sensitivity_max
                        if args.data_sensitivity_max is None
                        else args.data_sensitivity_max
                    ),
                )
                env = ASIoTEnvironment(
                    config,
                    seed=seed,
                    baseline_name="proposed",
                    run_id=run_id,
                    ablation_variant=variant,
                )
                env.run(args.steps)
                write_csv(run_dir / "interactions.csv", env.logger.interactions, INTERACTION_COLUMNS)
                write_csv(run_dir / "steps.csv", env.logger.steps, STEP_METRIC_COLUMNS)
                completed += 1

    paths = aggregate_ablation_outputs(output_root, output_root / "aggregated")
    print(f"ablation_runs_completed={completed}")
    for name, path in paths.items():
        print(f"{name}={path}")
    print("load_level,ablation_variant,metric,mean")
    _print_compact_summary(paths["ablation_summary"])


def _parse_variants(raw: str) -> list[str]:
    variants = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = [name for name in variants if name not in ABLATION_VARIANTS]
    if unknown:
        raise ValueError(f"Unknown ablation variant: {', '.join(unknown)}")
    return variants


def _parse_load_levels(raw: str) -> list[str]:
    valid = {"low", "medium", "high", "extreme"}
    levels = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = [name for name in levels if name not in valid]
    if unknown:
        raise ValueError(f"Unknown load level: {', '.join(unknown)}")
    return levels


def _print_compact_summary(path: Path) -> None:
    import csv

    wanted = {"cooperation_rate", "task_completion_ratio", "privacy_exposure_risk", "avg_total_utility"}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["metric"] in wanted:
                print(
                    f"{row['load_level']},{row['ablation_variant']},"
                    f"{row['metric']},{float(row['mean']):.4f}"
                )


if __name__ == "__main__":
    main()
