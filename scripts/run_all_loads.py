"""Run all selected baselines across selected load levels and aggregate outputs."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asiot.baselines import BASELINE_REGISTRY
from asiot.config import load_config
from asiot.environment import ASIoTEnvironment
from asiot.logger import INTERACTION_COLUMNS, write_csv
from asiot.metrics import STEP_METRIC_COLUMNS
from scripts.aggregate_results import aggregate_outputs

DEFAULT_LOAD_LEVELS = ("low", "medium", "high", "extreme")


def main() -> None:
    """Run a reproducible multi-load baseline sweep."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--output-dir", default="outputs/all_loads")
    parser.add_argument("--baselines", default=",".join(BASELINE_REGISTRY))
    parser.add_argument("--load-levels", default=",".join(DEFAULT_LOAD_LEVELS))
    args = parser.parse_args()

    baselines = _parse_names(args.baselines, set(BASELINE_REGISTRY), "baseline")
    load_levels = _parse_names(args.load_levels, set(DEFAULT_LOAD_LEVELS), "load level")
    output_root = Path(args.output_dir)
    base_config = load_config("config/default.yaml")
    completed = 0

    for load_level in load_levels:
        for baseline_name in baselines:
            for run_id in range(args.runs):
                seed = args.seed_start + run_id
                run_dir = output_root / load_level / baseline_name / f"run_{run_id}"
                config = replace(
                    base_config,
                    steps=args.steps,
                    random_seed=seed,
                    load_level=load_level,
                    output_dir=run_dir,
                )
                env = ASIoTEnvironment(
                    config,
                    seed=seed,
                    baseline_name=baseline_name,
                    run_id=run_id,
                )
                env.run(args.steps)
                write_csv(run_dir / "interactions.csv", env.logger.interactions, INTERACTION_COLUMNS)
                write_csv(run_dir / "steps.csv", env.logger.steps, STEP_METRIC_COLUMNS)
                completed += 1

    aggregated_dir = output_root / "aggregated"
    paths = aggregate_outputs(output_root, aggregated_dir)
    print(f"runs_completed={completed}")
    for name, path in paths.items():
        print(f"{name}={path}")
    print("load_level,baseline_name,metric,mean")
    _print_compact_summary(paths["summary_by_load_baseline"])


def _parse_names(raw: str, allowed: set[str], label: str) -> list[str]:
    names = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = [name for name in names if name not in allowed]
    if unknown:
        raise ValueError(f"Unknown {label}: {', '.join(unknown)}")
    return names


def _print_compact_summary(path: Path) -> None:
    import csv

    wanted = {"cooperation_rate", "task_completion_ratio", "avg_total_utility"}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["metric"] in wanted:
                print(
                    f"{row['load_level']},{row['baseline_name']},"
                    f"{row['metric']},{float(row['mean']):.4f}"
                )


if __name__ == "__main__":
    main()
