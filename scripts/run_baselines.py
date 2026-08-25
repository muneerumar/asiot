"""Run all Phase 5 baseline policies for a short comparison."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asiot.baselines import BASELINE_REGISTRY
from asiot.config import load_config
from asiot.environment import ASIoTEnvironment
from asiot.logger import INTERACTION_COLUMNS, write_csv
from asiot.metrics import STEP_METRIC_COLUMNS


def main() -> None:
    """Run every registered baseline and write per-baseline plus combined CSVs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--load-level",
        choices=["low", "medium", "high", "extreme"],
        default="low",
    )
    parser.add_argument("--output-dir", default="outputs/baselines")
    args = parser.parse_args()

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    base_config = load_config("config/default.yaml")
    combined_interactions: list[dict[str, Any]] = []
    combined_steps: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for baseline_name in BASELINE_REGISTRY:
        config = replace(
            base_config,
            steps=args.steps,
            random_seed=args.seed,
            load_level=args.load_level,
            output_dir=root / baseline_name,
        )
        env = ASIoTEnvironment(config, seed=args.seed, baseline_name=baseline_name, run_id=0)
        result = env.run(args.steps)
        baseline_dir = root / baseline_name
        baseline_dir.mkdir(parents=True, exist_ok=True)
        write_csv(baseline_dir / "interactions.csv", env.logger.interactions, INTERACTION_COLUMNS)
        write_csv(baseline_dir / "steps.csv", env.logger.steps, STEP_METRIC_COLUMNS)
        combined_interactions.extend(env.logger.interactions)
        combined_steps.extend(env.logger.steps)
        summary = dict(result["final_summary"])
        summary["baseline_name"] = baseline_name
        summaries.append(summary)

    write_csv(root / "combined_interactions.csv", combined_interactions, INTERACTION_COLUMNS)
    write_csv(root / "combined_steps.csv", combined_steps, STEP_METRIC_COLUMNS)

    print("baseline_name,total_tasks,completed_tasks,cooperation_rate,active_nodes,average_trust")
    for summary in summaries:
        print(
            f"{summary['baseline_name']},{summary['total_tasks']},"
            f"{summary['completed_tasks']},{summary['cooperation_rate']:.4f},"
            f"{summary['active_nodes']},{summary['average_trust']:.4f}"
        )

if __name__ == "__main__":
    main()
