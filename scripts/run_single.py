"""Run a single ASIoT environment simulation."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asiot.config import load_config
from asiot.environment import ASIoTEnvironment
from asiot.baselines import BASELINE_REGISTRY


def main() -> None:
    """Load config, run the dynamic environment, save raw CSVs, and summarize."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--load-level",
        choices=["low", "medium", "high", "extreme"],
        default=None,
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--baseline", choices=sorted(BASELINE_REGISTRY), default="proposed")
    parser.add_argument("--run-id", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    config = replace(
        config,
        steps=args.steps if args.steps is not None else config.steps,
        random_seed=args.seed if args.seed is not None else config.random_seed,
        load_level=args.load_level if args.load_level is not None else config.load_level,
        output_dir=Path(args.output_dir) if args.output_dir is not None else config.output_dir,
    )
    env = ASIoTEnvironment(
        config,
        seed=config.random_seed,
        baseline_name=args.baseline,
        run_id=args.run_id,
    )
    result = env.run(config.steps)
    output_paths = env.logger.save_raw_outputs(config.output_dir)
    summary = result["final_summary"]

    print(f"total_tasks={summary['total_tasks']}")
    print(f"baseline_name={args.baseline}")
    print(f"completed_tasks={summary['completed_tasks']}")
    print(f"cooperation_rate={summary['cooperation_rate']:.4f}")
    print(f"active_nodes={summary['active_nodes']}")
    print(f"average_trust={summary['average_trust']:.4f}")
    print(f"output_directory={config.output_dir}")
    print(f"interactions_csv={output_paths['interactions']}")
    print(f"steps_csv={output_paths['steps']}")


if __name__ == "__main__":
    main()
