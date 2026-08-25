"""Generate paper-ready figures from ASIoT aggregated CSV outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asiot.plotting.plots import (
    generate_ablation_figures,
    generate_baseline_figures,
    generate_time_series_figures,
)


def main() -> None:
    """Generate baseline, ablation, and time-series figures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-agg-dir", default="outputs/all_loads/aggregated")
    parser.add_argument("--ablation-agg-dir", default="outputs/ablation/aggregated")
    parser.add_argument("--output-dir", default="outputs/figures")
    parser.add_argument("--load-level", default="high")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-time-series", action="store_true")
    args = parser.parse_args()

    generated: list[Path] = []
    if not args.skip_baselines:
        generated.extend(generate_baseline_figures(args.baseline_agg_dir, args.output_dir))
    if not args.skip_ablation:
        generated.extend(generate_ablation_figures(args.ablation_agg_dir, args.output_dir))
    if not args.skip_time_series:
        generated.extend(
            generate_time_series_figures(
                args.baseline_agg_dir,
                args.output_dir,
                load_level=args.load_level,
            )
        )

    print("generated_figures")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
