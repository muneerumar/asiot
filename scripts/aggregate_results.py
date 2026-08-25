"""Aggregate raw ASIoT CSV outputs into run and load/baseline summaries."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asiot.logger import INTERACTION_COLUMNS, write_csv
from asiot.metrics import STEP_METRIC_COLUMNS

RUN_SUMMARY_COLUMNS = (
    "run_id",
    "seed",
    "load_level",
    "baseline_name",
    "cooperation_rate",
    "task_completion_ratio",
    "resource_efficiency",
    "reliability_score",
    "fairness_index",
    "trust_stability_index",
    "privacy_exposure_risk",
    "candidate_privacy_risk",
    "accepted_privacy_exposure",
    "throughput_mbps",
    "packet_delivery_ratio",
    "e2e_delay_ms",
    "communication_overhead",
    "avg_total_utility",
    "active_nodes",
)

SUMMARY_LONG_COLUMNS = (
    "load_level",
    "baseline_name",
    "metric",
    "mean",
    "std",
    "count",
    "ci95_low",
    "ci95_high",
)

WIDE_COLUMNS = (
    "load_level",
    "baseline_name",
    "cooperation_rate_mean",
    "cooperation_rate_std",
    "task_completion_ratio_mean",
    "task_completion_ratio_std",
    "resource_efficiency_mean",
    "resource_efficiency_std",
    "reliability_score_mean",
    "reliability_score_std",
    "fairness_index_mean",
    "fairness_index_std",
    "trust_stability_index_mean",
    "trust_stability_index_std",
    "privacy_exposure_risk_mean",
    "privacy_exposure_risk_std",
    "candidate_privacy_risk_mean",
    "candidate_privacy_risk_std",
    "throughput_mbps_mean",
    "packet_delivery_ratio_mean",
    "e2e_delay_ms_mean",
    "communication_overhead_mean",
    "avg_total_utility_mean",
)

ABLATION_RUN_SUMMARY_COLUMNS = (
    "run_id",
    "seed",
    "load_level",
    "ablation_variant",
    "cooperation_rate",
    "task_completion_ratio",
    "resource_efficiency",
    "reliability_score",
    "fairness_index",
    "trust_stability_index",
    "privacy_exposure_risk",
    "candidate_privacy_risk",
    "accepted_privacy_exposure",
    "throughput_mbps",
    "packet_delivery_ratio",
    "e2e_delay_ms",
    "communication_overhead",
    "avg_total_utility",
    "active_nodes",
)

ABLATION_SUMMARY_COLUMNS = (
    "load_level",
    "ablation_variant",
    "metric",
    "mean",
    "std",
    "count",
    "ci95_low",
    "ci95_high",
)

ABLATION_WIDE_COLUMNS = (
    "load_level",
    "ablation_variant",
    "cooperation_rate_mean",
    "cooperation_rate_std",
    "task_completion_ratio_mean",
    "task_completion_ratio_std",
    "resource_efficiency_mean",
    "resource_efficiency_std",
    "reliability_score_mean",
    "reliability_score_std",
    "fairness_index_mean",
    "fairness_index_std",
    "trust_stability_index_mean",
    "trust_stability_index_std",
    "privacy_exposure_risk_mean",
    "privacy_exposure_risk_std",
    "candidate_privacy_risk_mean",
    "candidate_privacy_risk_std",
    "avg_total_utility_mean",
    "active_nodes_mean",
)

SUMMARY_METRICS = (
    "cooperation_rate",
    "task_completion_ratio",
    "resource_efficiency",
    "reliability_score",
    "fairness_index",
    "trust_stability_index",
    "privacy_exposure_risk",
    "candidate_privacy_risk",
    "accepted_privacy_exposure",
    "throughput_mbps",
    "packet_delivery_ratio",
    "e2e_delay_ms",
    "communication_overhead",
    "avg_total_utility",
)

BOUNDED_METRICS = {
    "cooperation_rate",
    "task_completion_ratio",
    "reliability_score",
    "fairness_index",
    "trust_stability_index",
    "privacy_exposure_risk",
    "candidate_privacy_risk",
    "accepted_privacy_exposure",
    "packet_delivery_ratio",
    "communication_overhead",
}


def load_step_csvs(input_dir: str | Path) -> list[dict[str, Any]]:
    """Recursively load all leaf ``steps.csv`` files under an input directory."""
    return _load_named_csvs(input_dir, "steps.csv")


def load_interaction_csvs(input_dir: str | Path) -> list[dict[str, Any]]:
    """Recursively load all leaf ``interactions.csv`` files under an input directory."""
    return _load_named_csvs(input_dir, "interactions.csv")


def aggregate_by_run(steps_df: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-step rows to one summary row per run/load/baseline."""
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in steps_df:
        key = (
            str(row.get("run_id", 0)),
            str(row.get("seed", 0)),
            str(row.get("load_level", "low")),
            str(row.get("baseline_name", "proposed")),
        )
        grouped[key].append(row)

    summaries = []
    for (run_id, seed, load_level, baseline_name), rows in sorted(grouped.items()):
        total_generated = sum(_to_int(row.get("generated_tasks")) for row in rows)
        total_completed = sum(_to_int(row.get("completed_tasks")) for row in rows)
        total_attempted = sum(_to_int(row.get("attempted_interactions")) for row in rows)
        total_successful = sum(_to_int(row.get("successful_cooperations")) for row in rows)
        total_failed = sum(_to_int(row.get("failed_interactions")) for row in rows)
        total_blocked = sum(_to_int(row.get("blocked_privacy_interactions")) for row in rows)
        energy = sum(_to_float(row.get("energy_consumed")) for row in rows)
        bandwidth = sum(_to_float(row.get("bandwidth_consumed")) for row in rows)
        compute = sum(_to_float(row.get("compute_consumed")) for row in rows)
        final_active = _to_int(rows[-1].get("active_nodes")) if rows else 0
        summaries.append(
            {
                "run_id": _to_int(run_id),
                "seed": _to_int(seed),
                "load_level": load_level,
                "baseline_name": baseline_name,
                "cooperation_rate": _safe_ratio(total_successful, total_attempted),
                "task_completion_ratio": _safe_ratio(total_completed, total_generated),
                "resource_efficiency": _resource_efficiency(
                    total_completed,
                    total_generated,
                    energy,
                    bandwidth,
                    compute,
                ),
                "reliability_score": _safe_ratio(
                    total_successful,
                    total_successful + total_failed + total_blocked,
                ),
                "fairness_index": _mean_column(rows, "fairness_index"),
                "trust_stability_index": _mean_column(rows, "trust_stability_index"),
                "privacy_exposure_risk": _mean_column(rows, "privacy_exposure_risk"),
                "candidate_privacy_risk": _mean_column(rows, "candidate_privacy_risk"),
                "accepted_privacy_exposure": _mean_column(rows, "accepted_privacy_exposure"),
                "throughput_mbps": _mean_column(rows, "throughput_mbps"),
                "packet_delivery_ratio": _mean_column(rows, "packet_delivery_ratio"),
                "e2e_delay_ms": _mean_nonzero_column(rows, "e2e_delay_ms"),
                "communication_overhead": _mean_column(rows, "communication_overhead"),
                "avg_total_utility": _mean_column(rows, "avg_total_utility"),
                "active_nodes": final_active,
            }
        )
    return summaries


def aggregate_by_load_baseline(run_summary_df: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate run summaries by load level and baseline with 95% CIs."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in run_summary_df:
        grouped[(str(row["load_level"]), str(row["baseline_name"]))].append(row)

    summary_rows = []
    for (load_level, baseline_name), rows in sorted(grouped.items()):
        for metric in SUMMARY_METRICS:
            values = [_to_float(row.get(metric)) for row in rows]
            count = len(values)
            avg = sum(values) / count if count else 0.0
            std = _sample_std(values)
            ci95 = 1.96 * std / math.sqrt(count) if count else 0.0
            ci_low = avg - ci95
            ci_high = avg + ci95
            if metric in BOUNDED_METRICS:
                ci_low = _clip01(ci_low)
                ci_high = _clip01(ci_high)
            summary_rows.append(
                {
                    "load_level": load_level,
                    "baseline_name": baseline_name,
                    "metric": metric,
                    "mean": avg,
                    "std": std,
                    "count": count,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                }
            )
    return summary_rows


def create_wide_summary(summary_long_df: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create a compact wide summary table from long-format metric summaries."""
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in summary_long_df:
        key = (str(row["load_level"]), str(row["baseline_name"]))
        wide = grouped.setdefault(
            key,
            {"load_level": key[0], "baseline_name": key[1]},
        )
        metric = str(row["metric"])
        if f"{metric}_mean" in WIDE_COLUMNS:
            wide[f"{metric}_mean"] = row["mean"]
        if f"{metric}_std" in WIDE_COLUMNS:
            wide[f"{metric}_std"] = row["std"]
    return [
        {column: row.get(column, 0.0) for column in WIDE_COLUMNS}
        for row in grouped.values()
    ]


def aggregate_outputs(input_dir: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Load raw CSVs, aggregate them, and write all Phase 6 output tables."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    steps = load_step_csvs(input_dir)
    interactions = load_interaction_csvs(input_dir)
    run_summary = aggregate_by_run(steps)
    summary_long = aggregate_by_load_baseline(run_summary)
    summary_wide = create_wide_summary(summary_long)

    paths = {
        "combined_steps": output / "combined_steps.csv",
        "combined_interactions": output / "combined_interactions.csv",
        "run_summary": output / "run_summary.csv",
        "summary_by_load_baseline": output / "summary_by_load_baseline.csv",
        "summary_by_load_baseline_wide": output / "summary_by_load_baseline_wide.csv",
    }
    write_csv(paths["combined_steps"], steps, STEP_METRIC_COLUMNS)
    write_csv(paths["combined_interactions"], interactions, INTERACTION_COLUMNS)
    write_csv(paths["run_summary"], run_summary, RUN_SUMMARY_COLUMNS)
    write_csv(paths["summary_by_load_baseline"], summary_long, SUMMARY_LONG_COLUMNS)
    write_csv(paths["summary_by_load_baseline_wide"], summary_wide, WIDE_COLUMNS)
    return paths


def aggregate_ablation_by_run(steps_df: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate ablation step rows by run/load/variant."""
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in steps_df:
        key = (
            str(row.get("run_id", 0)),
            str(row.get("seed", 0)),
            str(row.get("load_level", "low")),
            str(row.get("ablation_variant", "none")),
        )
        grouped[key].append(row)

    summaries = []
    for (run_id, seed, load_level, variant), rows in sorted(grouped.items()):
        total_generated = sum(_to_int(row.get("generated_tasks")) for row in rows)
        total_completed = sum(_to_int(row.get("completed_tasks")) for row in rows)
        total_attempted = sum(_to_int(row.get("attempted_interactions")) for row in rows)
        total_successful = sum(_to_int(row.get("successful_cooperations")) for row in rows)
        total_failed = sum(_to_int(row.get("failed_interactions")) for row in rows)
        total_blocked = sum(_to_int(row.get("blocked_privacy_interactions")) for row in rows)
        energy = sum(_to_float(row.get("energy_consumed")) for row in rows)
        bandwidth = sum(_to_float(row.get("bandwidth_consumed")) for row in rows)
        compute = sum(_to_float(row.get("compute_consumed")) for row in rows)
        summaries.append(
            {
                "run_id": _to_int(run_id),
                "seed": _to_int(seed),
                "load_level": load_level,
                "ablation_variant": variant,
                "cooperation_rate": _safe_ratio(total_successful, total_attempted),
                "task_completion_ratio": _safe_ratio(total_completed, total_generated),
                "resource_efficiency": _resource_efficiency(
                    total_completed,
                    total_generated,
                    energy,
                    bandwidth,
                    compute,
                ),
                "reliability_score": _safe_ratio(
                    total_successful,
                    total_successful + total_failed + total_blocked,
                ),
                "fairness_index": _mean_column(rows, "fairness_index"),
                "trust_stability_index": _mean_column(rows, "trust_stability_index"),
                "privacy_exposure_risk": _mean_column(rows, "privacy_exposure_risk"),
                "candidate_privacy_risk": _mean_column(rows, "candidate_privacy_risk"),
                "accepted_privacy_exposure": _mean_column(rows, "accepted_privacy_exposure"),
                "throughput_mbps": _mean_column(rows, "throughput_mbps"),
                "packet_delivery_ratio": _mean_column(rows, "packet_delivery_ratio"),
                "e2e_delay_ms": _mean_nonzero_column(rows, "e2e_delay_ms"),
                "communication_overhead": _mean_column(rows, "communication_overhead"),
                "avg_total_utility": _mean_column(rows, "avg_total_utility"),
                "active_nodes": _to_int(rows[-1].get("active_nodes")) if rows else 0,
            }
        )
    return summaries


def aggregate_ablation_summary(run_summary_df: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate ablation run summaries by load level and variant."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in run_summary_df:
        grouped[(str(row["load_level"]), str(row["ablation_variant"]))].append(row)

    rows_out = []
    for (load_level, variant), rows in sorted(grouped.items()):
        for metric in SUMMARY_METRICS:
            values = [_to_float(row.get(metric)) for row in rows]
            count = len(values)
            avg = sum(values) / count if count else 0.0
            std = _sample_std(values)
            ci95 = 1.96 * std / math.sqrt(count) if count else 0.0
            ci_low = avg - ci95
            ci_high = avg + ci95
            if metric in BOUNDED_METRICS:
                ci_low = _clip01(ci_low)
                ci_high = _clip01(ci_high)
            rows_out.append(
                {
                    "load_level": load_level,
                    "ablation_variant": variant,
                    "metric": metric,
                    "mean": avg,
                    "std": std,
                    "count": count,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                }
            )
    return rows_out


def create_ablation_wide_summary(summary_long_df: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create a compact wide ablation summary table."""
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in summary_long_df:
        key = (str(row["load_level"]), str(row["ablation_variant"]))
        wide = grouped.setdefault(
            key,
            {"load_level": key[0], "ablation_variant": key[1]},
        )
        metric = str(row["metric"])
        if f"{metric}_mean" in ABLATION_WIDE_COLUMNS:
            wide[f"{metric}_mean"] = row["mean"]
        if f"{metric}_std" in ABLATION_WIDE_COLUMNS:
            wide[f"{metric}_std"] = row["std"]
    return [
        {column: row.get(column, 0.0) for column in ABLATION_WIDE_COLUMNS}
        for row in grouped.values()
    ]


def aggregate_ablation_outputs(input_dir: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Write combined and aggregated ablation output CSVs."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    steps = load_step_csvs(input_dir)
    interactions = load_interaction_csvs(input_dir)
    run_summary = aggregate_ablation_by_run(steps)
    summary = aggregate_ablation_summary(run_summary)
    summary_wide = create_ablation_wide_summary(summary)
    paths = {
        "combined_steps": output / "combined_steps.csv",
        "combined_interactions": output / "combined_interactions.csv",
        "ablation_run_summary": output / "ablation_run_summary.csv",
        "ablation_summary": output / "ablation_summary.csv",
        "ablation_summary_wide": output / "ablation_summary_wide.csv",
    }
    write_csv(paths["combined_steps"], steps, STEP_METRIC_COLUMNS)
    write_csv(paths["combined_interactions"], interactions, INTERACTION_COLUMNS)
    write_csv(paths["ablation_run_summary"], run_summary, ABLATION_RUN_SUMMARY_COLUMNS)
    write_csv(paths["ablation_summary"], summary, ABLATION_SUMMARY_COLUMNS)
    write_csv(paths["ablation_summary_wide"], summary_wide, ABLATION_WIDE_COLUMNS)
    return paths


def main() -> None:
    """CLI entry point for aggregating existing outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs")
    parser.add_argument("--output-dir", default="outputs/aggregated")
    args = parser.parse_args()
    paths = aggregate_outputs(args.input_dir, args.output_dir)
    for name, path in paths.items():
        print(f"{name}={path}")


def _load_named_csvs(input_dir: str | Path, filename: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = Path(input_dir)
    for path in sorted(root.rglob(filename)):
        if "aggregated" in path.parts:
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        return 0.0
    return min(max(float(numerator) / float(denominator), 0.0), 1.0)


def _resource_efficiency(
    completed_tasks: int,
    generated_tasks: int,
    total_energy: float,
    total_bandwidth: float,
    total_compute: float,
) -> float:
    completion = _safe_ratio(completed_tasks, generated_tasks)
    weighted_cost = max(
        0.4 * total_energy + 0.3 * total_bandwidth + 0.3 * total_compute,
        0.0,
    )
    normalized_resource_cost = _clip01(weighted_cost / max(float(generated_tasks), 1.0))
    return _clip01(completion * (1.0 - normalized_resource_cost))


def _clip01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_int(value: object) -> int:
    return int(_to_float(value))


def _mean_column(rows: list[dict[str, Any]], column: str) -> float:
    values = [_to_float(row.get(column)) for row in rows]
    return sum(values) / len(values) if values else 0.0


def _mean_nonzero_column(rows: list[dict[str, Any]], column: str) -> float:
    values = [_to_float(row.get(column)) for row in rows if _to_float(row.get(column)) > 0.0]
    return sum(values) / len(values) if values else 0.0


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = sum(values) / len(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


if __name__ == "__main__":
    main()
