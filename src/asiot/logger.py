"""Simulation logging helpers for raw and combined CSV outputs."""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any

from asiot.datatypes import InteractionResult
from asiot.metrics import STEP_METRIC_COLUMNS

INTERACTION_COLUMNS = (
    "run_id",
    "seed",
    "load_level",
    "baseline_name",
    "ablation_variant",
    "time_step",
    "task_id",
    "source_id",
    "target_id",
    "attempted",
    "success",
    "blocked_by_privacy",
    "failed_reason",
    "utility",
    "expected_success",
    "total_utility",
    "system_utility",
    "social_utility",
    "resource_utility",
    "privacy_utility",
    "fairness_utility",
    "incentive_utility",
    "privacy_risk",
    "candidate_privacy_risk",
    "accepted_privacy_exposure",
    "trust_before",
    "trust_after",
    "energy_cost",
    "bandwidth_cost",
    "compute_cost",
    "delay_ms",
)


class SimulationLogger:
    """Collect interaction and step logs in memory and write CSV files."""

    def __init__(self) -> None:
        self.interactions: list[dict[str, Any]] = []
        self.steps: list[dict[str, Any]] = []

    def log_interaction(self, result: InteractionResult) -> None:
        """Log one interaction result using stable interaction columns."""
        row = asdict(result)
        self.interactions.append(_ordered_row(row, INTERACTION_COLUMNS))

    def log_step(self, summary: dict[str, Any]) -> None:
        """Log one per-step summary using stable step columns."""
        self.steps.append(_ordered_row(summary, STEP_METRIC_COLUMNS))

    def get_interactions_dataframe(self) -> Any:
        """Return interactions as a pandas DataFrame when available, else rows."""
        try:
            import pandas as pd

            return pd.DataFrame(self.interactions)
        except ModuleNotFoundError:
            return list(self.interactions)

    def get_steps_dataframe(self) -> Any:
        """Return step summaries as a pandas DataFrame when available, else rows."""
        try:
            import pandas as pd

            return pd.DataFrame(self.steps)
        except ModuleNotFoundError:
            return list(self.steps)

    def save_raw_outputs(self, output_dir: str | Path) -> dict[str, Path]:
        """Write ``interactions.csv`` and ``steps.csv`` under ``output_dir/raw``."""
        raw_dir = Path(output_dir) / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        return self.save_combined_outputs(raw_dir, self.interactions, self.steps)

    def save_combined_outputs(
        self,
        output_dir: str | Path,
        interactions_rows: list[dict[str, Any]],
        steps_rows: list[dict[str, Any]],
    ) -> dict[str, Path]:
        """Write interaction and step rows directly under an output directory."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        interactions_path = out_dir / "interactions.csv"
        steps_path = out_dir / "steps.csv"
        write_csv(interactions_path, interactions_rows, INTERACTION_COLUMNS)
        write_csv(steps_path, steps_rows, STEP_METRIC_COLUMNS)
        return {"interactions": interactions_path, "steps": steps_path}


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: tuple[str, ...] | list[str] | None = None,
) -> None:
    """Write dictionaries to CSV with stable field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = _fieldnames_from_rows(rows)
    if not rows:
        path.write_text(",".join(fieldnames) + "\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(_ordered_row(row, fieldnames))


def _ordered_row(row: dict[str, Any], fieldnames: tuple[str, ...] | list[str]) -> dict[str, Any]:
    return {key: row.get(key, 0.0) for key in fieldnames}


def _fieldnames_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames
