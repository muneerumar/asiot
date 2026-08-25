"""Replace corrected cells in an attack sweep and regenerate its aggregates.

This is intentionally key-based: a correction may contain only one attack or
fraction subset, and every matching base run is replaced rather than appended.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

RUN_KEY = (
    "model",
    "attack",
    "fraction",
    "defense",
    "load",
    "run_id",
    "adaptive_mu",
)
GROUP_KEY = (
    "model",
    "attack",
    "fraction",
    "defense",
    "load",
    "adaptive_mu",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base run_summary.csv")
    parser.add_argument("--correction", required=True, help="Corrected run_summary.csv")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    merge_attack_sweeps(args.base, args.correction, args.output_dir)


def merge_attack_sweeps(
    base_path: str | Path,
    correction_path: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Replace keyed base rows with corrections and write both summary files."""
    base_rows, base_fields = _read(base_path)
    correction_rows, correction_fields = _read(correction_path)
    if not correction_rows:
        raise ValueError("correction sweep is empty")

    correction_by_key = {_key(row, RUN_KEY): row for row in correction_rows}
    if len(correction_by_key) != len(correction_rows):
        raise ValueError("correction sweep contains duplicate run keys")
    merged_by_key = {_key(row, RUN_KEY): row for row in base_rows}
    missing = sorted(set(correction_by_key).difference(merged_by_key))
    if missing:
        raise ValueError(f"correction contains {len(missing)} run keys absent from base")
    merged_by_key.update(correction_by_key)

    fields = list(correction_fields)
    fields.extend(field for field in base_fields if field not in fields)
    rows = sorted(merged_by_key.values(), key=lambda row: _key(row, RUN_KEY))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_path = output / "run_summary.csv"
    with run_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "0.000000") for field in fields})

    metric_fields = [field for field in fields if field not in RUN_KEY]
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[_key(row, GROUP_KEY)].append(row)
    aggregate_path = output / "attack_summary.csv"
    summary_fields = list(GROUP_KEY) + [
        "metric",
        "mean",
        "std",
        "count",
        "ci95_low",
        "ci95_high",
    ]
    with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for group_key, group_rows in sorted(grouped.items()):
            identifiers = dict(zip(GROUP_KEY, group_key))
            for metric in metric_fields:
                values = [_finite_float(row.get(metric, "")) for row in group_rows]
                values = [value for value in values if value is not None]
                if not values:
                    continue
                mean = statistics.mean(values)
                std = statistics.stdev(values) if len(values) > 1 else 0.0
                half_width = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
                writer.writerow(
                    identifiers
                    | {
                        "metric": metric,
                        "mean": f"{mean:.6f}",
                        "std": f"{std:.6f}",
                        "count": len(values),
                        "ci95_low": f"{mean - half_width:.6f}",
                        "ci95_high": f"{mean + half_width:.6f}",
                    }
                )
    print(f"merged {len(correction_rows)} corrected runs into {len(rows)} total runs")
    print(run_path)
    print(aggregate_path)
    return run_path, aggregate_path


def _read(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or ())


def _key(row: dict[str, str], fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in fields)


def _finite_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


if __name__ == "__main__":
    main()
