"""One-at-a-time sensitivity analysis over every tunable coefficient.

Answers the reviewer objection that the framework carries 20+ untuned
coefficients with no evidence about which of them matter. Each parameter in
asiot.parameters.PARAMETERS is perturbed to -50%, -25%, +25% and +50% of its
default while every other parameter is held at its default, and the resulting
cooperation rate, task completion ratio and total utility are measured.

Protocol: seeds are 1000 + run_id, identical across every configuration, so a
parameter's effect is measured against the same worlds the baseline saw. The
baseline (all defaults) is run once and reused for every comparison, and the
difference is reported as a paired difference over shared seeds -- an unpaired
comparison would drown these effects in between-seed variance.

Parameters whose value must remain a probability are clipped to [0, 1]; when a
perturbation is clipped the actual value applied is recorded in the output, so
a "+50%" row that was truncated is visible rather than silently misleading.
The proposed-policy registry sweep retains the Nitti-only fallback as an
explicit non-applicability control; use the second command below to measure
that assumption on the Nitti baseline itself.

Usage:
    python scripts/run_sensitivity.py --runs 20 --steps 500 --workers 10
    python scripts/run_sensitivity.py --parameters nitti_empty_kij_fallback \
        --values 0.0,0.3,0.5,0.7 --model nitti_subjective_trust --runs 20
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asiot.config import load_config  # noqa: E402
from asiot.environment import ASIoTEnvironment  # noqa: E402
from asiot.parameters import PARAMETERS, apply_perturbation, by_name  # noqa: E402

METRICS = ("cooperation_rate", "task_completion_ratio", "avg_total_utility")
PERTURBATIONS = (-0.50, -0.25, 0.25, 0.50)


def one_run(job):
    """Run one (parameter, value, seed) cell and return mean per-step metrics."""
    field, applied, run_id, steps, load_level, model, config_path = job
    base = load_config(config_path)
    seed = 1000 + run_id
    overrides = {"steps": steps, "load_level": load_level, "random_seed": seed}
    if field is not None:
        # Convex-combination groups are renormalised so the configuration stays
        # valid; see asiot.parameters.apply_perturbation.
        overrides.update(apply_perturbation(base, by_name(field), applied))
    config = replace(base, **overrides)
    env = ASIoTEnvironment(config, seed=seed, baseline_name=model, run_id=run_id)
    rows = env.run(steps)["steps"]
    out = {m: statistics.mean(float(r[m]) for r in rows if m in r) for m in METRICS}
    return field, applied, run_id, out


def paired_difference(treatment: dict[int, float], baseline: dict[int, float]):
    """Mean paired difference, t interval/test, and paired effect size."""
    shared = sorted(set(treatment) & set(baseline))
    diffs = [treatment[r] - baseline[r] for r in shared]
    if len(diffs) < 2:
        return {
            "n": len(diffs), "mean": float("nan"), "low": float("nan"),
            "high": float("nan"), "p": float("nan"), "d_z": float("nan"),
        }
    mean = statistics.mean(diffs)
    standard_deviation = statistics.stdev(diffs)
    if math.isclose(standard_deviation, 0.0, abs_tol=1e-15):
        p_value = 1.0 if math.isclose(mean, 0.0, abs_tol=1e-15) else 0.0
        effect = 0.0 if p_value == 1.0 else math.copysign(float("inf"), mean)
        half = 0.0
    else:
        standard_error = standard_deviation / math.sqrt(len(diffs))
        t_statistic = mean / standard_error
        p_value = float(2.0 * stats.t.sf(abs(t_statistic), df=len(diffs) - 1))
        half = float(stats.t.ppf(0.975, df=len(diffs) - 1) * standard_error)
        effect = mean / standard_deviation
    return {
        "n": len(diffs), "mean": mean, "low": mean - half,
        "high": mean + half, "p": p_value, "d_z": effect,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm family-wise adjusted p-values in original row order."""
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(order) - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR adjusted q-values in original row order."""
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 1.0
    for reverse_rank in range(len(order) - 1, -1, -1):
        index = order[reverse_rank]
        rank = reverse_rank + 1
        candidate = min(1.0, p_values[index] * len(order) / rank)
        running = min(running, candidate)
        adjusted[index] = running
    return adjusted


def build_jobs(args, parameters):
    """Enumerate (parameter, value) cells, recording clipped values honestly."""
    cells = []
    if args.values:
        values = [float(v) for v in args.values.split(",")]
        for field in parameters:
            for value in values:
                cells.append((field, value, None))
        return cells
    for field in parameters:
        parameter = by_name(field)
        default = float(getattr(load_config(args.config), field))
        for delta in PERTURBATIONS:
            raw = default * (1.0 + delta)
            applied = min(1.0, max(0.0, raw)) if parameter.unit_interval else raw
            cells.append((field, applied, delta))
    return cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--load-level", default="high")
    ap.add_argument("--model", default="proposed")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--parameters", default=None,
                    help="Comma-separated subset; default is every registered parameter.")
    ap.add_argument("--values", default=None,
                    help="Explicit absolute values instead of +/-25%% and +/-50%%.")
    ap.add_argument("--output-dir", default="outputs/sensitivity")
    args = ap.parse_args()

    parameters = ([p.strip() for p in args.parameters.split(",")]
                  if args.parameters else [p.field for p in PARAMETERS])
    if args.model != "nitti_subjective_trust" and any(
        by_name(field).group == "external_baseline" for field in parameters
    ):
        print(
            "[sensitivity] external-baseline parameters are non-applicability "
            "controls for this model; run their documented model-specific sweep",
            flush=True,
        )
    cells = build_jobs(args, parameters)

    jobs = [(None, None, r, args.steps, args.load_level, args.model, args.config)
            for r in range(args.runs)]
    for field, applied, _delta in cells:
        jobs.extend((field, applied, r, args.steps, args.load_level, args.model,
                     args.config) for r in range(args.runs))

    print(f"[sensitivity] {len(parameters)} parameters, {len(cells)} cells, "
          f"{len(jobs)} runs, {args.workers} workers", flush=True)

    collected: dict[tuple, dict[int, dict[str, float]]] = {}
    done, t0 = 0, time.time()
    def collect(result):
        nonlocal done
        field, applied, run_id, metrics = result
        collected.setdefault((field, applied), {})[run_id] = metrics
        done += 1
        if done % 200 == 0 or done == len(jobs):
            rate = done / (time.time() - t0)
            print(f"  {done}/{len(jobs)} ({rate:.1f}/s, "
                  f"ETA {(len(jobs)-done)/rate/60:.1f} min)", flush=True)

    if args.workers == 1:
        for job in jobs:
            collect(one_run(job))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(one_run, j) for j in jobs]
            for future in as_completed(futures):
                collect(future.result())

    baseline = collected[(None, None)]
    rows = []
    for field, applied, delta in cells:
        parameter = by_name(field)
        default = float(getattr(load_config(args.config), field))
        treatment = collected[(field, applied)]
        row = {
            "parameter": field, "symbol": parameter.symbol, "group": parameter.group,
            "equation": parameter.equation, "default": f"{default:.6f}",
            "perturbation": "explicit" if delta is None else f"{delta:+.0%}",
            "value_applied": f"{applied:.6f}",
            "clipped": bool(delta is not None and not math.isclose(
                applied, default * (1.0 + delta), rel_tol=1e-9)),
            "simplex_renormalised": parameter.simplex or "",
        }
        for metric in METRICS:
            result = paired_difference(
                {r: v[metric] for r, v in treatment.items()},
                {r: v[metric] for r, v in baseline.items()},
            )
            row[f"{metric}_baseline"] = f"{statistics.mean(v[metric] for v in baseline.values()):.6f}"
            row[f"{metric}_delta"] = f"{result['mean']:.6f}"
            row[f"{metric}_ci_low"] = f"{result['low']:.6f}"
            row[f"{metric}_ci_high"] = f"{result['high']:.6f}"
            row[f"{metric}_n_pairs"] = result["n"]
            row[f"{metric}_cohens_dz"] = result["d_z"]
            row[f"{metric}_p_raw"] = result["p"]
            # Retained for compatibility: this is the unadjusted CI decision.
            row[f"{metric}_significant"] = bool(
                result["low"] > 0.0 or result["high"] < 0.0
            )
        rows.append(row)

    # Each metric is one prespecified 220-cell family (or the selected explicit
    # cell count for a targeted run). Corrections are calculated after every
    # row exists, so no result is selected on significance.
    for metric in METRICS:
        raw_p = [float(row[f"{metric}_p_raw"]) for row in rows]
        holm_p = holm_adjust(raw_p)
        bh_q = benjamini_hochberg(raw_p)
        for row, adjusted_p, adjusted_q in zip(rows, holm_p, bh_q):
            row[f"{metric}_p_holm"] = adjusted_p
            row[f"{metric}_q_bh"] = adjusted_q
            row[f"{metric}_holm_significant"] = adjusted_p < 0.05
            row[f"{metric}_bh_significant"] = adjusted_q < 0.05

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"sensitivity_{args.model}_{args.load_level}.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    influence = {}
    for row in rows:
        influence.setdefault(row["parameter"], []).append(abs(float(row["cooperation_rate_delta"])))
    ranked = sorted(influence.items(), key=lambda kv: -max(kv[1]))
    print("\n=== parameter influence on cooperation_rate (max |paired delta|) ===")
    for name, deltas in ranked:
        significant = any(r["cooperation_rate_holm_significant"] for r in rows
                          if r["parameter"] == name)
        marker = "" if significant else "   (no significant cell)"
        print(f"  {name:34s} {max(deltas):.4f}{marker}")
    inert = [n for n, _ in ranked
             if not any(r["cooperation_rate_holm_significant"] for r in rows if r["parameter"] == n)]
    print(f"\nHOLM-NONSIGNIFICANT ({len(inert)}/{len(ranked)}): no perturbation moved "
          f"cooperation after family-wise correction:\n  " + ", ".join(inert)
          if inert else "\nall parameters had at least one Holm-significant cell")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
