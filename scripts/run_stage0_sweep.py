"""Stage-0 honest rerun: parallel benign sweep + ablation sweep.

Runs the corrected simulator (post integrity fixes) across all baselines,
load levels, and seeds in parallel, then writes aggregates in the same
long format as outputs/final/all_loads/aggregated/summary_by_load_baseline.csv
so existing comparison/plotting tooling keeps working.

Protocol matches the release: seed = seed_start + run_id (seed_start=1000),
per-step metrics averaged over steps within a run, then mean/std/CI across runs.

Usage (M5, from repo root):
    python scripts/run_stage0_sweep.py --runs 100 --steps 500 --workers 10
    python scripts/run_stage0_sweep.py --mode ablation --runs 50 --steps 500 --workers 10
    # quick check first:
    python scripts/run_stage0_sweep.py --runs 5 --steps 100 --workers 8
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asiot.ablation import ABLATION_VARIANTS  # noqa: E402
from asiot.config import load_config  # noqa: E402
from asiot.environment import ASIoTEnvironment  # noqa: E402
from asiot.metrics import STEP_METRIC_COLUMNS  # noqa: E402

BASELINES = [
    "non_agentic_static",
    "greedy_utility",
    "standard_marl_no_social",
    "trust_unaware",
    "honesty_based_social",
    "game_theoretic_social",
    "nitti_subjective_trust",
    "proposed",
]
LOADS = ["low", "medium", "high", "extreme"]

# Excluded from reporting until real message accounting exists (Stage-0
# integrity decision): communication_overhead is hardcoded 0.0 in metrics.py
# and network figures previously did not derive from the pipeline.
EXCLUDED_METRICS = {"communication_overhead"}
_ID_COLUMNS = {"run_id", "seed", "load_level", "baseline_name", "ablation_variant", "time_step"}
NUMERIC_METRICS = [m for m in STEP_METRIC_COLUMNS
                   if m not in EXCLUDED_METRICS and m not in _ID_COLUMNS]


def one_run(args_tuple):
    mode, name, load, run_id, seed_start, steps, config_path = args_tuple
    base = load_config(config_path)
    seed = seed_start + run_id
    cfg = replace(base, steps=steps, load_level=load, random_seed=seed,
                  output_dir=Path("/tmp/asiot_stage0_scratch"))
    if mode == "baseline":
        env = ASIoTEnvironment(cfg, seed=seed, baseline_name=name, run_id=run_id)
    else:
        env = ASIoTEnvironment(cfg, seed=seed, baseline_name="proposed",
                               run_id=run_id, ablation_variant=name)
    res = env.run(steps)
    rows = res["steps"]
    out = {}
    for metric in NUMERIC_METRICS:
        vals = [float(r[metric]) for r in rows if metric in r]
        if vals:
            out[metric] = statistics.mean(vals)
    return name, load, run_id, seed, out


def aggregate(per_run, key_field):
    rows = []
    keys = sorted({(n, l) for (n, l, *_id) in [(r[0], r[1], r[2]) for r in per_run]})
    by_key = {}
    for name, load, run_id, seed, metrics in per_run:
        by_key.setdefault((name, load), []).append(metrics)
    for (name, load), runs in sorted(by_key.items()):
        for metric in NUMERIC_METRICS:
            vals = [r[metric] for r in runs if metric in r]
            if not vals:
                continue
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            half = 1.96 * std / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
            rows.append({
                "load_level": load, key_field: name, "metric": metric,
                "mean": f"{mean:.6f}", "std": f"{std:.6f}", "count": len(vals),
                "ci95_low": f"{mean - half:.6f}", "ci95_high": f"{mean + half:.6f}",
            })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "ablation"], default="baseline")
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed-start", type=int, default=1000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--output-dir", default="outputs/stage0")
    args = p.parse_args()

    names = BASELINES if args.mode == "baseline" else list(ABLATION_VARIANTS)
    key_field = "baseline_name" if args.mode == "baseline" else "ablation_variant"
    jobs = [(args.mode, n, l, r, args.seed_start, args.steps, args.config)
            for n in names for l in LOADS for r in range(args.runs)]
    print(f"[stage0] {args.mode} sweep: {len(names)} models x {len(LOADS)} loads x "
          f"{args.runs} runs = {len(jobs)} runs, {args.workers} workers")

    out_dir = Path(args.output_dir) / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    per_run, done, t0 = [], 0, time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(one_run, j) for j in jobs]
        for fut in as_completed(futures):
            per_run.append(fut.result())
            done += 1
            if done % 50 == 0 or done == len(jobs):
                rate = done / (time.time() - t0)
                eta = (len(jobs) - done) / rate if rate else 0
                print(f"  {done}/{len(jobs)}  ({rate:.1f} runs/s, ETA {eta/60:.1f} min)")

    # per-run table (audit trail)
    run_path = out_dir / "run_summary.csv"
    with open(run_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([key_field, "load_level", "run_id", "seed"] + NUMERIC_METRICS)
        for name, load, run_id, seed, m in sorted(per_run, key=lambda r: (r[0], r[1], r[2])):
            w.writerow([name, load, run_id, seed] + [f"{m.get(k, float('nan')):.6f}" for k in NUMERIC_METRICS])

    # aggregated long table (release-compatible format)
    agg = aggregate(per_run, key_field)
    agg_path = out_dir / ("summary_by_load_baseline.csv" if args.mode == "baseline"
                          else "ablation_summary.csv")
    with open(agg_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["load_level", key_field, "metric", "mean",
                                          "std", "count", "ci95_low", "ci95_high"])
        w.writeheader()
        w.writerows(agg)
    print(f"[stage0] wrote {run_path} and {agg_path} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
