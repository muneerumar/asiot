"""Measure simulator scalability with fresh-process, sequential repetitions.

Runtime measurements are deliberately sequential: parallel workers would
contend for CPU and make the timing table unsuitable for publication.  Each
cell runs in a fresh Python process so peak resident memory is attributable to
one node-count/seed cell and cumulative allocator state cannot leak between
measurements.  Interpreter startup is excluded from the reported timers.

The benchmark measures the implemented analytical simulator, not packet-level
network performance.  Results are hardware- and software-specific and should
be used to describe computational overhead and scaling, not deployment delay.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asiot.baselines import BASELINE_REGISTRY  # noqa: E402
from asiot.config import load_config  # noqa: E402
from asiot.environment import ASIoTEnvironment  # noqa: E402


RAW_COLUMNS = (
    "node_count",
    "run_id",
    "seed",
    "steps",
    "load_level",
    "baseline_name",
    "initialization_wall_seconds",
    "simulation_wall_seconds",
    "simulation_cpu_seconds",
    "wall_ms_per_step",
    "cpu_ms_per_step",
    "peak_rss_mib",
    "initial_mean_degree",
    "final_mean_degree",
    "generated_tasks_per_step",
    "attempted_interactions_per_step",
    "cooperation_rate",
    "task_completion_ratio",
    "final_active_nodes",
    "python_version",
    "platform",
)


def _peak_rss_mib() -> float:
    """Return process peak resident memory in MiB on macOS or Linux."""
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return raw / (1024.0 * 1024.0)
    return raw / 1024.0


def run_single_cell(args: argparse.Namespace) -> dict[str, object]:
    """Run and time one benchmark cell inside its fresh child process."""
    config = replace(
        load_config(args.config),
        node_count=args.single_node_count,
        steps=args.steps,
        load_level=args.load_level,
        random_seed=args.single_seed,
    )

    init_start = time.perf_counter()
    environment = ASIoTEnvironment(
        config,
        seed=args.single_seed,
        baseline_name=args.baseline,
        run_id=args.single_run_id,
    )
    initialization_wall = time.perf_counter() - init_start
    initial_mean_degree = environment.graph.average_degree()

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    result = environment.run(args.steps)
    simulation_cpu = time.process_time() - cpu_start
    simulation_wall = time.perf_counter() - wall_start
    summary = result["final_summary"]
    step_rows = result["steps"]
    attempted_interactions = sum(
        int(row["attempted_interactions"]) for row in step_rows
    )

    steps = max(1, int(args.steps))
    return {
        "node_count": int(args.single_node_count),
        "run_id": int(args.single_run_id),
        "seed": int(args.single_seed),
        "steps": int(args.steps),
        "load_level": args.load_level,
        "baseline_name": args.baseline,
        "initialization_wall_seconds": initialization_wall,
        "simulation_wall_seconds": simulation_wall,
        "simulation_cpu_seconds": simulation_cpu,
        "wall_ms_per_step": 1000.0 * simulation_wall / steps,
        "cpu_ms_per_step": 1000.0 * simulation_cpu / steps,
        "peak_rss_mib": _peak_rss_mib(),
        "initial_mean_degree": initial_mean_degree,
        "final_mean_degree": environment.graph.average_degree(),
        "generated_tasks_per_step": float(summary["total_tasks"]) / steps,
        "attempted_interactions_per_step": attempted_interactions / steps,
        "cooperation_rate": float(summary["cooperation_rate"]),
        "task_completion_ratio": (
            float(summary["completed_tasks"]) / float(summary["total_tasks"])
            if summary["total_tasks"]
            else 0.0
        ),
        "final_active_nodes": int(summary["active_nodes"]),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def _mean_ci(series: pd.Series) -> tuple[float, float, float, float]:
    """Return mean, sample SD, and two-sided 95% t interval."""
    values = series.astype(float).dropna()
    mean = float(values.mean())
    if len(values) < 2:
        return mean, float("nan"), float("nan"), float("nan")
    standard_deviation = float(values.std(ddof=1))
    half_width = float(
        stats.t.ppf(0.975, df=len(values) - 1)
        * standard_deviation
        / math.sqrt(len(values))
    )
    return mean, standard_deviation, mean - half_width, mean + half_width


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate timing, memory, workload, and outcome by node count."""
    metrics = (
        "wall_ms_per_step",
        "cpu_ms_per_step",
        "peak_rss_mib",
        "initialization_wall_seconds",
        "final_mean_degree",
        "generated_tasks_per_step",
        "attempted_interactions_per_step",
        "cooperation_rate",
        "task_completion_ratio",
        "final_active_nodes",
    )
    rows: list[dict[str, object]] = []
    for node_count, cell in raw.groupby("node_count", sort=True):
        row: dict[str, object] = {
            "node_count": int(node_count),
            "runs": int(len(cell)),
            "steps_per_run": int(cell["steps"].iloc[0]),
            "load_level": str(cell["load_level"].iloc[0]),
            "baseline_name": str(cell["baseline_name"].iloc[0]),
        }
        for metric in metrics:
            mean, standard_deviation, ci_low, ci_high = _mean_ci(cell[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = standard_deviation
            row[f"{metric}_ci95_low"] = ci_low
            row[f"{metric}_ci95_high"] = ci_high
            row[f"{metric}_median"] = float(cell[metric].median())
        rows.append(row)
    return pd.DataFrame(rows)


def scaling_fit(summary: pd.DataFrame) -> dict[str, float]:
    """Fit wall time per step = constant * node_count**exponent in log space."""
    if len(summary) < 2:
        return {"exponent": float("nan"), "r_squared": float("nan"), "p_value": float("nan")}
    fit = stats.linregress(
        summary["node_count"].astype(float).map(math.log),
        summary["wall_ms_per_step_mean"].astype(float).map(math.log),
    )
    return {
        "exponent": float(fit.slope),
        "r_squared": float(fit.rvalue**2),
        "p_value": float(fit.pvalue),
    }


def write_report(raw: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    """Write a concise, manuscript-ready scalability interpretation."""
    fit = scaling_fit(summary)
    first = summary.sort_values("node_count").iloc[0]
    last = summary.sort_values("node_count").iloc[-1]
    runtime_ratio = float(last["wall_ms_per_step_mean"] / first["wall_ms_per_step_mean"])
    memory_ratio = float(last["peak_rss_mib_mean"] / first["peak_rss_mib_mean"])
    lines = [
        "# Scalability and computational-overhead report",
        "",
        f"Protocol: {int(summary['runs'].iloc[0])} fresh-process repetitions per node count, "
        f"{int(summary['steps_per_run'].iloc[0])} steps per run, "
        f"`{summary['load_level'].iloc[0]}` load, `{summary['baseline_name'].iloc[0]}` policy.",
        "Runs were sequential to avoid CPU contention. Interpreter startup is excluded",
        "from timing; peak RSS includes the Python runtime and imported libraries.",
        "",
        "| Nodes | Wall ms/step (95% CI) | CPU ms/step | Peak RSS MiB | Cooperation | Active nodes |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values("node_count").itertuples(index=False):
        lines.append(
            f"| {row.node_count} | {row.wall_ms_per_step_mean:.3f} "
            f"[{row.wall_ms_per_step_ci95_low:.3f}, {row.wall_ms_per_step_ci95_high:.3f}] | "
            f"{row.cpu_ms_per_step_mean:.3f} | {row.peak_rss_mib_mean:.1f} | "
            f"{row.cooperation_rate_mean:.4f} | {row.final_active_nodes_mean:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Scaling interpretation",
            "",
            f"Across {int(first['node_count'])}–{int(last['node_count'])} nodes, mean wall time "
            f"per step increased by {runtime_ratio:.2f}× and peak RSS by {memory_ratio:.2f}×.",
            f"A log-log fit gives an empirical runtime exponent of {fit['exponent']:.3f} "
            f"(R²={fit['r_squared']:.3f}, p={fit['p_value']:.4g}).",
            "",
            "The code-level bound is dominated by the all-node reputation refresh:",
            "each step considers reports for every target/reporter pair, O(N²). Dynamic",
            "edge maintenance is O(Nd) for bounded degree d, and partner scoring is",
            "O(Td) for T generated tasks. The empirical exponent is a finite-range",
            "measurement, not a replacement for that asymptotic bound.",
            "",
            "## Scope boundary",
            "",
            "These are simulator compute measurements. They do not measure packet delay,",
            "broker traffic, radio energy, or real-device latency.",
            "",
            f"Raw benchmark rows: {len(raw)}.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _child_command(args: argparse.Namespace, node_count: int, run_id: int, seed: int) -> list[str]:
    """Build the exact fresh-process command for one timing cell."""
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-job",
        "--single-node-count",
        str(node_count),
        "--single-run-id",
        str(run_id),
        "--single-seed",
        str(seed),
        "--steps",
        str(args.steps),
        "--load-level",
        args.load_level,
        "--baseline",
        args.baseline,
        "--config",
        args.config,
    ]


def run_sweep(args: argparse.Namespace) -> None:
    """Run all sequential cells and write raw, summary, and report artifacts."""
    node_counts = [int(value.strip()) for value in args.node_counts.split(",") if value.strip()]
    if len(node_counts) < 2 or any(value < 2 for value in node_counts):
        raise ValueError("--node-counts requires at least two integer values >= 2")
    rows: list[dict[str, object]] = []
    total = len(node_counts) * args.runs
    completed = 0
    started = time.perf_counter()
    for node_count in node_counts:
        for run_id in range(args.runs):
            seed = args.seed_start + run_id
            completed_process = subprocess.run(
                _child_command(args, node_count, run_id, seed),
                check=True,
                capture_output=True,
                text=True,
            )
            rows.append(json.loads(completed_process.stdout.strip()))
            completed += 1
            elapsed = time.perf_counter() - started
            rate = completed / elapsed
            eta = (total - completed) / rate if rate else float("nan")
            print(
                f"[scalability] {completed}/{total} node_count={node_count} "
                f"run={run_id} ETA={eta / 60.0:.1f} min",
                flush=True,
            )

    raw = pd.DataFrame(rows, columns=RAW_COLUMNS)
    aggregate = summarize(raw)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_dir / "run_summary.csv", index=False)
    aggregate.to_csv(output_dir / "scalability_summary.csv", index=False)
    write_report(raw, aggregate, output_dir / "SCALABILITY.md")
    fit = scaling_fit(aggregate)
    print(f"runtime_exponent={fit['exponent']:.6f}")
    print(f"runtime_r_squared={fit['r_squared']:.6f}")
    print(f"output_dir={output_dir}")


def build_parser() -> argparse.ArgumentParser:
    """Create the public and internal benchmark CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-counts", default="25,50,100,200,400")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=7000)
    parser.add_argument("--load-level", choices=("low", "medium", "high", "extreme"), default="high")
    parser.add_argument("--baseline", choices=sorted(BASELINE_REGISTRY), default="proposed")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output-dir", default="supplementary_results/scalability")
    parser.add_argument("--single-job", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--single-node-count", type=int, default=50, help=argparse.SUPPRESS)
    parser.add_argument("--single-run-id", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--single-seed", type=int, default=7000, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    """Dispatch a single child cell or the full sequential sweep."""
    args = build_parser().parse_args()
    if args.runs < 1 or args.steps < 1:
        raise ValueError("--runs and --steps must be positive")
    if args.single_job:
        print(json.dumps(run_single_cell(args), sort_keys=True))
    else:
        run_sweep(args)


if __name__ == "__main__":
    main()
