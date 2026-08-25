"""Full MARL protocol: 3 seeds x 2 observation variants, run in parallel.

Launches one `run_marl_training.py` process per (variant, seed) cell. Each cell
is independent -- separate agent, separate seed band, separate output and
checkpoint directories -- so they parallelise cleanly. Torch is pinned to one
thread per process because the networks are small MLPs: without the pin, six
processes oversubscribe the cores and each runs slower than it would alone.

Seed bands are disjoint by construction:
    training          seed_start + episode        (3000+, 6000+, 9000+)
    periodic eval     eval_seed_start             (900000+)
    final eval        eval_seed_start + 500000    (1400000+)
so no reported number comes from a seed used for training or model selection.

Usage:
    python scripts/run_marl_full.py --episodes 3000 --steps 500
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("neural_marl_social", "neural_marl_no_social")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=3000)
    ap.add_argument("--seed-stride", type=int, default=3000)
    ap.add_argument("--load-level", default="high")
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--checkpoint-interval", type=int, default=250)
    ap.add_argument("--eval-episodes", type=int, default=3)
    ap.add_argument("--epsilon-decay-fraction", type=float, default=0.65)
    ap.add_argument("--output-root", default="outputs/marl_full")
    ap.add_argument("--log-root", default="outputs/marl_full/logs")
    args = ap.parse_args()

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    # One thread per process: six small-MLP trainers otherwise fight for cores.
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCH_NUM_THREADS"):
        env[key] = "1"

    log_root = Path(args.log_root)
    log_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for variant in VARIANTS:
        for index in range(args.seeds):
            seed_start = args.seed_base + index * args.seed_stride
            cell = f"{variant}_seed{seed_start}"
            output_dir = Path(args.output_root) / cell
            command = [
                sys.executable, str(ROOT / "scripts" / "run_marl_training.py"),
                "--episodes", str(args.episodes),
                "--steps", str(args.steps),
                "--load-level", args.load_level,
                "--policy-type", variant,
                "--seed-start", str(seed_start),
                "--eval-interval", str(args.eval_interval),
                "--checkpoint-interval", str(args.checkpoint_interval),
                "--eval-episodes", str(args.eval_episodes),
                "--epsilon-decay-fraction", str(args.epsilon_decay_fraction),
                "--eval-runs", "10",
                "--output-dir", str(output_dir),
                "--checkpoint-dir", str(Path(args.output_root) / "checkpoints"),
            ]
            handle = open(log_root / f"{cell}.log", "w")
            process = subprocess.Popen(command, cwd=ROOT, env=env,
                                       stdout=handle, stderr=subprocess.STDOUT)
            jobs.append((cell, process, handle))
            print(f"launched {cell} (pid {process.pid})", flush=True)

    start = time.time()
    failures = []
    for cell, process, handle in jobs:
        code = process.wait()
        handle.close()
        status = "ok" if code == 0 else f"FAILED (exit {code})"
        if code != 0:
            failures.append(cell)
        print(f"[{(time.time()-start)/3600:.2f} h] {cell}: {status}", flush=True)

    print(f"\nall cells finished in {(time.time()-start)/3600:.2f} h")
    if failures:
        print("FAILED CELLS: " + ", ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    main()
