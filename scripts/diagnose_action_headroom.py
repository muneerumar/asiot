"""How much reward headroom does the partner-selection action actually have?

The neural policy chooses among candidates that the heuristic has ALREADY
scored and ranked. If every candidate in the offered set yields nearly the same
multi-objective utility, the action is close to reward-irrelevant: no amount of
training can separate policies, because there is almost nothing to separate.
That is a property of the problem formulation, measurable without training.

Three candidate sets are compared at identical decision points:

  top8        the shipped formulation (top_k_candidates = 8)
  topK        top-K with K raised substantially, reaching further down the
              heuristic's ranking
  feasible    every privacy-feasible neighbour, i.e. the heuristic's ranking
              discarded entirely -- the widest set the environment permits

For each: mean valid actions, best-minus-worst utility spread, and the gap an
ORACLE (always the best candidate) would enjoy over always picking the worst.
The oracle gap is the ceiling on what ANY policy, learned or otherwise, could
gain by choosing well within that set.

Usage:
    python scripts/diagnose_action_headroom.py --steps 200 --load-level high
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asiot.config import load_config  # noqa: E402
from asiot.environment import ASIoTEnvironment  # noqa: E402
from asiot.marl.observations import build_top_k_observation  # noqa: E402


def _collect(steps: int, load_level: str, seed: int, top_k_small: int, top_k_large: int,
             min_neighbors: int | None = None, max_neighbors: int | None = None):
    """Walk identical decision points once, recording each candidate set.

    ``min_neighbors``/``max_neighbors`` override the configured graph degree.
    That is a THROWAWAY diagnostic only: changing degree changes the simulated
    system and would invalidate every existing sweep, so it must never be used
    to produce a reported result. It answers one question -- whether topology,
    rather than K, is what bounds the decision.
    """
    config = replace(load_config("config/default.yaml"), load_level=load_level,
                     random_seed=seed)
    if min_neighbors is not None:
        config = replace(config, min_neighbors=int(min_neighbors))
    if max_neighbors is not None:
        config = replace(config, max_neighbors=int(max_neighbors))
    env = ASIoTEnvironment(config, seed=seed, baseline_name="proposed", run_id=0)
    sets: dict[str, list[list[float]]] = {
        f"top{top_k_small}": [], f"top{top_k_large}": [], "feasible": [],
    }
    neighbour_counts: list[int] = []

    for _ in range(steps):
        env.graph.update_edges([n.state for n in env.nodes.values()], env.time_step)
        for task in env.generate_tasks(env.time_step):
            requester = env.nodes[task.requester_id]
            neighbours = requester.observe_neighbors(env.graph)
            scores = env.policy.score_neighbors(requester, neighbours, task,
                                                env.graph, config, env.nodes)
            if not scores:
                continue
            neighbour_counts.append(len(scores))

            for label, k in ((f"top{top_k_small}", top_k_small),
                             (f"top{top_k_large}", top_k_large)):
                _obs, _mask, candidates = build_top_k_observation(
                    requester, task, scores, env.nodes, env.graph, config,
                    top_k_candidates=k, include_social_features=True,
                )
                utilities = [float(scores[c]["total_utility"]) for c in candidates]
                if len(utilities) >= 2:
                    sets[label].append(utilities)

            # Privacy-feasible set: the heuristic's ranking discarded entirely.
            feasible = [float(s["total_utility"]) for s in scores.values()
                        if int(s["privacy_allowed"]) == 1]
            if len(feasible) >= 2:
                sets["feasible"].append(feasible)
        env.time_step += 1
    return sets, neighbour_counts


def _summarize(label: str, candidate_utilities: list[list[float]]) -> dict[str, float]:
    sizes = [len(u) for u in candidate_utilities]
    spreads = [max(u) - min(u) for u in candidate_utilities]
    rel = [(max(u) - min(u)) / statistics.mean(u) for u in candidate_utilities]
    best = statistics.mean(max(u) for u in candidate_utilities)
    worst = statistics.mean(min(u) for u in candidate_utilities)
    return {
        "candidate_set": label,
        "decision_points": len(candidate_utilities),
        "mean_valid_actions": statistics.mean(sizes),
        "max_valid_actions": max(sizes),
        "spread_mean": statistics.mean(spreads),
        "spread_median": statistics.median(spreads),
        "spread_max": max(spreads),
        "spread_pct_of_mean_utility": 100.0 * statistics.mean(rel),
        "mean_best_utility": best,
        "mean_worst_utility": worst,
        "oracle_vs_worst_gap_pct": 100.0 * (best - worst) / worst,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--load-level", default="high")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--top-k-small", type=int, default=8)
    ap.add_argument("--top-k-large", type=int, default=32)
    ap.add_argument("--min-neighbors", type=int, default=None,
                    help="THROWAWAY degree override; never use for reported results.")
    ap.add_argument("--max-neighbors", type=int, default=None,
                    help="THROWAWAY degree override; never use for reported results.")
    ap.add_argument("--output", default="outputs/marl_diagnostics/action_headroom.csv")
    args = ap.parse_args()

    sets, neighbour_counts = _collect(args.steps, args.load_level, args.seed,
                                      args.top_k_small, args.top_k_large,
                                      args.min_neighbors, args.max_neighbors)
    rows = [_summarize(label, utilities) for label, utilities in sets.items() if utilities]

    print(f"load={args.load_level} seed={args.seed} steps={args.steps}")
    print(f"scored neighbours per decision: mean={statistics.mean(neighbour_counts):.2f} "
          f"max={max(neighbour_counts)}\n")
    for row in rows:
        print(f"--- {row['candidate_set']} ({row['decision_points']} decision points) ---")
        print(f"  valid actions      mean={row['mean_valid_actions']:.2f}  max={row['max_valid_actions']}")
        print(f"  utility spread     mean={row['spread_mean']:.4f}  median={row['spread_median']:.4f}  max={row['spread_max']:.4f}")
        print(f"                     = {row['spread_pct_of_mean_utility']:.2f}% of mean utility")
        print(f"  best={row['mean_best_utility']:.4f}  worst={row['mean_worst_utility']:.4f}")
        print(f"  ORACLE vs worst    {row['oracle_vs_worst_gap_pct']:.2f}%  <-- ceiling for ANY policy\n")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
