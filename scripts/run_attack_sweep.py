"""Stage-2 adversarial sweep: attack type x attacker fraction x model.

Produces the security-evaluation tables and resilience curves. Attacker
identity is drawn from a seed-derived RNG, so at a given seed every compared
model faces the same attacker set (paired comparison).

Usage (M5, from repo root):
    # quick check (~3 min)
    python scripts/run_attack_sweep.py --runs 3 --steps 150 --workers 8

    # full sweep (~2-4 h with 10 workers; run overnight)
    python scripts/run_attack_sweep.py --runs 50 --steps 500 --workers 10

    # defense ablation only (feedback attacks, defense on vs off)
    python scripts/run_attack_sweep.py --mode defense --runs 50 --steps 500 --workers 10
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

from asiot.attacks import ALL_ATTACKS  # noqa: E402
from asiot.config import load_config  # noqa: E402
from asiot.environment import ASIoTEnvironment  # noqa: E402
from asiot.metrics import STEP_METRIC_COLUMNS  # noqa: E402
from asiot.social_cognition import (  # noqa: E402
    compute_adaptive_mu,
    compute_effective_trust,
)

# Fixed-mu fallback, mirroring the constant used in AgenticNode.compute_neighbor_scores
# when config.adaptive_mu_enabled is False (paper Eq. 38 default).
FIXED_MU = 0.65

MODELS = ["proposed", "trust_unaware", "greedy_utility",
          "standard_marl_no_social", "nitti_subjective_trust"]
FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4]
_ID = {"run_id", "seed", "load_level", "baseline_name", "ablation_variant", "time_step"}
METRICS = [m for m in STEP_METRIC_COLUMNS if m not in _ID and m != "communication_overhead"]


def decision_distortion(env) -> dict[str, float]:
    """Mean |T_eff(i, j) - T_direct(i, j)| over honest reporter/honest target pairs.

    This is the quantity adaptive mu targets: how far collective reputation
    (hearsay) pulls a node's decision input away from its own direct evidence.
    Effective trust is recomputed exactly as AgenticNode.compute_neighbor_scores
    computes it -- same mu path, same reputation source -- so the number
    describes the decisions the simulation actually made.

    Read-only over end-of-run state: it draws no random numbers and therefore
    cannot perturb determinism.

    Four values are returned.

    ``decision_distortion`` / ``decision_distortion_evidenced``
        Distortion under the mu rule this run actually used. The ``evidenced``
        variant restricts to pairs where the reporter has at least one direct
        interaction with the target; pairs with no evidence are identical under
        both rules by construction (adaptive mu reduces to base_mu at n = 0) and
        so only dilute any effect. Both are reported; neither is chosen after
        the fact.

    ``decision_distortion_counterfactual`` / ``..._counterfactual_evidenced``
        The same end-of-run state scored under the OTHER mu rule. This exists
        because adaptive mu satisfies mu_eff >= base_mu pointwise, and
        distortion is (1 - mu) * |T - R|, so a reduction is guaranteed by the
        definition for any fixed (T, R). Only the part of the observed
        reduction that exceeds this mechanical term reflects a real change in
        the trajectory. Reporting the distortion drop without this control
        would be reporting an identity as a finding.
    """
    adaptive = bool(getattr(env.config, "adaptive_mu_enabled", True))
    attackers = env.attacker_ids
    actual: list[float] = []
    actual_ev: list[float] = []
    counter: list[float] = []
    counter_ev: list[float] = []
    for node_id, node in env.nodes.items():
        if node_id in attackers:
            continue
        for target_id in env.graph.get_neighbors(node_id):
            if target_id == node_id or target_id in attackers:
                continue
            local_trust = node.state.social.trust.get(target_id, env.config.initial_trust)
            reputation = env.graph.get_node_reputation(target_id)
            evidence = float(node.state.social.interaction_i_to_j.get(target_id, 0.0))
            mu_adaptive = compute_adaptive_mu(local_trust, reputation, evidence)
            mu_used = mu_adaptive if adaptive else FIXED_MU
            mu_other = FIXED_MU if adaptive else mu_adaptive
            d_used = abs(compute_effective_trust(local_trust, reputation, mu=mu_used)
                         - local_trust)
            d_other = abs(compute_effective_trust(local_trust, reputation, mu=mu_other)
                          - local_trust)
            actual.append(d_used)
            counter.append(d_other)
            if evidence > 0.0:
                actual_ev.append(d_used)
                counter_ev.append(d_other)

    def _mean(values: list[float]) -> float:
        return statistics.mean(values) if values else float("nan")

    return {
        "decision_distortion": _mean(actual),
        "decision_distortion_evidenced": _mean(actual_ev),
        "decision_distortion_counterfactual": _mean(counter),
        "decision_distortion_counterfactual_evidenced": _mean(counter_ev),
    }


def one_run(job):
    model, attack, frac, defense, load, run_id, steps, cfg_path, adaptive = job
    base = load_config(cfg_path)
    seed = 1000 + run_id
    cfg = replace(base, steps=steps, load_level=load, random_seed=seed,
                  output_dir=Path("/tmp/asiot_attack_scratch"),
                  adaptive_mu_enabled=bool(adaptive))
    env = ASIoTEnvironment(cfg, seed=seed, baseline_name=model, run_id=run_id,
                           attack_type=attack, attacker_fraction=frac,
                           defense_enabled=defense)
    res = env.run(steps)
    rows = res["steps"]
    out = {m: statistics.mean(float(r[m]) for r in rows if m in r)
           for m in METRICS if any(m in r for r in rows)}
    aid = env.attacker_ids
    honest = [n.state.social.reputation for i, n in env.nodes.items() if i not in aid]
    att = [n.state.social.reputation for i, n in env.nodes.items() if i in aid]
    out["reputation_honest"] = statistics.mean(honest) if honest else float("nan")
    out["reputation_attacker"] = statistics.mean(att) if att else float("nan")
    # Reputation separation: positive means the system ranks honest nodes above
    # attackers, which is the property a trust system must preserve.
    out["reputation_separation"] = (
        out["reputation_honest"] - out["reputation_attacker"] if att else float("nan")
    )
    stats = env.attack_summary()
    out["attacker_defection_rate"] = stats["attacker_defection_rate"]
    out["whitewash_events"] = stats["whitewash_events"]
    out.update(decision_distortion(env))
    return model, attack, frac, defense, load, run_id, adaptive, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["attacks", "defense", "adaptive_mu"], default="attacks")
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--load", default="high")
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--output-dir", default="outputs/stage2")
    p.add_argument(
        "--attack-types",
        help="Comma-separated subset of registered attacks (attacks mode only).",
    )
    p.add_argument(
        "--models",
        help="Comma-separated subset of registered sweep models (attacks mode only).",
    )
    p.add_argument(
        "--fractions",
        help="Comma-separated attacker fractions, for example 0.1,0.2,0.3,0.4.",
    )
    args = p.parse_args()

    jobs = []
    if args.mode == "attacks":
        attacks = _parse_subset(args.attack_types, ALL_ATTACKS, "attack")
        models = _parse_subset(args.models, MODELS, "model")
        fractions = _parse_fractions(args.fractions)
        for model in models:
            for attack in attacks:
                for frac in fractions:
                    if frac == 0.0 and attack != attacks[0]:
                        continue  # one shared clean control
                    for r in range(args.runs):
                        jobs.append((model, attack if frac > 0 else "none", frac, True,
                                     args.load, r, args.steps, args.config, True))
    elif args.mode == "adaptive_mu":
        for attack in ("none", "bad_mouthing", "ballot_stuffing", "selective", "collusion"):
            for frac in ((0.0,) if attack == "none" else (0.2, 0.3, 0.4)):
                for adaptive in (True, False):
                    for r in range(args.runs):
                        jobs.append(("proposed", attack, frac, True, args.load, r,
                                     args.steps, args.config, adaptive))
    else:
        for attack in ("bad_mouthing", "ballot_stuffing", "collusion", "sybil"):
            for frac in (0.1, 0.2, 0.3, 0.4):
                for defense in (True, False):
                    for r in range(args.runs):
                        jobs.append(("proposed", attack, frac, defense,
                                     args.load, r, args.steps, args.config, True))

    print(f"[stage2:{args.mode}] {len(jobs)} runs, {args.workers} workers")
    out_dir = Path(args.output_dir) / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    results, done, t0 = [], 0, time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(one_run, j) for j in jobs]
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if done % 100 == 0 or done == len(jobs):
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(jobs)} ({rate:.1f}/s, ETA {(len(jobs)-done)/rate/60:.1f} min)")

    report = ["reputation_honest", "reputation_attacker", "reputation_separation",
              "attacker_defection_rate", "whitewash_events", "decision_distortion",
              "decision_distortion_evidenced", "decision_distortion_counterfactual",
              "decision_distortion_counterfactual_evidenced"]
    cols = METRICS + report
    with open(out_dir / "run_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "attack", "fraction", "defense", "load", "run_id",
                    "adaptive_mu"] + cols)
        for model, attack, frac, dfn, load, rid, adp, m in sorted(results, key=lambda x: x[:6]):
            w.writerow([model, attack, frac, dfn, load, rid, adp]
                       + [f"{m.get(c, float('nan')):.6f}" for c in cols])

    grouped = {}
    for model, attack, frac, dfn, load, rid, adp, m in results:
        grouped.setdefault((model, attack, frac, dfn, load, adp), []).append(m)
    with open(out_dir / "attack_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "attack", "fraction", "defense", "load",
                                          "adaptive_mu", "metric", "mean", "std", "count",
                                          "ci95_low", "ci95_high"])
        w.writeheader()
        for (model, attack, frac, dfn, load, adp), runs in sorted(grouped.items()):
            for c in cols:
                vals = [r[c] for r in runs if c in r and not math.isnan(r.get(c, float("nan")))]
                if not vals:
                    continue
                mean = statistics.mean(vals)
                std = statistics.stdev(vals) if len(vals) > 1 else 0.0
                half = 1.96 * std / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
                w.writerow({"model": model, "attack": attack, "fraction": frac,
                            "defense": dfn, "load": load, "adaptive_mu": adp, "metric": c,
                            "mean": f"{mean:.6f}", "std": f"{std:.6f}", "count": len(vals),
                            "ci95_low": f"{mean-half:.6f}", "ci95_high": f"{mean+half:.6f}"})
    print(f"[stage2] wrote {out_dir}/ in {(time.time()-t0)/60:.1f} min")


def _parse_subset(raw: str | None, allowed, label: str) -> list[str]:
    """Parse a comma-separated CLI subset and reject unknown names."""
    if not raw:
        return list(allowed)
    values = [value.strip() for value in raw.split(",") if value.strip()]
    unknown = sorted(set(values).difference(allowed))
    if unknown:
        raise SystemExit(f"unknown {label}(s): {', '.join(unknown)}")
    return values


def _parse_fractions(raw: str | None) -> list[float]:
    """Parse and validate attacker fractions."""
    values = FRACTIONS if not raw else [float(value) for value in raw.split(",")]
    if not values or any(value < 0.0 or value > 1.0 for value in values):
        raise SystemExit("fractions must be one or more values in [0, 1]")
    return list(values)


if __name__ == "__main__":
    main()
