# A-SIoT — agentic partner selection in the Social Internet of Things

Deterministic analytical simulator for socially intelligent, attack-resilient
partner selection in the Social Internet of Things. Each requester is a bounded,
goal-directed agent that scores its local neighbours through one pipeline —
trust, reputation, preference, reciprocity, resource state and a per-node
privacy gate — and selects a partner by a five-term multi-objective utility.
This repository contains the simulator, the full experimental protocol, every
aggregated result behind the paper's tables and figures, and the scripts that
regenerate them. All policies share task generation, topology, resource
dynamics, outcomes and random seeds, so comparisons differ only in candidate
scoring.

**Paper:** Maqsood, S.; Umar, M.M.; Iqbal, Z.; Mehmood, A. *Socially Intelligent
and Attack-Resilient Partner Selection in the Social Internet of Things: An
Agentic Simulation Study.* Big Data and Cognitive Computing (MDPI), 2026.
DOI to be added on acceptance. See `CITATION.cff`.

## Quick start

```bash
git clone <repository-url> asiot && cd asiot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src

# 2-minute smoke test: verifies the install and the simulator
pytest tests/ -q                                    # expect 190 passed
python scripts/run_stage0_sweep.py --runs 2 --steps 100 --workers 4 \
    --output-dir outputs/smoke
```

Python ≥ 3.11 is required. `torch` is needed only for the MARL experiments;
everything else runs without it.

## Regenerating each manuscript table and figure

Run from the repository root with `PYTHONPATH=src`. Times assume `--workers 10`
on the reference machine. Full detail, including a reduced-run mode, is in
[`docs/REPRODUCE.md`](docs/REPRODUCE.md).

| Manuscript item | Command | Time | Committed source |
|---|---|---|---|
| **Table 2** — benign cooperation, 8 policies × 4 loads | `python scripts/run_stage0_sweep.py --runs 100 --steps 500 --workers 10 --output-dir outputs/benign` | ~18 min | `results/benign/summary_by_load_baseline.csv` |
| **Table 3** — cooperation at attacker fraction 0.40 | `python scripts/run_attack_sweep.py --mode attacks --runs 50 --steps 500 --workers 10 --output-dir outputs/attacks` | ~53 min | `results/attacks/attack_summary.csv` |
| **Table 4** — high-load ablation | `python scripts/run_stage0_sweep.py --mode ablation --runs 50 --steps 500 --workers 10 --output-dir outputs/ablation` | ~8 min | `results/ablation/ablation_summary.csv` |
| **Table 5** — Holm-robust sensitivity effects | `python scripts/run_sensitivity.py --runs 20 --steps 500 --workers 10 --output-dir outputs/sensitivity` | ~30 min | `results/sensitivity/sensitivity_proposed_high.csv` |
| **Table 6** — fresh-process scalability | `python scripts/run_scalability.py --node-counts 25,50,100,200,400 --runs 20 --steps 500 --output-dir outputs/scalability` | ~10 min | `results/scalability/scalability_summary.csv` |
| **Table 7** — frozen-policy MARL comparison | see [`docs/REPRODUCE.md` §6](docs/REPRODUCE.md) (training ~24 h, or reuse the committed checkpoints, ~15 min) | ~24 h / ~15 min | `results/marl/frozen_policy_table.csv` |
| **Table 1** — experimental protocol | descriptive; no computation | — | — |
| **Figure 2** — benign cooperation by load | `python scripts/plot_validated_results.py --benign-summary results/benign/summary_by_load_baseline.csv --attack-summary results/attacks/attack_summary.csv --ablation-summary results/ablation/ablation_summary.csv --png-dir outputs/figures --pdf-dir outputs/figures` | <1 min | `figures/benign_cooperation.{png,pdf}` |
| **Figure 3** — attack resilience curves | same command as Figure 2 | <1 min | `figures/attack_resilience.{png,pdf}` |
| **Figure 4** — high-load ablation | same command as Figure 2 | <1 min | `figures/ablation_high_load.{png,pdf}` |
| **Figure 5** — MARL learning curves | `python scripts/plot_marl_learning_curves.py --run-root outputs/marl_full --out-dir outputs/figures` | <1 min | `figures/marl_learning_curves.{png,pdf}` |
| **Figure 1** — decision-cycle diagram | drawn in TikZ inside the manuscript; no data | — | — |
| Paired inference (all reported *p*, CI, *d_z*) | `python scripts/analyze_publication_statistics.py --benign results/benign/run_summary.csv --attacks results/attacks/run_summary.csv --ablation results/ablation/run_summary.csv --output-dir outputs/statistics` | <1 min | `results/statistics/paired_primary_comparisons.csv` |
| Parameter table (all 55 coefficients) | `python scripts/make_parameter_table.py` | <1 s | `docs/parameter_table.md` |
| Action-space headroom diagnostic | `python scripts/diagnose_action_headroom.py --steps 200 --load-level high --top-k-large 32 --output outputs/action_headroom.csv` | <1 min | `results/marl/action_headroom.csv` |

Figures 2–4 are produced by a single command; it writes all three.

## Hardware and total compute

All results were produced on one machine: **Apple M5, 10 cores, 24 GB RAM,
macOS 15 (Darwin arm64), Python 3.13.13.**




## Determinism

Results are seed-deterministic: the same seed produces byte-identical output.
Seeds are `1000 + run_id` for sweeps and are derived from `run_id` rather than
scheduling order, so determinism holds regardless of `--workers`. Verified in
`tests/test_reproducibility.py` and reproducible directly:

```bash
python scripts/run_stage0_sweep.py --runs 3 --steps 200 --workers 3 --output-dir outputs/det_a
python scripts/run_stage0_sweep.py --runs 3 --steps 200 --workers 3 --output-dir outputs/det_b
diff outputs/det_a/baseline/run_summary.csv outputs/det_b/baseline/run_summary.csv && echo IDENTICAL
```

## Repository layout

```
src/asiot/      simulator: social cognition, utility, policies, attacks, MARL
scripts/        every experiment driver and analysis script
tests/          full suite (190 tests), including the model-identity guard
config/         all configurations used in the paper
docs/           parameter table, assumptions, gap register, reproduction guide
results/        aggregated and per-run summary CSVs behind every table
figures/        final PNG and PDF figures
```

## Known limitations

Read [`docs/formalism_gaps.md`](docs/formalism_gaps.md) before interpreting any
result. It records 22 places where the original formalism and the implementation
diverged, what was fixed, and what the paper's text says instead. The most
important entries:

- **The benign high-load result is a tie**, not a win: proposed 0.727692 versus
  greedy utility 0.725427 (paired *d_z* = 0.200). The proposed policy also loses
  on task completion and fairness at that load — a safety-versus-throughput
  trade-off created by the privacy gate. The paper's claim rests on adversarial
  resilience, not benign performance.
- **Learning does not improve partner selection here.** The trained policy is
  statistically indistinguishable from the deterministic one, and a randomly
  initialised network performs the same. The measured cause is that the decision
  has a ~5.5% oracle ceiling over 4–6 candidates.
- **Feedback attacks (bad-mouthing, ballot-stuffing) show ~100% retention** for
  every policy and are excluded from the resilience claim; two bad-mouthing
  cells are significant *in favour of* the greedy baseline and are reported.
- **Network metrics are analytic proxies.** No MQTT broker, packet emulator,
  radio model or testbed. `communication_overhead` is not reported at all.
- **Not implemented:** federated learning, negotiation, role dynamics, migration,
  message-level accounting, and the Nitti objective/DHT model. These appear in
  the original architecture description but not in the evaluated code.
- **No 2024–2026 attack-resilient SIoT method has been reproduced** as a
  baseline. The external comparator is Nitti et al. (2014), implemented from the
  published equations with seven documented assumptions
  ([`docs/nitti_assumptions.md`](docs/nitti_assumptions.md)).

### Data integrity

An earlier version of this simulator contained a per-model-name constant inside
`compute_expected_success()` that granted the proposed model a +0.10 success
bonus and the static baseline −0.05, independent of any state variable. It
accounted for roughly half the originally reported advantage. **No output
predating that fix appears in this repository or in the paper**, and
`tests/test_model_identity_guard.py` now fails if any model name reaches the
scoring path — verified against a deliberately reintroduced copy of the bug.

## What is not included

- **Per-step interaction logs** (`interactions*.csv`, `steps.csv`). Large, and no
  table depends on them. Regenerate by rerunning any sweep.
- **Intermediate MARL checkpoints.** Only the final checkpoint per cell is kept
  (6 files, 4.6 MB). The every-250-episode checkpoints are omitted for size; the
  learning-curve data they support is committed as
  `results/marl/learning_curves.csv`.
- **The manuscript source.** Held separately by the authors.

## License

MIT — see `LICENSE`.
