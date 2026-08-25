# Reproducing every result

Run all commands from the repository root with the virtual environment active
and `PYTHONPATH=src` set. Every command writes to `outputs/`, which is
gitignored, so a reproduction never overwrites the committed evidence in
`results/`. Compare your output against `results/` when a run finishes.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=src              # Windows: set PYTHONPATH=src
```

## Runtimes

Measured on the reference machine (Apple M5, 10 cores, macOS 15, Python
3.13.13) with `--workers 10`. The four sweep figures are extrapolated from
timed slices of 160–330 runs each, so treat them as ±20%. The MARL figure is a
measured end-to-end wall time.

| Stage | Runs | Wall time |
|---|---|---|
| Benign sweep | 3,200 | ~18 min |
| Attack sweep | 8,250 | ~53 min |
| Ablation sweep | 1,600 | ~8 min |
| Sensitivity sweep | 4,520 | ~30 min |
| Scalability benchmark | 100 | ~10 min |
| Statistics + figures | — | <1 min |
| **Subtotal (everything except MARL)** | **17,670** | **~2 h** |
| MARL training (6 cells in parallel) | 6 × 3,000 episodes | **~24 h** |
| MARL frozen evaluation | 540 | ~15 min |
| **Full reproduction** | | **~26 h** |

Reduce `--workers` if you have fewer cores; runtimes scale roughly inversely.

## 0. Smoke test (2 minutes)

Confirms the install works and the simulator is deterministic.

```bash
pytest tests/ -q
python scripts/run_stage0_sweep.py --runs 2 --steps 100 --workers 4 --output-dir outputs/smoke
```

Expect 190 passing tests and a written `summary_by_load_baseline.csv`.

## 1. Benign sweep — 8 policies × 4 loads × 100 seeds

```bash
python scripts/run_stage0_sweep.py --runs 100 --steps 500 --workers 10 \
    --output-dir outputs/benign
```

**~18 min.** Produces `outputs/benign/baseline/summary_by_load_baseline.csv`
and `run_summary.csv`. Compare with `results/benign/`.

## 2. Attack sweep — 5 policies × (clean + 8 attacks × 4 fractions) × 50 seeds

```bash
python scripts/run_attack_sweep.py --mode attacks --runs 50 --steps 500 \
    --workers 10 --output-dir outputs/attacks
```

**~53 min.** Produces `outputs/attacks/attacks/attack_summary.csv`. Attacker
identity is drawn from the seed, so every policy faces the same attacker set at
a given seed. Compare with `results/attacks/`.

## 3. Ablation sweep — 8 variants × 4 loads × 50 seeds

```bash
python scripts/run_stage0_sweep.py --mode ablation --runs 50 --steps 500 \
    --workers 10 --output-dir outputs/ablation
```

**~8 min.** Produces `outputs/ablation/ablation/ablation_summary.csv`.
Compare with `results/ablation/`.

## 4. Sensitivity sweep — 55 coefficients × 4 perturbations × 20 seeds

```bash
python scripts/run_sensitivity.py --runs 20 --steps 500 --workers 10 \
    --load-level high --output-dir outputs/sensitivity

# The Nitti-only A4 assumption sweep (Section 5 of the paper):
python scripts/run_sensitivity.py --runs 20 --steps 500 --workers 10 \
    --model nitti_subjective_trust \
    --parameters nitti_empty_kij_fallback --values 0.0,0.3,0.5,0.7 \
    --output-dir outputs/sensitivity
```

**~30 min.** Weight groups constrained to sum to 1.0 are renormalised across
the group when one member is perturbed; unit-interval perturbations are clipped
and both requested and applied values are recorded. Compare with
`results/sensitivity/`.

## 5. Scalability benchmark — 5 node counts × 20 fresh processes

```bash
python scripts/run_scalability.py --node-counts 25,50,100,200,400 --runs 20 \
    --steps 500 --load-level high --output-dir outputs/scalability
```

**~10 min.** Each run executes in a fresh Python process so interpreter startup
is excluded from timing but peak RSS includes the runtime. Timing figures are
hardware-specific and will not match the paper exactly on other machines; the
*shape* (near-linear over this range, O(N²) code bound) should reproduce.

## 6. MARL training and frozen evaluation

Training is the long pole. Six independent cells (2 observation variants ×
3 seeds) run in parallel, one thread each.

```bash
python scripts/run_marl_full.py --episodes 3000 --steps 500 --seeds 3 \
    --load-level high --output-root outputs/marl_full
```

**~24 h.** Writes per-episode logs, a checkpoint every 250 episodes
(seed-namespaced), and periodic frozen ε=0 evaluations on held-out seeds.
Optional liveness watchdog in a second terminal:

```bash
./scripts/watchdog_marl.sh outputs/marl_full/logs 6 60
```

Then the frozen evaluation and tables:

```bash
python scripts/analyze_marl_frozen_eval.py --run-root outputs/marl_full \
    --runs 10 --output outputs/marl_frozen_benign.csv

python scripts/evaluate_marl_under_attack.py \
    --checkpoint outputs/marl_full/checkpoints/neural_marl_social_high_seed3000_ep3000.pt \
    --seeds 30 --output outputs/marl_attack_eval.csv

python scripts/build_frozen_policy_table.py \
    --attack-root outputs/marl_attack_eval --benign outputs/marl_frozen_benign.csv \
    --out-dir outputs/frozen_policy_table
```

**~15 min.** Evaluation seeds (1,400,000+) are disjoint from training seeds
(3000/6000/9000 + episode) and from mid-training selection seeds (900,000+), so
no reported number comes from a seed the policy was trained or selected on.

To skip training entirely, the six final checkpoints are committed under
`results/marl/checkpoints/` and can be passed directly to the two evaluation
commands above.

### Action-space headroom diagnostic

```bash
python scripts/diagnose_action_headroom.py --steps 200 --load-level high \
    --top-k-large 32 --output outputs/action_headroom.csv
```

**<1 min.** Explains the MARL null result: at the configured degree bound the
oracle ceiling is ~5.5%.

## 7. Paired statistics

```bash
python scripts/analyze_publication_statistics.py \
    --benign results/benign/run_summary.csv \
    --attacks results/attacks/run_summary.csv \
    --ablation results/ablation/run_summary.csv \
    --output-dir outputs/statistics
```

**<1 min.** Produces the paired t comparisons with Holm and Benjamini–Hochberg
correction for all three prespecified families (28 + 64 + 28 cells). Point the
inputs at your own `outputs/` run summaries to recompute from scratch.

## 8. Figures

```bash
python scripts/plot_validated_results.py \
    --benign-summary results/benign/summary_by_load_baseline.csv \
    --attack-summary results/attacks/attack_summary.csv \
    --ablation-summary results/ablation/ablation_summary.csv \
    --png-dir outputs/figures --pdf-dir outputs/figures

python scripts/plot_marl_learning_curves.py --run-root outputs/marl_full \
    --out-dir outputs/figures
```

**<1 min.** The learning-curve figure requires the training logs from step 6;
it plots frozen ε=0 evaluation points only, never training reward.

## 9. Parameter table

```bash
python scripts/make_parameter_table.py --markdown outputs/parameter_table.md \
    --csv outputs/parameter_table.csv
```

Regenerates `docs/parameter_table.md` from the registry in
`src/asiot/parameters.py`.

## Reduced-run mode for a faster check

Cuts the non-MARL reproduction from ~2 h to **~12 min**. Means will shift
slightly and confidence intervals will widen; the ordering of policies and the
sign of every reported effect should be unchanged. Do not quote numbers from
reduced runs.

```bash
python scripts/run_stage0_sweep.py --runs 10 --steps 500 --workers 10 --output-dir outputs/q_benign
python scripts/run_attack_sweep.py --mode attacks --runs 5 --steps 500 --workers 10 --output-dir outputs/q_attacks
python scripts/run_stage0_sweep.py --mode ablation --runs 10 --steps 500 --workers 10 --output-dir outputs/q_ablation
python scripts/run_sensitivity.py --runs 5 --steps 500 --workers 10 --output-dir outputs/q_sensitivity
python scripts/run_scalability.py --node-counts 25,50,100 --runs 5 --steps 500 --output-dir outputs/q_scalability
```

For MARL, train one cell for 300 episodes (~25 min) instead of six for 3,000:

```bash
python scripts/run_marl_training.py --episodes 300 --steps 500 --load-level high \
    --seed-start 3000 --eval-interval 100 --checkpoint-interval 100 \
    --output-dir outputs/q_marl --checkpoint-dir outputs/q_marl/checkpoints
```

The frozen evaluation will already show the paper's finding at this scale:
the trained policy does not separate from the untrained control.

## Determinism

Every run is seeded. The same seed reproduces byte-identical output:

```bash
python scripts/run_stage0_sweep.py --runs 3 --steps 200 --workers 3 --output-dir outputs/det_a
python scripts/run_stage0_sweep.py --runs 3 --steps 200 --workers 3 --output-dir outputs/det_b
diff outputs/det_a/baseline/run_summary.csv outputs/det_b/baseline/run_summary.csv && echo IDENTICAL
```

Determinism holds across worker counts: the seed is derived from `run_id`, not
from scheduling order.
