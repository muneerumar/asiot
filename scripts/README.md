# Scripts

Run everything from the repository root with `PYTHONPATH=src`.

## Paper-critical drivers

These produce the results behind the manuscript's tables and figures. Exact
commands are in [`../docs/REPRODUCE.md`](../docs/REPRODUCE.md).

| Script | Produces |
|---|---|
| `run_stage0_sweep.py` | Benign sweep (`--mode baseline`) and ablation sweep (`--mode ablation`) — Tables 2 and 4 |
| `run_attack_sweep.py` | Adversarial sweep, 8 attacks × 4 fractions — Table 3 |
| `run_sensitivity.py` | One-at-a-time sensitivity over all 55 coefficients — Table 5 |
| `run_scalability.py` | Fresh-process scaling benchmark — Table 6 |
| `run_marl_full.py` | MARL training, 2 variants × 3 seeds (wraps `run_marl_training.py`) |
| `analyze_marl_frozen_eval.py` | Frozen ε=0 benign evaluation of trained policies |
| `evaluate_marl_under_attack.py` | Frozen evaluation under attack, with the untrained control |
| `build_frozen_policy_table.py` | Table 7 |
| `analyze_publication_statistics.py` | All paired inference: CIs, Cohen's d_z, Holm and Benjamini–Hochberg |
| `plot_validated_results.py` | Figures 2, 3 and 4 |
| `plot_marl_learning_curves.py` | Figure 5 |
| `make_parameter_table.py` | `docs/parameter_table.md` |

## Diagnostics reported in the paper

| Script | Purpose |
|---|---|
| `diagnose_action_headroom.py` | Oracle ceiling of the partner-selection decision — explains the MARL null result |
| `analyze_always_accept.py` | Accept/reject degeneracy of the no-social learner |
| `diagnose_marl_reward_scale.py` | Shows the two observation variants score different objectives, so their rewards are not comparable |
| `analyze_adaptive_mu.py` | Adaptive-μ evaluation, including the mechanical-versus-trajectory decomposition |
| `stress_test_collusion.py` | n=30 re-test of the MARL collusion cell |
| `diagnose_marl_learning.py` | Frozen evaluation against an untrained control |

## Utilities

`aggregate_results.py`, `merge_attack_sweeps.py`, `build_code_documentation.py`,
`watchdog_marl.sh` (liveness monitor for long training runs), and
`run_single.py` / `run_baselines.py` / `run_ablation.py` / `run_all_loads.py` /
`evaluate_marl.py` / `plot_all.py` / `plot_results.py` — earlier single-purpose
drivers, retained because they still run. The sweep drivers above supersede them
and are what the paper used; prefer those.

Note: `run_all_loads.py` and `plot_results.py` default to output paths from an
earlier layout (`outputs/all_loads/`). They do not read any committed result and
are not part of the reproduction path.
