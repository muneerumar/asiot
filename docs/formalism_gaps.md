# Formalism-vs-implementation gaps

Places where the manuscript describes something the code does not do, or does
differently. **Every entry has to be reconciled in the text before
submission** — either by fixing the code, correcting the equation, or scoping
the claim explicitly as future work.

Status values: **FIXED** (code now matches the formalism), **TEXT MUST
CHANGE** (formalism is wrong or overstated; the text has to be corrected),
**SCOPED** (deliberately deferred, must be stated as a limitation).

Each entry records how the gap was found, so a reviewer asking "how do you
know you found them all?" gets an honest answer: most were found by audit or
by an experiment failing, not by systematic proof. This list is not claimed to
be exhaustive.

---

## G1 — Per-model success bonus (CRITICAL) — FIXED

`compute_expected_success()` added a constant keyed on the **model's name**:
+0.10 for the proposed framework, −0.05 for the static baseline, independent
of any state variable. No equation in the paper contains such a term.
Counterfactual reruns showed it produced roughly half the reported advantage.

*Text impact:* every number from the pre-fix code is void. **No result
predating this fix may appear in the resubmission.**
*Now:* single calibration constant `success_base = 0.18`, identical for all
models. `tests/test_model_identity_guard.py` fails if any model name reappears
on the scoring path, verified against a deliberately reintroduced copy of the
bug.
*Found by:* code audit.

## G2 — Reputation aggregation (Eqs. 36–37) was unreachable — FIXED

Reputation was random-initialised at construction and **never updated**, so
the aggregation equations the paper presents never executed. Any claim that
the framework performs reputation aggregation was false for the submitted
code.

*Text impact:* Eqs. 36–37 were presented as operative. They now are, but the
benign results changed when they became live (see G3).
*Found by:* code audit (Stage 2).

## G3 — Published benign results do not reproduce — TEXT MUST CHANGE

`outputs/stage0/` predates G2 and was never regenerated. Re-running the
identical protocol on current code puts the committed mean **outside the
reproduced 95% CI in 17 of 28 model × load cells**.

*Text impact:* the benign table must be regenerated from
`outputs/stage0_current/`. Critically, **the high-load advantage over
`greedy_utility` collapses to +0.0023 (paired p = 0.048, d_z = 0.20)** — a
statistical tie, not a lead. Any sentence claiming a benign high-load
advantage must be rewritten. Retention percentages computed against the old
0.7334 clean control must be recomputed against 0.7277.
*Found by:* attempting to verify a code change was inert.

## G4 — Social tie w_ij (Eqs. 6, 47) computed but never read — FIXED

The adaptive edge weight was updated every step and never consumed by partner
selection, making `without_social_graph_adaptation` a no-op ablation.
*Now:* enters interaction probability via `alpha_tie` (default 0.3).
*Found by:* code audit (Stage 0).

## G5 — θ, the privacy gate threshold (Eq. 21), was dead code — FIXED / TEXT MUST CHANGE

The earlier scalar `config.privacy_threshold = 0.35` was never read. The live
model draws a per-node threshold θ_i from
`[privacy_threshold_min, privacy_threshold_max] = [0.60, 0.95]` and gates on
that value. The unused scalar has now been removed from the dataclass, YAML,
parameter registry, and sensitivity design.

*Text impact:* Eq. 21 must use θ_i and state the uniform initialization range;
0.35 does not describe the evaluated runs.
*Found by:* the sensitivity sweep — perturbing the removed scalar by ±50%
moved nothing at all.

## G6 — `role` assigned but never read — PARTLY FIXED / SCOPED

Every node is given a role (`sensor`, `relay`, `coordinator`, `actuator`) at
construction. Until the Nitti baseline, **no code path read it**: zero
references in `utility.py`, `social_cognition.py`, `metrics.py` or the MARL
observation encoder.

*Text impact:* any claim of role-aware behaviour or role selection is
unsupported. Role now has exactly one consumer — Nitti's computation
capability (their Table IV) — which does not constitute role-based
decision-making by the proposed framework.
*Found by:* checking whether role-selection headroom could be measured.

## G7 — The learner's action space is one decision, not nine — SCOPED

The manuscript's action set (reported as Eq. 24) includes role selection,
negotiation and migration. These are absent **from the environment**, not
merely from the learner. Negotiation, federated aggregation and message-bus
modules are now explicit extension boundaries rather than misleading TODO
stubs; privacy risk/gating is fully implemented but is a constraint on partner
selection, not a distinct learned action. Migration appears nowhere. There are
no state dynamics or utility terms for the omitted actions.

*Text impact:* the action space must be described as implemented — partner
selection plus accept/reject — with the rest stated as future work. Measured
consequence: with 4–6 candidates per decision and a 5.4% utility spread, the
**oracle ceiling for any policy is a 5.5% gain over worst-choice**, which is
the honest explanation for why learning does not improve performance.
*Caveat:* Eq. 24's exact content was reported by the author; it was not
verified against the manuscript, which is not in this repository.
*Found by:* action-space headroom diagnostic.

## G8 — MARL reward was not the paper's utility — FIXED

The DQN optimised a bespoke 7-coefficient reward containing
`1.00·success + 0.30·success` (a duplicated term summing to 1.30), while the
Eq. 39–46 multi-objective utility was computed for every candidate and
discarded. The learner and the baseline it is compared against were optimising
different objectives, making the comparison uninterpretable.
*Now:* reward **is** `total_utility` from the same utility path;
`tests/test_marl_reward_matches_utility.py` pins the equality to 1e-12.
*Found by:* reading the training code before launching a long run.

## G9 — ε schedule made training vacuous — FIXED

ε decayed over 20,000 *transitions* ≈ 6 episodes, so a 3,000-episode run would
have spent 99.8% of its time exploiting a still-random network.
*Text impact:* none if unreported, but any earlier "trained" claim rests on a
run with effectively no exploration schedule.
*Found by:* pre-launch review of the training protocol.

## G10 — `communication_overhead` is a hardcoded 0.0 — TEXT MUST CHANGE

`metrics.py` defines a real `communication_overhead()` function and then emits
the constant `0.0` in the step summary. Network metrics (throughput, packet
delivery ratio, end-to-end delay) are analytic proxies, not message-level
accounting; `mqtt_bus.py` now explicitly marks that scope boundary.
*Text impact:* no network figure may be plotted from this field, and network
metrics must be labelled as proxies. Excluded from all reporting since Stage 0.
*Found by:* code audit.

## G11 — `reliability_score` duplicates `task_completion_ratio` — TEXT MUST CHANGE

`reliability_score = successes / (successes + failures + blocked)`. When
blocked ≈ 0 it is numerically identical to `task_completion_ratio`. Confirmed
in the regenerated benign results: `greedy_utility` at high load reports
**0.725427 for both**, to six decimals, because it blocks nothing.
*Text impact:* reporting both as distinct metrics inflates the apparent
evidence. Either redefine reliability as completion under adversarial
conditions, or drop it.
*Found by:* reading the regenerated benign table.

## G12 — Placeholder ablations were byte-identical to the full model — FIXED

`without_federated_placeholder` and `without_negotiation_placeholder` toggled
unimplemented modules; their rows duplicated `full_proposed` exactly. Removed.
`without_resource_awareness` was also non-behavioural and is now real
(−14.1 pts at high load).
*Found by:* code audit (Stage 0).

## G13 — `standard_marl_no_social` is not MARL — TEXT MUST CHANGE

The repository's own documentation states it is "a deterministic non-social
placeholder, not a neural training result". If the paper presents it as a MARL
baseline, that is inaccurate.
*Text impact:* rename or relabel as a non-social heuristic baseline.
*Found by:* repository documentation review.

## G14 — Adaptive μ (Eq. 38) reduces only a quantity it defines — TEXT MUST CHANGE

Adaptive μ satisfies μ_eff ≥ μ_base pointwise, and decision distortion is
(1−μ)|T−R|, so a reduction is **guaranteed by the definition**. Decomposition
over 1,300 paired runs: the mechanical term accounts for **84–131%** of the
observed reduction, and the trajectory term is indistinguishable from zero in
25 of 26 cells. No effect on cooperation, task completion or reputation
separation.
*Text impact:* must not be presented as a defense contribution. The Stage-2b
claim that the reduction "grows with attacker fraction" also does not
replicate at 50 runs — it is flat, and equally large at 0% attackers.
*Found by:* pre-registered counterfactual control.

## G15 — Nitti baseline: objective model and multi-hop trust not implemented — SCOPED

The objective model (their Eqs. 10–12) needs a DHT of pre-trusted objects;
non-adjacent trust (their Eqs. 8–9) is unreachable because only direct
neighbours are ranked. Seven further assumptions are recorded in
`docs/nitti_assumptions.md`.
*Text impact:* state that the **subjective** model is the comparator, and cite
the assumption list.
*Found by:* implementing from the published paper.

## G16 — `eval_reward` is an episode sum, not a per-decision score — TEXT MUST CHANGE

`frozen_evaluation` accumulates `total += reward` over every transition of the
episode (trainer.py:275-285), so `eval_reward` is a **cumulative sum over 500
steps**, not a per-decision mean. A policy that accepts more interactions
accrues more reward regardless of decision quality. Confirmed on the
post-training frozen evaluation (ε=0, held-out seeds 1,400,000+): the no_social
learner accepts 100% of interactions (zero rejects), so its reward
(~3,001) sits above the social learner's (~2,537) purely from accept volume
plus its higher-scale legacy heuristic objective (see `scripts/
diagnose_marl_reward_scale.py` and `scripts/analyze_always_accept.py`).

*Text impact:* `eval_reward` must NOT be reported as a performance metric in
the paper — not raw, not across variants. It is a training diagnostic only.
The comparable cross-variant metric is cooperation rate (and task-completion /
utility where the score function matches). Report reward, if at all, only
within a variant as a training curve, labelled "cumulative episode reward".
*Found by:* post-training frozen-eval analysis (Task A, Aug 2026).

## G17 — Always-accept degeneracy of the no_social learner — RESULT TO REPORT

Without social-cognitive features the learned policy degenerates to a trivial
always-accept policy: accept rate ≈ 1.00 in every condition and every seed
(social learner ≈ 0.96, i.e. ~4% rejected). This explains its inflated
cumulative reward, its lower cooperation (0.70 vs 0.73), and its collapse
under selective/collusion attack (0.51 vs 0.67 cooperation), because the
attack signal travels through the social features the no_social learner cannot
see. This is a substantive finding, not a bug: it shows the social features
are load-bearing, and it disqualifies no_social from being presented as a
competitive MARL baseline (consistent with G13).
*Text impact:* report the always-accept degeneracy and its consequences; do
not present no_social reward as evidence of anything.
*Found by:* post-training frozen-eval analysis (Task A, Aug 2026).

## G18 — Fairness was represented inconsistently in the action utility — FIXED / TEXT MUST CHANGE

`utility.py` originally exposed a six-term total containing
`fairness_utility`, while the actual proposed and ablation policies silently
overrode it with a normalized five-term total that omitted fairness. The score
finalizer and raw logger then discarded the fairness diagnostic. Consequently
`utility_weight_fairness` had no effect on evaluated decisions.

*Now:* there is one canonical five-term action objective (system, social,
resource, privacy, incentive), its normalized weights preserve the evaluated
ratios, the dead fairness weight is removed, and `fairness_utility` is retained
as a diagnostic column. Outcome fairness remains Jain's index over realized
contributions.
*Text impact:* any six-dimensional action-utility equation must become the
five-term equation; Jain fairness and workload balance must be separate.
*Found by:* end-to-end score-schema and sensitivity audit.

## G19 — Whitewashing attack never reset identity state — FIXED / RESULTS REGENERATED

`WhitewashingAttack.should_wash()` existed, but the environment never called
it. Historical whitewashing rows therefore measured only selfish service.

*Now:* at clocks 40, 80, … the environment clears social evidence involving
the logical identity, resets reputation and exposure, neutralizes learned edge
ties, and clears policy-owned Nitti feedback. The physical ID remains stable
for paired auditing. Event counts are logged and regression-tested. All
affected full-protocol whitewashing cells were rerun; older rows are
superseded.
*Found by:* adversarial-hook reachability audit.

## G20 — Episode reset and load preset state leaked or was ignored — FIXED

`reset()` did not reseed attack or NumPy generators, clear attack statistics,
or reset stateful policy ledgers. Separately, legacy load YAML files used
ignored keys and failed to set `simulation.load_level`, so high and extreme
presets parsed as medium.

*Now:* reset is episode-complete and replay-tested. Load presets use the live
schema, select their named level, and hold all other experimental factors
constant. Dead `tasks_per_step` and scalar privacy fields are removed.
Five unused high-level weights and inactive MQTT/federated/MARL feature flags
are also removed rather than exposing configuration that cannot affect a run.
*Found by:* configuration round-trip and repeated-reset audit.

## G21 — Frozen MARL cooperation was measured from a single time step — FIXED / RESULTS REGENERATED

`analyze_marl_frozen_eval.py` computed each run's cooperation as
`steps[-1]["cooperation_rate"]` — the final step alone, which at high load is
3–8 interactions — instead of the mean over all 500 steps used everywhere else
in the repository. The resulting per-run values were small-denominator
fractions (3/8, 3/7, 5/7), inflating the 95% CI by about 19x.

*Impact:* the manuscript's neural cooperation figures were 0.735000 and
0.710119; the correct run-level values are **0.727879** and **0.699378**. The
direction is unchanged and the true gap is slightly larger, but the reported
uncertainty was wrong enough to change an inference: a two-sample test on the
published numbers gives `p = 0.55`, while the corrected values give
`p = 1.5e-17`. Anyone re-analysing the supplied per-seed CSV would have
concluded the difference was noise.
*Now:* run-level mean; `frozen_benign.csv` regenerated; manuscript updated.
*Found by:* verifying manuscript numbers against source CSVs — the impossible
fraction values were the tell.

## G22 — The frozen-policy table used an undeclared statistical test — FIXED / TEXT MUST CHANGE

The paper declares paired *t* inference with Holm and Benjamini–Hochberg
correction for the benign, attack and ablation families. The frozen MARL table
used Wilcoxon signed-rank instead, without saying so.

The two tests disagree on exactly the cells the learning claim rests on. Over
the six trained-versus-untrained comparisons with Holm correction:

| Cell | mean diff | d_z | paired *t* | Wilcoxon |
|---|---|---|---|---|
| no_social / benign | −0.003278 | −4.64 | survives (negative) | survives (negative) |
| social / collusion | +0.007006 | +0.53 | **survives** | fails |
| social / selective | +0.005355 | +0.50 | **survives** | fails |

*Impact:* under the declared method, training does produce small
Holm-significant gains over random initialization under attack — the opposite
of what a Wilcoxon-only table shows. Neither reading changes the headline,
because the trained social policy remains statistically indistinguishable from
the deterministic policy in all three conditions (paired *t*: `p = 0.95`,
`0.07`, `0.12`; `|d_z| ≤ 0.35`), but the paper must not present one test's
answer while declaring another's method.
*Now:* the table records both tests plus paired `d_z`; the manuscript reports
the disagreement explicitly.
*Found by:* recomputing a cited p-value from the per-seed data and getting a
different number.

---

## Summary for the manuscript

| Class | Entries |
|---|---|
| Results void / must be regenerated | G1, G3; G19 whitewashing rows only; G21 neural rows only |
| Equation does not match code | G5, G7, G11, G13, G18 |
| Claim must be weakened or dropped | G3, G6, G10, G14, G16 |
| Fixed in code | G2, G4, G5, G8, G9, G12, G18, G19, G20, G21, G22 |
| Scope as future work | G7, G15 |
| Result to report | G17, G22 |

The single most consequential item is **G3**: the headline benign advantage at
high load is a statistical tie. The adversarial results, including the
corrected whitewashing cells in G19, remain strong — proposed retains 87.8%
of clean cooperation under selective
attack at f = 0.4 versus 62.8% for greedy utility — so the paper's resilience
claim carries the contribution, not the benign comparison.
