# Nitti et al. (2014) baseline — implementation notes and assumptions

Reference: M. Nitti, R. Girau, L. Atzori, "Trustworthiness Management in the
Social Internet of Things," *IEEE Transactions on Knowledge and Data
Engineering*, 26(5):1253–1266, 2014.

Implementation: `src/asiot/nitti_trust.py` (equations),
`asiot.baselines.NittiSubjectiveTrustModel` (policy), registry name
`nitti_subjective_trust`.

## What is implemented

The **subjective** model, Eqs. (1)–(7), verbatim:

| Eq. | Quantity | Implementation |
|-----|----------|----------------|
| (1) | `T_ij = (1-α-β)R_ij + α·O_dir_ij + β·O_ind_ij` | `subjective_trustworthiness` |
| (2) | `R_ij = |K_ij| / (|N_i|-1)` | `centrality` |
| (3) | experience-weighted direct opinion | `direct_opinion` |
| (4) | long-term opinion over `L_lon` | `FeedbackLedger.opinion` |
| (5) | short-term opinion over `L_rec` | `FeedbackLedger.opinion` |
| (6) | credibility-weighted common-friend opinion | `indirect_opinion` |
| (7) | `C_ik = η·O_dir_ik + (1-η)·R_ik` | `credibility` |

Parameters are the paper's own optimal configuration (Table V, subjective
model): α = 0.4, β = 0.3, γ = 0.5, δ = 0.5, η = 0.7, L_lon = 50, L_rec = 5.
Table IV supplies the relationship factor and computation-capability values.
**No parameter was re-fitted for this simulator.**

Two further choices are the *authors'*, not ours: binary feedback
`f ∈ {0,1}`, and transaction factor `ω_l = 1` for all transactions ("we
considered all the transactions equally important").

## What is NOT implemented, and why

- **Objective model (Eqs. 10–12).** Presumes a DHT of Pre-Trusted Objects and
  a network-wide view of every node's feedback. This simulator has no such
  infrastructure. The subjective model is also the socially-grounded variant,
  which is the fair comparator for a social-cognition framework.
- **Non-adjacent trust `T'_ij` (Eqs. 8–9).** The multi-hop product over a
  friendship chain is unreachable here: partner selection only ever ranks the
  requester's direct neighbours, so `p_i` and `p_j` are adjacent by
  construction.

## ASSUMPTIONS — choices the paper leaves open

Each is marked in the source with the same identifier.

**A1 — Relationship factor `F_ij`.** The paper's taxonomy (OOR 1.0, CLOR 0.8,
CWOR 0.8, SOR 0.6, POR 0.5) has no counterpart in this simulator, whose nodes
carry a `domain` instead. Same-domain pairs are treated as **CWOR (0.8)**
(objects working in the same domain are co-workers); cross-domain pairs as
**SOR (0.6)**, the generic social relation. OOR is unreachable (no node owns
another) and POR is unreachable (no manufacturer/batch notion). The chosen
pair spans the middle of the paper's range rather than its extremes, so the
baseline is neither flattered nor penalised by the mapping.

**A2 — Computation capability `I_j`.** Mapped onto this simulator's node
`role`, which was otherwise dead code. Nitti's Class 2 is "any object just
capable of providing a measure of the environment status", so `sensor` →
Class 2 (0.2); `relay`, `coordinator`, `actuator` are programmable devices →
Class 1 (0.8). Note the paper's convention that **higher capability lowers
trust** — Eq. (3) uses `(1 - I_j)` — because a smarter object can cheat more
effectively.

**A3 — Centrality when `|N_i| = 1`.** Eq. (2) divides by `|N_i| - 1`, which
vanishes for a single-friend node; the paper does not define this case. We
return 0.0 (no basis for judging shared social position). The graph enforces
`min_neighbors ≥ 3`, so this is defensive rather than routine.

**A4 — Indirect opinion with no common friends.** Eq. (6) is undefined when
`K_ij = ∅` or all credibilities are zero. We return **0.5** (maximal
ignorance) rather than 0.0, which would be indistinguishable from unanimous
condemnation and would unfairly penalise the baseline in a sparse graph. With
mean degree ≈ 4.5, empty `K_ij` is common, so this choice is consequential —
it is the assumption most worth revisiting if the baseline underperforms.

**A5 — Feedback history storage.** Eqs. (4)–(5) need the last `L_lon`
transactions for an *ordered pair*; this simulator tracks only aggregate
interaction counts. A `FeedbackLedger` owned by the policy instance (one per
environment) records binary feedback per pair. This required a new
`BaselinePolicy.observe_outcome` hook, a no-op for every other policy.
Verified bit-identical results for all existing models across 192 runs.

**A6 — Embedding `T_ij` in the scoring pipeline.** The paper ranks providers
by `T_ij` alone. To keep the comparison about the *trust model* rather than
the harness, `T_ij` is placed in the same scoring skeleton the other social
baselines use (QoS, distance, resource and utility terms identical to
`honesty_based_social`), with `T_ij` supplying `effective_trust` and
`preference`. Ranking remains dominated by `T_ij` (0.7 weight) as the paper
specifies: "a node chooses the provider of the service on the basis of the
highest computed trustworthiness level".

**A7 — Privacy gate.** Nitti's model has no privacy notion. Like the other
external baselines, `privacy_allowed = 1` and `privacy_risk = 0.5`, so the
baseline is not charged for a mechanism its authors never proposed.
