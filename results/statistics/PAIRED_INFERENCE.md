# Prespecified paired-inference report

The primary endpoint is run-level cooperation rate. Every difference is
focal minus comparator, so positive values favor the proposed/full model.
Holm correction controls family-wise error within each declared family;
Benjamini-Hochberg q-values are supplied as a complementary FDR analysis.
Confidence intervals and Cohen's paired d_z quantify practical effects.

| Family | Comparisons | Pairs/cell | Holm significant (+/−) | BH significant |
|---|---:|---:|---:|---:|
| benign | 28 | 100 | 28 (28/0) | 28 |
| attacks | 64 | 50 | 61 (59/2) | 61 |
| ablation | 28 | 50 | 15 (15/0) | 16 |

## Prespecified headline cells

| Family/cell | Comparator | Mean difference | 95% CI | d_z | Holm p |
|---|---|---:|---:|---:|---:|
| benign: high | game_theoretic_social | 0.016290 | [0.013853, 0.018728] | 1.326 | 4.449e-23 |
| benign: high | greedy_utility | 0.002266 | [0.000023, 0.004508] | 0.200 | 0.04773 |
| benign: high | honesty_based_social | 0.039131 | [0.036811, 0.041451] | 3.346 | 1.165e-54 |
| benign: high | nitti_subjective_trust | 0.074401 | [0.072007, 0.076795] | 6.166 | 1.852e-79 |
| benign: high | non_agentic_static | 0.164240 | [0.161948, 0.166532] | 14.216 | 8.034e-115 |
| benign: high | standard_marl_no_social | 0.024756 | [0.022420, 0.027093] | 2.102 | 2.462e-37 |
| benign: high | trust_unaware | 0.053176 | [0.050853, 0.055500] | 4.540 | 7.784e-67 |
| attack: collusion, f=0.4 | greedy_utility | 0.180092 | [0.173943, 0.186241] | 8.323 | 2.345e-45 |
| attack: collusion, f=0.4 | nitti_subjective_trust | 0.088217 | [0.083063, 0.093370] | 4.865 | 1.632e-34 |
| attack: selective, f=0.4 | greedy_utility | 0.184590 | [0.179373, 0.189806] | 10.056 | 2.673e-49 |
| attack: selective, f=0.4 | nitti_subjective_trust | 0.094484 | [0.089240, 0.099727] | 5.121 | 1.65e-35 |
| attack: sybil, f=0.4 | greedy_utility | 0.186409 | [0.180687, 0.192130] | 9.259 | 1.442e-47 |
| attack: sybil, f=0.4 | nitti_subjective_trust | 0.083531 | [0.078484, 0.088578] | 4.704 | 7.413e-34 |
| attack: whitewashing, f=0.4 | greedy_utility | 0.094240 | [0.087810, 0.100671] | 4.165 | 1.932e-31 |
| attack: whitewashing, f=0.4 | nitti_subjective_trust | 0.045299 | [0.038008, 0.052590] | 1.766 | 1.086e-15 |
| ablation: high | without_incentive | 0.002849 | [-0.000480, 0.006177] | 0.243 | 1 |
| ablation: high | without_preference | 0.005686 | [0.002715, 0.008656] | 0.544 | 0.005541 |
| ablation: high | without_privacy_gate | 0.000678 | [-0.002450, 0.003806] | 0.062 | 1 |
| ablation: high | without_reciprocity | 0.013961 | [0.010388, 0.017534] | 1.111 | 6.415e-09 |
| ablation: high | without_resource_awareness | 0.140930 | [0.137472, 0.144387] | 11.585 | 1.203e-52 |
| ablation: high | without_social_graph_adaptation | 0.003229 | [0.000400, 0.006059] | 0.324 | 0.34 |
| ablation: high | without_trust | 0.025787 | [0.022634, 0.028940] | 2.324 | 3.134e-20 |

## Interpretation rule

Statistical detection alone is not described as practical superiority.
The manuscript should interpret the confidence interval, paired d_z,
and the metric scale together. In particular, the benign high-load
proposed-versus-greedy result remains a small effect even if a raw or
adjusted test crosses 0.05.
For attacks, significance is not synonymous with superiority: the two
Holm-significant negative cells are bad-mouthing versus greedy utility at
fractions 0.30 and 0.40 (differences about −0.006). Bad-mouthing acts on
feedback, so reputation separation and decision distortion are the more
direct endpoints for that mechanism.
