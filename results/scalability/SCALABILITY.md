# Scalability and computational-overhead report

Protocol: 20 fresh-process repetitions per node count, 500 steps per run, `high` load, `proposed` policy.
Runs were sequential to avoid CPU contention. Interpreter startup is excluded
from timing; peak RSS includes the Python runtime and imported libraries.

| Nodes | Wall ms/step (95% CI) | CPU ms/step | Peak RSS MiB | Cooperation | Active nodes |
|---:|---:|---:|---:|---:|---:|
| 25 | 1.767 [1.752, 1.781] | 1.761 | 145.0 | 0.6662 | 23.1 |
| 50 | 2.611 [2.588, 2.634] | 2.607 | 145.2 | 0.7267 | 50.0 |
| 100 | 4.332 [4.293, 4.370] | 4.323 | 145.8 | 0.7378 | 100.0 |
| 200 | 8.387 [8.213, 8.560] | 8.360 | 146.7 | 0.7360 | 200.0 |
| 400 | 22.077 [20.975, 23.179] | 22.039 | 147.8 | 0.7365 | 400.0 |

## Scaling interpretation

Across 25–400 nodes, mean wall time per step increased by 12.50× and peak RSS by 1.02×.
A log-log fit gives an empirical runtime exponent of 0.897 (R²=0.968, p=0.002409).

The code-level bound is dominated by the all-node reputation refresh:
each step considers reports for every target/reporter pair, O(N²). Dynamic
edge maintenance is O(Nd) for bounded degree d, and partner scoring is
O(Td) for T generated tasks. The empirical exponent is a finite-range
measurement, not a replacement for that asymptotic bound.

## Scope boundary

These are simulator compute measurements. They do not measure packet delay,
broker traffic, radio energy, or real-device latency.

Raw benchmark rows: 100.
