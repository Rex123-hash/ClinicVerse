# M4 Milestone Report — Corrected Acquisition-Policy Ranking Stability

**Date:** 2026-08-11
**Design:** predeclared in [`M4_DESIGN.md`](M4_DESIGN.md) at commit `1f1ddc7`
**Review:** [`ADVERSARIAL_REVIEW_4.md`](ADVERSARIAL_REVIEW_4.md)
**Classification:** **M4-C — LARGELY STABLE / NULL**
**Status:** **ACCEPT after Review #4 repair**

Review #4 found that the batched `fixed_domain_order` implementation repeatedly selected its first
affordable priority rather than advancing through the predeclared sequence. The original M4 report
and its fixed-policy values are superseded. The policy was repaired without changing its declared
order, all primary and grid predictions were rerun, and the artifacts were independently rescored.

## Corrected primary result

Primary condition: `support_blind / shared_plus_marginal / mask 0.6`, n = 8,000. Lower AUNLLC is
better.

| rank | policy | AUNLLC |
|---:|---|---:|
| 1 | `fixed_domain_order` | **0.319414** |
| 2 | `random_train_frequency` | 0.320947 |
| 3 | `random_uniform_all` | 0.323660 |
| 4 | `greedy_eig` | 0.326586 |
| 5 | `greedy_eig_per_cost` | 0.326713 |
| 6 | `no_acquisition` | 0.326957 |

The paired difference `fixed_domain_order − random_train_frequency` is **−0.001533**, with a
patient-level 1,000-replicate percentile 95% CI **[−0.002814, −0.000251]**. The corrected primary
top two are distinguishable under the predeclared bootstrap.

## Corrected fair-protocol stability

`fixed_domain_order` ranks first in **all 8/8 support-blind conditions**. Across the 28 fair-protocol
condition pairs, mean Kendall tau-b is **+0.776** and the minimum is **+0.600**. There are **zero
descriptive fair-protocol winner changes and therefore 0 supported fair-protocol reversals**.

The support-aware runs remain a **diagnostic historical-availability oracle**, not deployable
evidence. Their descriptive rank changes do not upgrade the corrected fair-protocol result.

## Acquisition accounting at full primary budget

Failure means a **request that costs its full declared price and discloses zero new cells**. It is a
request-level denominator, not a patient-level failure. A partially successful group is successful;
cells disclosed by overlaps or repeats cannot be counted again because disclosed hidden cells are
removed from engine state.

| policy | requests/patient | successful/patient | failed/patient | request failure | new cells/patient | spend/patient |
|---|---:|---:|---:|---:|---:|---:|
| `fixed_domain_order` | 10.000 | 3.013 | 6.987 | 69.87% | 13.946 | 12.300 |
| `random_train_frequency` | 8.348 | 2.004 | 6.344 | **75.99%** | **11.556** | 11.898 |
| `random_uniform_all` | 9.662 | 1.912 | 7.750 | 80.21% | 8.783 | 11.839 |
| `greedy_eig` | 9.482 | 0.513 | 8.969 | **94.59%** | **2.428** | 11.899 |
| `greedy_eig_per_cost` | 9.530 | 0.509 | 9.021 | 94.66% | 2.338 | 11.900 |

The historical key `greedy_eig` is retained for artifact compatibility. Technically it is a
**surrogate expected-entropy-reduction heuristic** based on three equal-weight training quantiles,
not mutual-information EIG. It does not model acquisition-success probability. The only supported
secondary statement is that, in this benchmark condition, it has a much higher zero-new-cell
request rate than the training-frequency baseline, limiting disclosure.

## Decision

M4-A is not supported. With a stable fair-protocol winner and no fair winner changes, the corrected
ranking result is **M4-C**. M4 is accepted after repair; M5 is safe to begin, but this report does not
start or design M5.
