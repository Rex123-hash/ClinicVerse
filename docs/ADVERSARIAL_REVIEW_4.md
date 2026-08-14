# the independent review Review + Repair #4 — Narrow Scientific Audit

**Date:** 2026-08-11
**Scope:** the five user-prioritized M4 validity questions, plus Set C and split isolation
**Verdict:** **PASS after one result-invalidating implementation repair**

## 1. Bug found and repaired

`FixedOrderBatch.score_batch()` ignored step/history and returned the same static priority on every
request. Under support-blind replay, `fixed_domain_order` therefore repeatedly bought the first
affordable group. This contradicted both the predeclared deterministic sequence and the tested
`twinbench.episode.FixedOrder`, and produced non-monotone fixed-policy budget curves.

The repair adds per-patient fixed-order cursors to the batched policy. Each patient now advances
through the unchanged predeclared sequence once, while availability, affordability, repeated-action
legality, charging, and disclosure remain owned by `DisclosureEngine`. Repeats remain valid engine
operations for any policy that selects them. Regression tests cover sequence advancement,
unavailable/unaffordable skipping, and exhaustion.

Because the defect changed the primary policy and grid rankings, the full primary + 16-condition M4
run was necessarily repeated. No policy, order, mask rate, cost regime, budget, endpoint, seed,
sample, model, or bootstrap choice changed.

## 2. Independent raw-artifact rescore

The repaired `primary_predictions.npz` contains 8,000 aligned labels/record IDs and 48 policy-budget
prediction arrays. A direct NumPy implementation independently computed patient log losses and the
normalized trapezoidal integral over β = 0, .1, .2, .3, .4, .5, .75, 1.

| rank | policy | independently rescored AUNLLC |
|---:|---|---:|
| 1 | `fixed_domain_order` | **0.319414417325** |
| 2 | `random_train_frequency` | 0.320947451764 |
| 3 | `random_uniform_all` | 0.323660196664 |
| 4 | `greedy_eig` | 0.326586374192 |
| 5 | `greedy_eig_per_cost` | 0.326713476172 |
| 6 | `no_acquisition` | 0.326957381285 |

All 48 primary NLLs agree with `results.json` within **1.12e-16**. As a repair check beyond the
narrow request, all 768 grid NLLs, all 96 grid AUNLLCs, and all 16 rankings also agree within
**1.12e-16** with zero rank mismatches.

## 3. Primary top-two paired interval

An independent explicit bootstrap rebuilt the eight budget-point NLL curves for the same 8,000
resampled patient indices, reintegrated each curve, and differenced them in every one of 1,000
replicates (seed 20260809).

`fixed_domain_order − random_train_frequency = −0.001533034439`, percentile 95% CI
**[−0.002814387852, −0.000251109072]**. Lower is better, so the repaired fixed-order primary result
is statistically resolved against training frequency under this predeclared analysis.

## 4. Support-blind leakage audit

PASS.

- `PolicyView` contains disclosed values/mask, detached immutable action metadata, boundary, and
  spend only. It contains no hidden support, hidden values, historical observed mask, hidden-cell
  count, natural/synthetic missingness flag, future state, or label.
- Under `support_blind`, `requestable_panels()` is the constant whole catalogue. The evaluator adds
  only affordability from public action cost and remaining budget; it does not support-filter the
  candidate list.
- EIG-like scores receive features rebuilt only from `engine.view()`. Quantiles are fitted from the
  outer-training patients; hidden true values and support never enter scoring.
- Random-frequency weights use only the current fold's outer-training mask. Random and fixed
  baselines receive no patient-specific hidden state.
- Existing hidden-value invariance tests pass. Review #4 adds a hidden-support invariance test:
  identical visible states with different hidden availability and identical RNG state produce the
  same next support-blind action.
- Empty requests are charged first at the same declared cost as successful requests. There is no
  free availability probe.

The artifact key `greedy_eig` is historical. The exact computation is
`H(p_now) − mean(H(p_q25), H(p_q50), H(p_q75))` after feature-space completions. Equal-weight
quantiles are not a coherent posterior predictive distribution, `p_now` need not equal their mean,
acquisition-success probability is absent, and simulated slope/global distinct-count features are
left stale. The accurate term is **surrogate expected-entropy-reduction heuristic**, not true EIG.

## 5. Request failure and disclosure audit

At β = 1 in the primary condition, exact totals from all five fold rows give:

| policy | requests/patient | success/patient | failure/patient | request failure | unique new cells/patient | spend/patient |
|---|---:|---:|---:|---:|---:|---:|
| `random_train_frequency` | 8.34825 | 2.00425 | 6.34400 | **75.99197%** | **11.55600** | 11.89836 |
| `greedy_eig` | 9.48188 | 0.51275 | 8.96913 | **94.59231%** | **2.42813** | 11.89877 |

The denominator is **requests**. A failure is one paid action returning exactly zero newly disclosed
cells. It is neither patient-level failure nor “entire patient got nothing.” Partially successful
groups count as successful. `Purchase.n_disclosed` counts only `hidden ∩ group ∩ current boundary`;
the engine clears those cells immediately, preventing overlap/repeat double counting.

Trace retention was repaired from a truncated first-policy prefix to complete histories for two
patients per fold and requesting policy. The repaired artifact holds 215 action rows covering 50
complete sampled policy-patient histories. Independently summed costs equal initial minus final
budget in every history; none overspends the β=.5 budget and no remaining budget is negative.

The EIG/frequency contrast survives the audit, but it is benchmark-specific and descriptive. It
does not establish that availability is generally more valuable than information utility.

## 6. Bootstrap and fair reversal count

The bootstrap resamples patients, uses one paired index array for both policies, calculates every
budget's NLL, integrates over the fixed normalized grid, and differences AUNLLC. It uses 1,000
replicates, seed 20260809, and a 2.5/97.5 percentile interval for `b − a`.

Review #4 optimized the implementation using linearity: the weighted eight-budget point loss is
integrated per patient once, then averaged on each paired resample. A regression against explicit
curve rebuilding matches point/interval results to **1e-15**.

The old report called a reversal supported when either one of its two condition-specific intervals
excluded zero, even if the other included zero. The predeclared one-condition flag is retained by
name for transparency, but `SUPPORTED REVERSAL` now requires both relevant paired intervals to
resolve. After the fixed-order repair:

- support-blind conditions: mean Kendall tau-b **+0.776**, minimum **+0.600**;
- fair-protocol winner: `fixed_domain_order` in **8/8** conditions;
- fair-protocol winner changes: **0/28 condition pairs**;
- fair-protocol supported reversals: **0**.

## 7. Set and split isolation

PASS with one disclosed nuance.

- Runtime logs and provenance show exactly 8,000 patients from sets A+B; Set C is excluded and was
  never loaded.
- Primary record IDs/labels match the development cohort exactly. The seeded 2,000-patient grid
  subset and stored hash reproduce exactly and contains 378/409/425/393/395 outer-test patients
  from folds 0–4, covering all 2,000 once.
- Per fold: 4,800 model-training rows fit the median imputer and XGBoost; a disjoint 1,600 clean
  calibration rows fit Platt scaling; a disjoint 1,600 outer-test rows are prediction-only. The five
  repaired full models use 223/225/227/225/229 split features, so the degenerate-booster guard holds.
- Policy frequencies and quantiles use the 6,400 outer-training patients (model-training plus
  calibration covariates), never the 1,600 outer-test patients and never calibration/test labels.
  Thus predictor/calibrator/test isolation and the predeclared no-outer-test policy boundary hold.
  The use of unlabeled calibration covariates for policy priors is recorded explicitly rather than
  being misdescribed as model-training-only.

## 8. Decision

**PASS after repair. M4-C — largely stable/null. M5 SAFE: YES.**

The strongest supported headline is: under the fair support-blind replay, the corrected fixed
domain-motivated sequence ranks first in all eight predeclared cost/mask conditions, with no fair
winner changes; the entropy heuristic has a much higher zero-new-cell request rate than the
training-frequency comparator in the primary condition.

Forbidden: “EIG is generally inferior,” “availability matters more than information,” or any claim
about prospective clinical ordering, causality, deployment, or real-world hospital utility.
