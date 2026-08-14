# M3 Milestone Report - Repaired After the independent review Review #3

**Original run:** 2026-08-09
**Repair:** 2026-08-10
**Classification:** **M3-B - FEATURE-IDENTITY EFFECT**
**Recommendation:** **PASS M3 WITH NONBLOCKING RISKS; M4 may proceed only under the repaired contract.**

The original metrics reproduce, but the original claim that group structure
matters beyond amount does not survive the stronger control. The implemented
`group_structured` condition removes every occurrence of each variable in a
selected co-measurement group across the full 24-hour window. It does not remove
an order event or a patient-hour co-measurement instance.

## Protocol and provenance

Five patient-stratified outer folds use PhysioNet 2012 sets a+b only (4,000 each;
n=8,000; prevalence 0.14025). Per fold, 4,800 clean model-training patients fit
the imputer, scaler, and frozen M2 model; 1,600 clean calibration patients fit
only the calibrator; 1,600 outer-test patients fit nothing. Loss is applied to
the truncated cohort before representation construction. Preprocessing is never
refitted under stress.

The repaired artifact is `cliniverse.m3/2`, generated from clean source
`df18f97e72ad389260a5a11bb5a2c708bd40f44c` (`git_dirty=false`). It stores labels,
unique record IDs, source set, fold identity, predictions, per-patient removal
counts, and per-patient/per-variable removal counts. Set-c records were not
materialized; the local cache contains no set-c record directory or archive.

## Repaired three-way result

Primary view: `values_mask` / XGBoost / clean-fitted Platt. Realized loss is the
mean fraction of eligible laboratory cells removed.

| realized loss | condition | AUROC | AP | Brier | NLL | slope | intercept | mean p |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | no loss | 0.8270 | 0.4473 | 0.0968 | 0.3151 | 0.988 | -0.010 | 0.1397 |
| 0.779 | count-matched random cell | 0.8016 | 0.4018 | 0.1022 | 0.3345 | 0.926 | +0.115 | 0.1184 |
| 0.779 | variable-matched scattered | 0.8002 | 0.3990 | 0.1049 | 0.3445 | 1.023 | +0.573 | 0.0944 |
| 0.779 | structured group | 0.8002 | 0.3990 | 0.1049 | 0.3445 | 1.023 | +0.573 | 0.0944 |

Structured and variable-matched masks and predictions are bit-identical at all
three severities. The paired Review #3 contrasts are therefore exact:

| severity | structured - variable-matched NLL | structured - variable-matched Brier |
|---:|---:|---:|
| 0.284 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 0.519 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| 0.779 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |

The historical amount-only contrasts reproduce exactly. At realized 0.519,
structured minus count-random NLL is +0.0076 [+0.0028, +0.0122] and Brier is
+0.0021 [+0.0005, +0.0036]. At realized 0.779, NLL is +0.0100 [+0.0057,
+0.0145] and Brier is +0.0027 [+0.0012, +0.0042]. These differences combine
analyte identity with whole-window group selection; they are not evidence for a
coherence effect alone.

## Variable-identity audit

The count-random mask differs in per-variable counts for 7,816, 7,838, and 6,817
patients at the three severities. Among patients with removed cells, median
patient-level total-variation distance between removed-analyte distributions is
0.714, 0.474, and 0.226. At the highest severity, structured loss removes 2,338
fewer HCT, 1,847 fewer Platelets, and 1,732 fewer WBC cells than count-random,
while removing 1,347 more pH, 1,298 more PaO2, and 1,293 more PaCO2 cells.

Descriptive clean-fold XGBoost gain ranks BUN, Bilirubin, Lactate, Platelets,
PaO2, PaCO2, and pH among the most important eligible analytes. Importance is
reported descriptively only and is not a causal decomposition.

## Risk underestimation and intercept convention

Calibration diagnostics fit
`logit(Y) = intercept + slope * logit(prediction)`. A positive intercept means
the predicted log-odds require an upward shift when interpreted jointly with the
slope. At highest structured/variable-matched loss, the intercept is +0.573,
mean predicted mortality is 0.0944, observed prevalence is 0.1403, and nine of
ten equal-mass reliability bins have observed mortality above mean prediction.
These agree on systematic mortality-risk underestimation. Generic
"overconfidence" is not used; entropy decline is not independent evidence.

## Calibration methods

At highest group loss, raw versus Platt results are:

| method | NLL | Brier | slope | intercept |
|---|---:|---:|---:|---:|
| raw | 0.3513 | 0.1061 | 0.952 | +0.570 |
| Platt | 0.3445 | 0.1049 | 1.023 | +0.573 |

Platt improves NLL by -0.0068 [-0.0079, -0.0056] and Brier by -0.0012
[-0.0014, -0.0010] at highest severity. It moves slope closer to one at each
tested group-loss severity, but does not remove the risk-level intercept drift.
This is an exploratory calibrator comparison, not a confirmatory treatment
effect.

Each isotonic fit uses 1,600 clean calibration patients and 224 deaths, yielding
17-25 distinct fitted probability steps. No highest-severity raw test prediction
falls outside clean isotonic input support, but 1,313 of 8,000 transformed
predictions land on a learned zero-probability step and are numerically clipped.
Under this clean-calibration/shifted-test protocol, isotonic has worse stress NLL
(0.3912) and slope (0.489) than Platt. This does not support the general claim
that isotonic calibration is harmful.

## Inference, determinism, and illustration

The original predeclared group-minus-count-random analysis tests NLL, Brier, and
AURC at three severities; NLL was named primary and Brier/AURC co-primary. No
multiplicity correction was predeclared, so metric/severity findings are
reported as a family and secondary calibrator/model/representation comparisons
are exploratory. The Review #3 variable-matched comparisons are post-hoc
falsification controls.

Bootstrap resampling is by patient; paired methods share identical resampled
indices and recompute the metric difference inside each replicate. Fold identity
stays attached in the NPZ. Two independent reduced end-to-end runs produced 155
bit-identical arrays (maximum absolute prediction difference 0.0), and every
full-run loss mask is generated twice and asserted identical.

The repaired illustrative case is record **142380**, a death. At requested 50%
structured loss, predicted mortality falls from 0.2355 to 0.1579; absolute error
worsens by 0.0775 versus a median eligible deterioration of 0.0778. It was chosen
from 866 eligible deaths by the declared median-deterioration rule, with no 0.5
classification threshold and no entropy criterion. It is an illustration, not
evaluation evidence.

## Corrected conclusion and M4 contract

**M3-B - FEATURE-IDENTITY EFFECT.** Whole-window removal of selected
co-measurement-group variables produces substantial mortality-risk
underestimation while discrimination degrades modestly. It is worse than equal
amount random-cell loss, but the stronger control shows no separable
co-occurrence-structure effect: analyte identity fully determines the implemented
structured mask.

M4 may proceed without implementing new event-loss experiments, but must inherit
support blindness, patient-paired random baselines, clean calibration isolation,
and probabilistic endpoints. Before results, it must declare NLL-vs-budget as
primary, Brier-vs-budget as co-primary, calibration intercept/slope as direct
secondary diagnostics, AUROC/AP as ranking diagnostics, and a single integration
rule for AUBC. M4 must not reuse the whole-window M3 mask as evidence that
coherent laboratory-event structure was tested.
