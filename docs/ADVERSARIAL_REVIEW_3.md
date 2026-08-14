# the independent review Repair #3 - M3 Structured-Loss Audit

## Executive Verdict

**PASS M3 WITH NONBLOCKING RISKS. M3-B - FEATURE-IDENTITY EFFECT.**

The frozen M3 metrics reproduce, but the original structural interpretation
does not. The implemented structured mechanism deletes all 24-hour occurrences
of selected analytes. An exact per-patient/per-analyte control is therefore
mask-identical, producing exact zero NLL and Brier differences. The valid result
is risk underestimation under whole-window selected-analyte loss, not a separable
coherence effect.

## M3 Metric Reproduction

Starting state was clean at `9d78ed38740f6151070a3d6f34b919d4c50f2735`.
M3 implementation/artifacts were introduced by `bff052a`; the milestone report
was added by `9d78ed3`. All 288 original tests passed.

Independent NumPy/scikit-learn scoring of raw NPZ predictions reproduced every
requested primary metric to floating-point equality:

| condition | AUROC | AP | Brier | NLL | slope | intercept | mean p | AURC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no loss | 0.8270 | 0.4473 | 0.0968 | 0.3151 | 0.988 | -0.010 | 0.1397 | 0.0354 |
| 0.779 count-random | 0.8016 | 0.4018 | 0.1022 | 0.3345 | 0.926 | +0.115 | 0.1184 | 0.0410 |
| 0.779 structured | 0.8002 | 0.3990 | 0.1049 | 0.3445 | 1.023 | +0.573 | 0.0944 | 0.0418 |

Independent patient bootstrap reproduced the two major NLL contrasts:
+0.007620 [+0.002846, +0.012241] at requested 0.50 and +0.010049
[+0.005720, +0.014532] at requested 0.75.

The original artifact had `git_dirty=true`, a provenance defect. Repaired schema
v2 artifacts point to clean source `df18f97e72ad389260a5a11bb5a2c708bd40f44c`
with `git_dirty=false`.

## Split / Calibration Audit

One traced outer fold follows the promised isolation:

1. 4,800 clean model-training rows fit the median imputer, logistic scaler where
   applicable, and frozen XGBoost/logistic model.
2. 1,600 disjoint clean calibration rows are transformed by train-fitted
   preprocessing and fit only identity, Platt, and isotonic calibrators.
3. 1,600 disjoint outer-test rows are transformed and scored; nothing is fit.

XGBoost uses fixed 200 rounds and no early stopping. No outer-test distribution
statistic defines inputs. Prediction artifacts now retain fold ID for every row.

## Information-Loss Audit

Actual order is truncated cohort -> observation loss -> representation ->
train-fitted imputer/scaler -> frozen model -> clean-fitted calibrator. Values
are set to NaN and observation-mask cells to false before summary features are
built. There is no post-hoc feature deletion.

`group_structured` selects a co-measurement feature group per patient and removes
every observed cell of every member variable at every eligible timestamp in the
24-hour window. Repeated measurements are all removed. A partially observed
group removes all naturally present member cells and invents none. These are not
verified orders, specimens, or patient-hour laboratory events.

Final severity selection accepts a shuffled candidate group only if adding it is
strictly closer to the target than stopping. Tests cover severity 0/1, invalid
values, no eligible cells, one oversized/only group, repeated measurements, and
matching fallback. Final realized means remain 0.284, 0.519, and 0.779; the
superseded 0.463/0.711 values are not used.

## Variable-Identity Confounding Audit

Equal total cell counts do not equalize analyte identity. Count-random differs
from structured per-variable counts for 7,816, 7,838, and 6,817 patients across
the three severities. Median within-patient total-variation distances are 0.714,
0.474, and 0.226. Aggregate removed-variable TV distances are 0.197, 0.033, and
0.035.

At highest severity the largest count differences (structured minus random) are
HCT -2,338, Platelets -1,847, WBC -1,732, pH +1,347, PaO2 +1,298, and PaCO2
+1,293. Clean-fold XGBoost gain importance ranks several affected analytes highly
(BUN, Bilirubin, Lactate, Platelets, PaO2, PaCO2, pH); this is descriptive only.

## Variable-Matched Control

`VARIABLE_MATCHED_SCATTERED` is deterministic, patient-matched, seed-family
matched, label-free, and operates on raw observations. It requests exactly the
structured count for each corresponding variable and samples only naturally
observed timestamps. Unavailable requests are clipped deterministically and
reported; the M3 run had zero mismatched patients/cells.

Because structured loss already removes all timestamps of each selected
variable, exact per-variable matching has zero degrees of freedom. All three
severity masks are bit-identical to structured loss. This degeneracy is the
scientific result of the stronger control, not an implementation failure.

## Corrected Three-Way Results

Highest severity, primary Platt run:

| condition | AUROC | AP | Brier | NLL | intercept | slope | mean p |
|---|---:|---:|---:|---:|---:|---:|---:|
| count-matched random cell | 0.8016 | 0.4018 | 0.1022 | 0.3345 | +0.115 | 0.926 | 0.1184 |
| variable-matched scattered | 0.8002 | 0.3990 | 0.1049 | 0.3445 | +0.573 | 1.023 | 0.0944 |
| structured group | 0.8002 | 0.3990 | 0.1049 | 0.3445 | +0.573 | 1.023 | 0.0944 |

Structured minus variable-matched is NLL 0.0000 [0.0000, 0.0000] and Brier
0.0000 [0.0000, 0.0000]. Structured minus count-random remains NLL +0.0100
[+0.0057, +0.0145] and Brier +0.0027 [+0.0012, +0.0042].

## Calibration-Intercept Audit

Implementation fits `logit(Y) = a + b*logit(p)`. Positive `a` means an upward
log-odds correction is required, interpreted jointly with slope `b`. At highest
structured loss, +0.573 agrees with mean p 0.0944 versus prevalence 0.1403 and
with nine of ten reliability bins lying above the identity line. The correct
direct wording is systematic mortality-risk underestimation/calibration drift.

## Platt Audit

At highest structured loss, raw -> Platt changes NLL 0.3513 -> 0.3445, Brier
0.1061 -> 0.1049, slope 0.952 -> 1.023, and intercept +0.570 -> +0.573. Paired
Platt-minus-raw intervals are -0.0068 [-0.0079, -0.0056] for NLL and -0.0012
[-0.0014, -0.0010] for Brier. Platt moves slope closer to one at every tested
severity and improves proper scores, but does not remove intercept/risk-level
drift. The comparison is exploratory.

## Isotonic Audit

Each clean calibration partition has n=1,600 and 224 deaths. Folds fit 17-25
distinct probability steps (33-50 thresholds). At highest stress, no raw test
prediction is outside clean isotonic input support; extrapolation clipping is
not the observed cause. However, 1,313 predictions map to a learned zero step
and are clipped to epsilon for finite NLL. Under this exact clean-calibration /
shifted-test protocol, isotonic produces NLL 0.3912 and slope 0.489 versus
Platt's 0.3445 and 1.023. No general claim that isotonic is harmful is supported.

## Bootstrap / Multiplicity Audit

Bootstrap indices resample patients, use identical indices for paired methods,
retain fold attachment in the NPZ, and recompute metric differences inside each
replicate. Seed is fixed at 20260809; single-class replicates are explicitly
skipped. Exact-equality pairs return the mathematically exact zero interval while
still counting valid resamples.

The original predeclaration named NLL primary and Brier/AURC co-primary at each
of three severities, a nine-comparison family without a predeclared correction.
It did not make every calibrator/model/representation contrast confirmatory.
Those are exploratory. Review #3 variable-matched comparisons are post-hoc
falsification controls. No retroactive multiplicity claim is made.

## Determinism Audit

Every full-run loss variant is generated twice and compared before modeling.
Unit tests pin group, count, and variable-matched semantics. Two independent
reduced end-to-end runs produced all 155 NPZ arrays bit-identically, with maximum
absolute numeric difference 0.0. This protects scientific semantics from future
format/lint rewrites.

## Demo-Patient Audit

The old 0.5-threshold/entropy rule was rejected. The repaired rule selects, at
requested 50% structured loss, the outcome-positive patient whose downward risk
shift and worsened absolute probability error are nearest the eligible median;
ties use lowest record ID. Record **142380** is selected from 866 eligible deaths:
p(death) 0.2355 -> 0.1579; error deterioration 0.0775 versus median 0.0778. It is
explicitly post-hoc and illustrative.

## Claim Repairs

Removed or narrowed: "structure matters beyond amount," "causes," generic
"overconfidence," "isotonic is actively harmful," clinical-panel/event wording,
and the old demo threshold story. The reliability figure's caption was also
corrected: observed-above-predicted points lie above, not below, the diagonal.

## Corrected Headline

Under whole-window removal of selected co-measurement-group variables,
mortality discrimination degrades modestly while predicted risk shifts downward
and calibration deteriorates. The excess degradation versus equal-count random
cell loss is attributable to analyte identity under the implemented semantics;
no separate coherence effect is identified.

## Remaining Risks

- One historical ICU dataset, one cutoff, one split assignment, no external validation.
- Synthetic deletion is not natural missingness or deployment shift.
- Whole-window variable deletion cannot test order-event coherence.
- The stronger control was added after original results and is a falsification analysis.
- Direct calibration diagnostics are pooled; fold IDs remain available.
- Isotonic behavior is protocol-specific and includes many clipped zero steps.

## M3 Classification

**M3-B - FEATURE-IDENTITY EFFECT.** Group versus exact variable-matched difference
fully disappears. M3 is still useful evidence of calibration fragility under
selected-analyte information loss, but not evidence for structure alone.

## M4 Contract

M4 must inherit the repaired support-blind protocol, fair patient-paired random
baselines, clean-only calibration architecture, and probabilistic endpoints.
Before results, declare NLL-vs-budget primary; Brier-vs-budget co-primary;
intercept/slope drift direct secondary diagnostics; AUROC/AP ranking secondary;
and one AUBC integration rule. Do not choose the headline metric after observing
policy rankings, and do not call the current whole-window mask an event-structure
experiment.

## Quality Gate

- `pytest`: 298 passed.
- `ruff check .`: clean.
- `ruff format --check .`: clean.
- strict MyPy across `cliniverse twinbench experiments tests scripts`: clean.
- `git diff --check`: clean.
- Independent raw-artifact rescore: exact agreement.
- Final clean-tree and commit checks are recorded in the handoff.

## Final Recommendation

**PASS M3 WITH NONBLOCKING RISKS. M4 SAFE: YES, only under the M4 contract above.**
Do not use M3 to claim coherent event loss is worse than analyte-matched scattered
loss. Do not begin acquisition-policy experiments until this repaired conclusion
is the live source of truth.
