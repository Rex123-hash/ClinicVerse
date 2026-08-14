# M3 Design — Calibration Robustness Under Structured Information Loss

**Status:** HISTORICAL PREDECLARATION, amended after results by the independent review Review #3.
**Date:** 2026-08-09

The original choices remain visible below. Review #3 corrections are explicitly
labelled and must not be mistaken for predeclared analyses.

Every choice below — primary representation, hyperparameters, severity ladder, primary contrast,
patient-selection rule — is fixed here so that none of them can be chosen after seeing results.

---

## 1. Central question

> When clinically coherent groups of measurements disappear, does the model become appropriately
> less confident, or does its stated confidence hold up while its probabilistic reliability
> deteriorates?

We do **not** assume overconfidence. If confidence adapts correctly, that is the finding and it
will be reported as such (pre-declared Outcome C).

## 2. Primary predictive representation — declared in advance

**PRIMARY: `values_mask` with XGBoost.**

Reasoning, recorded before results exist:

1. **It is the representation that can see the loss.** Removing a group drives its counts to 0,
   its ever-flags to 0 and its recency to the never-observed sentinel. The model is therefore
   *explicitly told* that information is gone. If a model that can see the loss still fails to
   adjust confidence, that is a substantially stronger result than the same failure in a model
   that is structurally blind to it.
2. **It is the representation M4 will use.** Disclosure changes availability, which only
   `values_mask` can represent. Choosing it now avoids swapping representations between milestones.
3. **No performance is sacrificed.** M2 corrected: `values_mask` 0.8295 vs `values_only` 0.8279,
   difference `+0.0016 [−0.0028, +0.0059]` — not distinguishable.

**SECONDARY comparator: `values_only` with XGBoost**, and **`values_mask` with logistic
regression** as a model-class sanity check.

Per the review's warning, differences between representations with different information sets are
**not** attributed to calibration. The secondary runs test whether the primary conclusion is
specific to one representation or model class; they are not a calibration comparison.

## 3. Model configuration — frozen, not re-searched

No architecture search. XGBoost hyperparameters are frozen to the modal M2 selection for
`values_mask::xgboost`, held identical across every loss condition and severity:

```
max_depth = 5, learning_rate = 0.05, min_child_weight = 10,
subsample = 0.8, colsample_bytree = 0.8, reg_lambda = 1.0,
n_estimators = 200, tree_method = "hist", eval_metric = "logloss"
```

`n_estimators` is fixed at 200 rather than early-stopped, because early stopping would require a
validation signal whose composition changes with the stress condition. A fixed round count keeps
the model identical across conditions, which is what a robustness test requires.

Logistic regression secondary uses `C = 0.01` (modal M2 selection for mask-bearing views).

## 4. Calibration isolation — the P0 requirement

M2's calibration diagnostics were descriptive because they reused the OOF labels they were
reported on. M3 isolates calibration data explicitly.

Per outer fold (5-fold stratified over patients, sets a+b, n = 8,000):

| partition | approx n | fits imputer/scaler | fits model | fits calibrator | evaluated |
|---|---:|---|---|---|---|
| **model-train** | 4,800 | **yes** | **yes** | no | no |
| **calibration** | 1,600 | no | no | **yes** | no |
| **outer test** | 1,600 | no | no | no | **yes** |

- The imputer and scaler are fitted on **model-train only** and never see calibration or test rows.
- The calibrator is fitted on **calibration only**, using model predictions on that partition.
- The outer test partition trains nothing: not the model, imputer, scaler, calibrator, threshold
  or abstention rule.
- **set-c is never loaded.** Only sets a+b.

**The calibration partition is always CLEAN (no information loss).** This is deliberate and is the
scientifically meaningful setup: a model is calibrated on ordinary data, and *then* information
disappears. Calibrating on stressed data would answer a different and much less interesting
question.

## 5. Preprocessing under stress — frozen

The imputer is fitted **once per fold on clean model-train data** and is **not refitted** under any
stress condition. Refitting per condition would let the pipeline adapt to the stress distribution
and would defeat the purpose of a robustness test.

Imputation strategy: **median**, applied identically everywhere. The M2 imputation sensitivity
finding is treated as a closed M2 result and is not re-litigated here.

## 6. Information-loss mechanisms

Loss is applied to the **cohort** — values and observation mask — *before* feature construction, so
information genuinely disappears rather than summary columns being edited after the fact.

Eligible information is restricted to **laboratory cells**, i.e. observed cells belonging to the 23
variables covered by the empirically derived co-measurement catalogue. Vitals and settings are not
removed, because they are near-continuously monitored and their removal would not represent a
group-level acquisition event.

| condition | mechanism |
|---|---|
| **A. `none`** | No loss. Original ≤24h information. |
| **B. `cell_random`** | Remove individual observed laboratory cells uniformly at random. Unstructured diagnostic. |
| **C. `group_structured`** | Remove entire co-measurement groups (`BMP_like`, `CBC_like`, `ABG_like`, `hepatic_like`, `Lactate`, `SaO2`, `Albumin`, `TroponinT`, `TroponinI`, `Cholesterol`) — every observed cell of that group across the whole window. |

**Temporal block loss is deliberately excluded.** It would add a third mechanism whose severity is
not comparable to the other two on a per-cell basis, complicating the matched-severity design
without addressing the central question. This is a scope decision, recorded rather than silently
taken.

**Terminology, binding.** These are **structured group-level information loss** over
**co-measurement groups**. They are not verified clinical orders and must never be described as
"clinically realistic missingness".

## 7. Matched-severity design — the core control

"Group loss is worse" must not be explainable by group loss simply removing more cells. Matching is
therefore done **per patient**, not on average:

1. For a patient at severity `s`, group loss shuffles that patient's present groups with a seeded
   generator and accepts a candidate group only when adding it moves the removed-cell count
   strictly closer to the target than stopping. This is the final repaired algorithm; the original
   reach-or-exceed implementation was superseded before the retained M3 artifacts.
2. The exact number of cells removed, `N`, is recorded.
3. Matched `cell_random` loss for that same patient removes **exactly `N`** cells, drawn uniformly
   from the same eligible pool.

By construction the two conditions remove an identical number of cells from the same patient, so
the comparison controls the **amount** removed. It does not control variable identity and therefore
does not, by itself, isolate structure. Realized severity is recorded per patient for both
conditions and reported.

## 8. Severity ladder — predeclared

**0%, 25%, 50%, 75%** of each patient's eligible laboratory cells.

Chosen before any result was inspected. Group indivisibility means realized severity overshoots the
request; realized severity is therefore recorded per patient and reported as a distribution, and
the matched cell condition matches the *realized* count, not the requested one.

## 9. Metrics

Per condition × severity, on the outer test partitions pooled across folds:

- **Prediction:** AUROC, average precision (AP)
- **Probabilistic:** Brier, NLL
- **Calibration:** slope, intercept (both with paired bootstrap intervals), equal-mass reliability
  curves with bin counts
- **Selective prediction:** risk–coverage curve, AURC
- **Confidence:** mean predicted probability, mean **predictive entropy**

`predictive entropy` is named exactly that. It is **not** called epistemic uncertainty — a single
model's entropy does not decompose into epistemic and aleatoric parts.

Deep ensembles are not used. They are not required to answer this question and would add
complexity without earning it.

## 10. Primary contrast — predeclared

**Primary:** paired patient-level bootstrap of

```
GROUP_STRUCTURED − MATCHED CELL_RANDOM
```

on **NLL** (primary), and on **Brier** and **AURC** (co-primary), at each severity, with identical
patients, identical realized cell counts and identical seeds.

**Confidence response** is reported as the separate curves Δmean-predicted-probability and
Δmean-predictive-entropy versus baseline.

**No composite "reliability gap" scalar is defined.** A single number combining performance
degradation and confidence response would require an arbitrary weighting between quantities in
different units, and would be easy to tune toward a desired story. The separate curves answer the
question without that risk. This is a deliberate refusal, recorded here in advance.

## 11. Demonstration patient — selection rule declared in advance

Selected **after** aggregate analysis, by this rule and no other:

> Among outer-test patients who (a) are predicted correctly at a 0.5 threshold under `none` and
> incorrectly under `group_structured` at 50% severity, and (b) whose predicted probability moves
> **toward** the wrong class while predictive entropy fails to increase, select the patient whose
> absolute calibration-error deterioration is **nearest the median** of that eligible set.

Median, not maximum — explicitly to avoid selecting the most dramatic case. If the eligible set is
empty, that is reported as a finding and no patient is selected. Only the reproducible record ID
and condition are stored. No UI is built.

## 12. Claim rules — binding

Not permitted unless the metrics establish them:

- "the model doesn't know what it doesn't know"
- "AI remains confidently wrong" — requires *both* demonstrated predictive deterioration *and*
  demonstrated insufficient confidence response
- "clinically realistic missingness" — use "structured group-level information loss"
- any causal, deployment-utility, or clinician-intent language

Permitted framing: associational, measurement-presence and preprocessing language only.

## 13. Pre-declared outcomes

- **A (strong):** group loss degrades probabilistic performance/calibration significantly more than
  matched cell loss, and confidence adapts insufficiently.
- **B (interesting):** both degrade, but the calibration method determines robustness — calibration
  robustness becomes the finding.
- **C (null):** group loss behaves like matched cell loss and calibrated confidence tracks
  degradation appropriately. **Reported as-is.** M4 policy-ranking stability then becomes more
  important, not less.

We will not force a safety-failure story.

---

## Review #3 amendment (2026-08-10; post-result repair)

This section is a falsification amendment, not part of the 2026-08-09
predeclaration.

### Variable-matched control

Review #3 adds `variable_matched_scattered`. For each patient it requests the
same number of removed cells from every variable as that patient's
`group_structured` realization, uses no labels, and samples timestamps
independently within each variable. Any naturally unavailable shortfall is
deterministically clipped and reported as a mismatch.

The implementation audit established that `group_structured` removes every
occurrence of each member variable of a selected co-measurement group across the
entire truncated window. It does not remove a laboratory-order event,
patient-hour group instance, or subset of repeated measurements. Consequently,
exact per-variable matching has no scattering freedom: matching a variable for
which all observed cells were removed necessarily selects those same cells. A
mask-identical control is retained and reported because it proves that the
original group-versus-cell contrast cannot identify a coherence or
co-occurrence-structure effect separately from analyte identity.

The repaired three-way comparison is:

1. `group_structured`;
2. `variable_matched_scattered` (identity- and amount-matched; expected to be
   mask-identical under the audited semantics);
3. `cell_random` (amount-matched only).

NLL and Brier are the Review #3 decision metrics. The new variable-matched
contrasts are post-hoc falsification controls, not original confirmatory tests.
The original group-versus-cell NLL, Brier, and AURC contrasts retain their
historical predeclared status, but comprise nine severity-by-metric comparisons;
calibrator, secondary-representation, and model-class comparisons are
exploratory. No multiplicity-adjusted claim is made.

### Demonstration-patient rule

The original 0.5-threshold/entropy rule is superseded because 0.5 is arbitrary
for a 14% endpoint and entropy decline is mechanically coupled to probabilities
moving toward zero. The repaired illustrative rule is:

> At requested 50% structured loss, among outcome-positive outer-test patients
> whose predicted mortality risk decreases and whose absolute probability error
> worsens, select the record nearest the median deterioration; break exact ties
> by lowest record ID.

Labels are used only for this declared post-hoc illustration. The selected case
is not evaluation evidence.
