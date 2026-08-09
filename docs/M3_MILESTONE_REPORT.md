# M3 Milestone Report — Calibration Robustness Under Structured Information Loss

**Date:** 2026-08-09
**Design:** predeclared in [`M3_DESIGN.md`](M3_DESIGN.md), committed before any experiment ran.
**Recommendation:** **GO to M4** (§12)

All numbers are read from `experiments/robustness/results/m3/results.json`. Raw and calibrated
predictions are retained in `predictions.npz` for independent recomputation.

> **Reproducibility note, recorded because it happened.** An intermediate lint auto-fix altered
> behaviour in `information_loss.py` part-way through this milestone, so an earlier draft table
> differed from the final one. Every figure below comes from the final code, which was then
> verified deterministic: two consecutive runs produce **bit-identical** predictions
> (max difference 0.000e+00 across all conditions). Only the final numbers are reported.

---

## 1. Calibration protocol (exact)

Per outer fold — 5-fold stratified over patients, sets a+b, n = 8,000, prevalence 14.03%:

| partition | n | fits imputer/scaler | fits model | fits calibrator | evaluated |
|---|---:|---|---|---|---|
| model-train | 4,800 | **yes** | **yes** | no | no |
| calibration | 1,600 | no | no | **yes** | no |
| outer test | 1,600 | no | no | no | **yes** |

- The imputer (median) is fitted **once per fold on clean model-train data** and is **never
  refitted under stress**, so the pipeline cannot adapt to the stress distribution.
- The calibration partition is **always clean**. The question is what happens when a
  normally-calibrated model meets information loss.
- The outer test partition trains nothing at all.
- **set-c was never loaded.** Provenance records sets `[a, b]`, excluded `[c]`.

Provenance: git `bff052a6168d`, cohort `f59c44f07556b7a6`, split `21cbeab1b5bc308f`,
calibration split `bf7584e60636875c`, config `2d22974f8daf179d`, seed 20260809.

**Model frozen from M2, not re-searched:** XGBoost `max_depth=5, learning_rate=0.05,
min_child_weight=10, n_estimators=200, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0`.
`n_estimators` is fixed rather than early-stopped, because a stopping signal whose composition
changes with the stress condition would make the model itself condition-dependent.

**Primary representation, declared in advance:** `values_mask` + XGBoost — chosen because it is
the representation that *can see* the loss (counts fall to zero, ever-flags clear, recency hits its
sentinel), because M4's disclosure benchmark needs it, and because M2 showed no performance cost.

## 2. Loss mechanisms

Loss is applied to the **cohort** — values and mask — before feature construction, so information
genuinely disappears. Eligible information is the 23 laboratory variables in the co-measurement
catalogue; vitals and ventilator settings are never removed.

- **`cell_random`** — individual observed laboratory cells removed uniformly at random.
- **`group_structured`** — entire co-measurement groups removed (`BMP_like`, `CBC_like`,
  `ABG_like`, `hepatic_like`, `Lactate`, `SaO2`, `Albumin`, `TroponinT`, `TroponinI`,
  `Cholesterol`): every observed cell of that group, across the whole window.

Temporal block loss was excluded, as declared, to keep severity comparable on a per-cell basis.

**These are structured group-level information loss conditions over co-measurement groups. They
are not verified clinical orders and are not "clinically realistic missingness".**

## 3. Realized severity

Groups are indivisible, so realized severity cannot equal the request. A first implementation
removed groups until the target was reached, which turned a requested 25% into a realized 46%;
it was replaced by a rule that accepts a group only when it lands *nearer* the target than
stopping.

| requested | realized mean | realized median | cells removed | matched? |
|---|---|---|---|---|
| 0.25 | **0.284** | 0.273 | 86,192 | yes — identical in both conditions |
| 0.50 | **0.519** | 0.511 | 157,625 | yes |
| 0.75 | **0.779** | 0.763 | 231,414 | yes |

Matching is **per patient**, and a test asserts it: `group.removed_cells == cell.removed_cells`
elementwise. "Group loss is worse" therefore cannot be explained by group loss removing more.
7,920 of 8,000 patients have any eligible laboratory cells.

## 4. Performance and calibration — primary (values_mask / XGBoost / Platt)

| severity | condition | AUROC | AP | Brier | NLL | AURC | slope | intercept | mean p | entropy |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.00 | none | 0.8270 | 0.4473 | 0.0968 | 0.3151 | 0.0354 | 0.988 | −0.010 | 0.1397 | 0.3129 |
| 0.284 | cell_random | 0.8226 | 0.4376 | 0.0979 | 0.3183 | 0.0362 | 0.976 | −0.041 | 0.1411 | 0.3163 |
| 0.284 | **group** | 0.8204 | 0.4339 | 0.0986 | 0.3212 | 0.0369 | 1.007 | **+0.204** | 0.1224 | 0.2944 |
| 0.519 | cell_random | 0.8151 | 0.4273 | 0.0991 | 0.3231 | 0.0380 | 0.954 | −0.020 | 0.1358 | 0.3107 |
| 0.519 | **group** | 0.8118 | 0.4142 | 0.1012 | 0.3308 | 0.0396 | 1.010 | **+0.360** | 0.1094 | 0.2792 |
| 0.779 | cell_random | 0.8016 | 0.4018 | 0.1022 | 0.3345 | 0.0410 | 0.926 | +0.115 | 0.1184 | 0.2896 |
| 0.779 | **group** | 0.8002 | 0.3990 | 0.1049 | 0.3445 | 0.0418 | 1.023 | **+0.573** | 0.0944 | 0.2596 |

**Reading this table.** Discrimination barely moves: AUROC falls 0.8270 → 0.8002 across a 78%
loss of laboratory information, and group and cell loss are nearly indistinguishable on AUROC.
The calibration intercept behaves completely differently: it is flat under random-cell loss
(−0.010 → +0.115) and rises steeply under structured group loss (−0.010 → **+0.573**). Mean
predicted risk falls from 0.1397 to 0.0944 while true prevalence is unchanged at 0.1403.

A positive intercept with a slope near 1 means predictions need shifting **upward**: under
structured loss the model systematically **understates** risk.

## 5. Paired primary contrast — group minus matched cell

Identical patients, identical realized cell counts, identical seeds. 2,000 patient-level
bootstrap resamples. Positive = group loss is worse.

| calibrator | severity | NLL | Brier | AURC |
|---|---|---|---|---|
| Platt | 0.284 | +0.0029 [−0.0005, +0.0065] | +0.0007 [−0.0005, +0.0019] | +0.0006 [−0.0005, +0.0018] |
| Platt | 0.519 | **+0.0076 [+0.0028, +0.0122]** | **+0.0021 [+0.0005, +0.0036]** | **+0.0016 [+0.0002, +0.0031]** |
| Platt | 0.779 | **+0.0100 [+0.0057, +0.0145]** | **+0.0027 [+0.0012, +0.0042]** | +0.0008 [−0.0006, +0.0024] |
| uncalibrated | 0.519 | **+0.0105 [+0.0054, +0.0155]** | **+0.0026 [+0.0008, +0.0042]** | **+0.0018 [+0.0003, +0.0033]** |
| uncalibrated | 0.779 | **+0.0129 [+0.0083, +0.0177]** | **+0.0031 [+0.0016, +0.0048]** | +0.0009 [−0.0005, +0.0025] |

Structured loss is significantly worse than matched random loss on **NLL and Brier at 50% and 75%
severity**, for both uncalibrated and Platt-calibrated models. At 25% the difference is in the same
direction but not distinguishable. **AURC separates only at 50%** — selective prediction is largely
insensitive to the distinction, which is a real limitation of AURC as a stress readout here.

## 6. Risk–coverage / AURC

AURC degrades modestly and monotonically with severity (0.0354 → 0.0418 under group loss;
0.0354 → 0.0410 under cell loss) and does **not** reliably distinguish the two conditions. The
confidence *ordering* remains useful even as the probability *level* drifts: the model still knows
which patients are higher risk, it just no longer knows how high.

## 7. Does calibration mitigate it? Partly — and not the part that matters

| calibrator | intercept at 0.779 group loss | slope | NLL |
|---|---|---|---|
| uncalibrated | +0.570 | 0.952 | 0.3513 |
| **Platt (clean-fitted)** | **+0.573** | 1.023 | **0.3445** |
| isotonic | −0.455 (sign flips, unstable across severities) | 0.489 | 0.3912 |

Platt improves NLL and keeps the slope near 1, but leaves the intercept drift essentially
untouched — it was fitted on clean data and cannot anticipate a shift that has not happened yet.
**Isotonic is actively harmful under stress**: slope collapses to ~0.5 and NLL is worst at every
severity, i.e. the more flexible calibrator is the less robust one.

So calibration method matters (pre-declared Outcome B), but no clean-data calibrator prevents the
risk-level drift caused by structured information loss.

## 8. Secondary representations — the finding is not an artifact of one choice

| run | intercept at baseline → 0.779 group loss | AUROC baseline → 0.779 group |
|---|---|---|
| **values_mask / XGBoost** (primary) | −0.010 → **+0.573** | 0.8270 → 0.8002 |
| values_only / XGBoost | +0.010 → **+0.606** | 0.8229 → 0.7955 |
| values_mask / logistic regression | −0.004 → **+0.590** | 0.8162 → 0.7726 |

The drift appears in all three, at similar magnitude, so it is not specific to the mask block or to
gradient boosting. Per the review's warning, these are **not** compared to attribute differences to
calibration; they are a robustness check on the conclusion.

## 9. Does confidence adapt? Direction, and an honest caveat

Mean predictive entropy **falls** under structured loss (0.3129 → 0.2596), i.e. the model becomes
nominally *more* decisive as it loses information and becomes less reliable.

**Caveat that must not be dropped.** At a base rate of 14%, entropy is monotone increasing in `p`
below 0.5, so an entropy decrease is a *mechanical consequence* of predicted risks falling toward
zero. The entropy drop is therefore **not independent evidence** of increased confidence.

The independent evidence is:
1. probabilistic performance deteriorates (NLL +0.029, Brier +0.008 from baseline to 78% group
   loss), and
2. the calibration intercept moves to +0.573 while slope stays ~1 — a systematic, direction-specific
   risk understatement, roughly **5× the drift** under matched random-cell loss.

## 10. Demonstration patient (selection rule applied mechanically)

136 outer-test patients flip from correct to incorrect under 50% group loss; 88 of those also show
no entropy increase. Of those 88, the patient nearest the **median** deterioration was selected —
median, not maximum, exactly as declared:

| field | value |
|---|---|
| record id | **137856** |
| true outcome | died |
| p(death), no loss | **0.597** (correct at 0.5) |
| p(death), 50% group loss | **0.375** (now wrong) |
| predictive entropy | 0.674 → 0.662 (did not rise) |
| deterioration | 0.222 (median of eligible: 0.226) |

Stored as `results/m3/m3_demo_patient.json`. No UI built.

## 11. Claim discipline

**Supported by these metrics:**
- Under structured group-level information loss, probabilistic reliability degrades significantly
  more than under matched random-cell loss (NLL, Brier; 50% and 75% severity).
- The model's risk estimates drift systematically downward, understating risk, while its ranking
  ability is largely preserved.
- Calibration fitted on clean data corrects slope but not this drift; isotonic makes it worse.

**Not claimed:** that "the model doesn't know what it doesn't know" (entropy evidence is
confounded, §9); that this is clinically realistic missingness; any causal or deployment claim.

## 12. Decision gate

**Verdict: pre-declared Outcome A, qualified by Outcome B.** Structured loss causes significantly
greater probabilistic and calibration degradation than matched random loss, and confidence does not
adapt in the protective direction — but the strongest evidence is the calibration-intercept drift,
not entropy, and the effect is absent at the lowest severity.

**Recommendation: GO to M4.**

M4 becomes *more* important, not less. M3 shows that a fixed evaluation protocol already hides a
large reliability failure behind a nearly-flat AUROC curve. If discrimination is this insensitive
to information loss, then acquisition-policy rankings scored on discrimination may be similarly
insensitive — or unstable for reasons unrelated to acquisition quality. M4 should therefore rank
policies on **probabilistic** and **calibration** metrics, not AUROC alone.

Carry into M4: the isolated three-way partition; the frozen model; matched-severity loss; and the
finding that clean-data calibration does not survive structured loss.

## 13. Limitations

- One dataset, one cutoff, one split assignment, no external validation.
- Synthetic deletion is **not** natural missingness and is not deployment shift.
- The 25% severity contrast is not distinguishable; conclusions rest on 50% and 75%.
- AURC barely separates the conditions — reported as a negative result on that metric.
- Single model, no ensembles; entropy from one model is not decomposable.
- Aggregate calibration slope/intercept are pooled over outer test folds; per-fold variability is
  retained in the artifact but not summarised here.

## 14. Artifacts

```
experiments/robustness/results/m3/
  results.json          metrics, contrasts, severity report, provenance
  predictions.npz       raw + calibrated predictions, labels, record ids
  m3_demo_patient.json  selection rule, eligible counts, chosen case
  figures/m3_degradation.png     headline: what degrades and what does not
  figures/m3_calibrators.png     calibrator comparison under group loss
  figures/m3_reliability.png     reliability at highest severity
```

```bash
python experiments/robustness/m3_calibration_under_loss.py --n-boot 2000
python experiments/robustness/m3_figures.py
```
