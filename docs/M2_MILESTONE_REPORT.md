# M2 Milestone Report — Corrected Representation Ablation

**Review date:** 2026-08-10
**Task:** T1 in-hospital mortality at a 24-hour information boundary
**Cohort:** PhysioNet/CinC 2012 sets a+b, n=8,000, mortality prevalence 14.025%; set-c excluded
**Decision:** PASS M2 WITH NONBLOCKING RISKS

The canonical values are in `experiments/baselines/results/m2/results.json`; raw aligned OOF
predictions, labels, and record IDs are in `predictions.npz`. The NPZ SHA-256 is recorded in the
manifest. Review #2 independently recalculated every AUROC, average precision, Brier score, and
NLL from the NPZ and found maximum absolute disagreement of `1.4e-17`.

## Corrected protocol

Five stratified outer folds give each patient one OOF prediction. Within each 6,400-patient outer
training fold, a stratified 5,120/1,280 inner split selects hyperparameters. The inner imputer and
scaler see only the 5,120 inner-training patients. After selection, preprocessing is refitted on
all 6,400 outer-training patients. Logistic regression is fitted there; XGBoost uses the
inner-selected boosting-round count and is also refitted on all 6,400 patients. The 1,600 outer
test patients are used only once for prediction.

This corrects two defects in the superseded M2 run: preprocessing had been fitted before the inner
split, and the final XGBoost model had remained fitted on only the inner-training subset despite a
comment claiming a full outer-training refit. Outer-test isolation was intact, but these defects
made the previous XGBoost representation comparison unreliable.

| Provenance | Value |
|---|---|
| executable git SHA | `ae7fbb8` (clean tree) |
| cohort fingerprint | `f59c44f07556b7a6…` |
| split hash | `21cbeab1b5bc308f…` |
| seed / bootstrap | 20260809 / 2,000 patient resamples |
| runtime | Python 3.12.13; NumPy 2.5.2; scikit-learn 1.9.0; XGBoost 3.4.0 |

XGBoost means `XGBClassifier` with binary-logistic objective, no class reweighting
(`scale_pos_weight=1`), depth `{3,5}`, learning rate `{0.05,0.1}`, minimum child weight `{1,10}`,
subsample and column-subsample 0.8, L2=1, histogram trees, and seed per fold. Early stopping is
used only on the inner split to choose the round count; the fixed count is then refitted without
outer-test inspection.

## Corrected M2 table

“AP” is scikit-learn `average_precision_score`, not trapezoidal PR-AUC.

| run | features | AUROC [95% CI] | AP | Brier | NLL | descriptive slope / intercept |
|---|---:|---:|---:|---:|---:|---:|
| prevalence reference | 0 | 0.5000 [0.5000, 0.5000] | 0.1403 | 0.1206 | 0.4054 | undefined |
| LR mask-only | 113 | 0.7278 [0.7128, 0.7423] | 0.2783 | 0.1114 | 0.3657 | 1.000 / −0.003 |
| XGBoost mask-only | 113 | 0.7319 [0.7169, 0.7457] | 0.2812 | 0.1111 | 0.3634 | 1.032 / +0.054 |
| LR values-only | 185 | 0.8095 [0.7964, 0.8219] | 0.4273 | 0.0997 | 0.3276 | 0.979 / −0.037 |
| XGBoost values-only | 185 | 0.8279 [0.8162, 0.8395] | 0.4471 | 0.0970 | 0.3151 | 1.003 / +0.060 |
| LR values+mask | 298 | 0.8240 [0.8121, 0.8359] | 0.4511 | 0.0969 | 0.3182 | 0.965 / −0.057 |
| XGBoost values+mask | 298 | 0.8295 [0.8177, 0.8411] | 0.4502 | 0.0968 | 0.3141 | 0.996 / +0.068 |

Supplementary models used the identical protocol: statics-only AUROC is 0.6310 (LR) and 0.6777
(XGBoost); values+mask+statics reaches 0.8309 and 0.8425 respectively. These are official
comparators, not part of the binding three-way contrast.

## Paired comparisons

| comparison | LR ΔAUROC [95% CI] | XGBoost ΔAUROC [95% CI] |
|---|---:|---:|
| values+mask − values-only | +0.0146 [+0.0089, +0.0204] | +0.0016 [−0.0028, +0.0059] |
| values-only − mask-only | +0.0817 [+0.0655, +0.0975] | +0.0960 [+0.0819, +0.1104] |
| values+mask − mask-only | +0.0962 [+0.0828, +0.1095] | +0.0976 [+0.0837, +0.1111] |

The prior XGBoost explicit-mask claim (`+0.0090`) does not survive repair. The corrected estimate
is small and its paired interval includes zero. The LR gain remains statistically distinguishable.

## Imputation diagnostics

Median-jitter (1% train-derived IQR) breaks exact equality to the median while staying local.
Empirical-marginal imputation samples every missing summary cell independently from that column’s
outer-training observations. It is not a realistic multivariate imputer: 74.1% of imputed
five-summary groups violate elementary last/mean/min/max ordering constraints.

| values-only comparison | LR ΔAUROC [95% CI] | XGBoost ΔAUROC [95% CI] |
|---|---:|---:|
| median − median-jitter | +0.0014 [+0.0002, +0.0028] | +0.0064 [+0.0025, +0.0104] |
| median − empirical-marginal | +0.0194 [+0.0142, +0.0247] | +0.0161 [+0.0105, +0.0220] |

A fold-honest classifier attempted to recover whether each source variable was originally absent
from its five post-imputation summaries. Median and median-jitter were almost perfectly
reconstructible (micro AUROC 0.9994 and 1.0000); empirical-marginal remained highly reconstructible
(0.9808). Thus values-only is not missingness-free under any tested control. However, because the
empirical control both retains missingness cues and destroys joint structure, its mortality gap
does **not** quantify how much median-coded absence helps prediction. The former `+0.0145` causal
interpretation is withdrawn.

## Calibration and baseline interpretation

Slope/intercept are maximum-likelihood fits of `label ~ intercept + slope × logit(OOF prediction)`
after probability clipping. They are fitted and reported on the same aggregate OOF labels and are
therefore descriptive diagnostics, not unbiased calibration estimates or post-hoc calibration.
For mask-only LR, aggregate 1.000/−0.003 hides fold slopes from 0.884 to 1.168 and intercepts from
−0.180 to +0.214. Reliability plots use OOF predictions and equal-mass bins with counts retained.

The prevalence line is a constant global development-prevalence reference. AUROC is therefore 0.5
by definition and AP equals prevalence. The superseded fold-specific constants created artificial
cross-fold ranking and AUROC 0.4994.

E-002 used fixed `C=1`; its retained artifact reports 0.7223745892. Re-executing that current
pipeline produced 0.7222844 (a small environment/artifact drift), whereas nested M2 selected
`C={0.01,0.01,0.1,0.01,0.01}` and produced 0.7277881. E-002 is qualitatively replicated; M2 is a
tuned comparable estimate, not an exact reproduction.

## Mask decomposition and interpretation

The fold-honest LR diagnostic gives AUROC 0.6898 for ever-measured flags, 0.7059 for counts and
frequency, 0.6802 for recency, 0.7110 for ever+counts, and 0.7278 for all mask components. The
signal is distributed, with frequency the strongest single component. Statics-only is weaker,
though this comparison does not establish causation or clinician intent.

The supported reading is: physiological summaries dominate mask-only features; measurement
presence alone remains predictive; explicit mask features add a reliable small LR gain but no
reliable XGBoost gain after repair; and preprocessing can retain missingness cues even without
explicit indicators. This supports Thesis D most strongly (evaluation conclusions are sensitive
to nesting, representation, model refit, and imputation), with Thesis B as a secondary summary.

## Artifacts and next gate

`results.json` now includes all feature names, per-feature inventories, per-fold selections and
calibration/reliability diagnostics, package versions, hashes, seeds, cutoff, and source sets. Raw
OOF predictions and mask-decomposition OOF predictions are checksummed. Diagnostic results are in
`imputation_diagnostics.json` and `mask_decomposition.json`.

M3 may proceed only after checkpoint sign-off. The recommended M3 is calibration robustness under
structured group-level information loss, with an isolated calibration split and discrimination,
proper-score, reliability, and risk-coverage reporting. M4 should later test acquisition-policy
ranking stability across support protocol, masking mechanism, and cost regime. This report does
not begin either milestone.
