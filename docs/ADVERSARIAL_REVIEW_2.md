# Cliniverse the independent review Review + Repair #2

# Executive Verdict

**PASS M2 WITH NONBLOCKING RISKS.** M2 had a real nested-preprocessing defect and a misleading
XGBoost final-refit implementation. After repair and full rerun, mask-only discrimination and the
large values-only advantage survive. The former XGBoost claim that explicit masks add `+0.0090`
AUROC does not: the corrected difference is `+0.0016 [−0.0028, +0.0059]`. The former interpretation
of the stochastic-imputation gap as an estimate of imputation-coded missingness is also rejected.

Severity: **P1**, not P0. The outer test fold never influenced tuning or preprocessing, but inner
validation rows did influence the preprocessing used for inner model selection, and the final
XGBoost fit did not use all outer-training rows as documented.

# Metric Reproduction

Before changing code, I loaded the superseded raw M2 NPZ and independently recalculated every
headline AUROC, average precision, Brier score, and NLL with scikit-learn. All matched its JSON to
floating-point tolerance. I also reproduced the two required old paired-bootstrap intervals.
Thus the old report faithfully described the old predictions; the problem was how those
predictions were produced and interpreted.

After repair, the same independent calculation was repeated. The new archive contains 8,000
finite predictions per run, 8,000 unique record IDs, exact alignment to the loader’s label and
record order, mortality prevalence 0.14025, and source sets exactly `{a,b}`. The maximum absolute
metric disagreement is `1.4e-17`. The manifest records NPZ SHA-256
`bf8ca767838ae94b0801405a185d974c7169c51462b86afbb8b5c907adda1c3c`; the final run records clean
source SHA `ae7fbb818491ed9bfbe62b84f2dc37213cde9715`.

# CV/Nesting Audit

The superseded runner fitted an imputer on the complete outer-training fold before making the
inner split. Consequently, inner-validation feature medians/pools influenced hyperparameter
selection. The outer test remained untouched. The runner also claimed to refit XGBoost on full
outer training but actually fitted only the 80% inner-training subset while using inner validation
for early stopping.

The repaired representative fold is:

| partition | n | used to fit preprocessing? | used to select parameters? | used in final fit? |
|---|---:|---|---|---|
| outer train | 6,400 | final imputer/scaler only | contains inner split | yes, all rows |
| inner train | 5,120 | inner imputer/scaler | trains candidates | included in final fit |
| inner validation | 1,280 | no | scores grid and stopping round | included only in final refit |
| outer test | 1,600 | no | no | no; prediction only |

The runner records these counts per fold. A regression test spies on imputer-fit sizes and requires
`[80,100]` in a 100-row unit example, preventing the original leakage pattern. Feature summaries
are deterministic per patient after the cohort is truncated to 24 hours; there is no global fitted
statistic in feature construction and no cache path in the runner.

# Representation Audit

The artifact now contains a machine-readable inventory for every feature with source variable,
statistic, numeric/presence role, count/recency/time-since-last flags, sentinel use, imputation
requirement, and cutoff safety.

| view | exact contents | explicit presence | implicit presence |
|---|---|---|---|
| mask-only (113) | 38 observation counts (37 variables + total), 37 ever flags, 37 recencies, one distinct-variable total | yes | not applicable |
| values-only (185) | last, mean, min, max, slope for each of 37 variables | no | yes: absent variables yield NaNs that a fold imputer replaces |
| values+mask (298) | union of the above | yes | yes |

Recency uses `−1` for never observed. Value features have no explicit indicator or fixed sentinel
before fitting, but the positions imputed are determined by measurement absence. Therefore
“values-only” means no explicit presence columns, not no missingness information.

# Imputation Audit

The old “stochastic” control sampled one independent empirical draw per missing cell and per
summary column from training-fold marginals. It was fold-honest and seeded, but it did not reuse a
patient- or variable-level draw and was not conditional on other summaries. It can combine a last,
mean, minimum, maximum, and slope that never co-occurred and may violate their mathematical
ordering. The new diagnostic measures a 74.1% coherence-violation rate among imputed groups for
this control. It therefore has a structural disadvantage unrelated to removing missingness cues.

Two minimal controls were retained: median-jitter breaks exact equality using 1% of the
outer-training IQR, and empirical-marginal is preserved under an accurate name as a deliberately
limited diagnostic. XGBoost median beats median-jitter by `+0.0064 [+0.0025,+0.0104]` AUROC and
empirical-marginal by `+0.0161 [+0.0105,+0.0220]`. Neither difference identifies a quantity called
“missingness contribution.” A mask-permutation mortality diagnostic was not added because it would
change the joint relationship between mask, value availability, and imputation; the direct
reconstructibility test answers the narrower question without pretending to be causal.

Tree-split/SHAP inspection was deprioritized as allowed by the brief. It would be model- and
threshold-fragile, whereas reconstructibility directly tests whether imputed representations
retain the missingness target.

# Missingness-Reconstructibility Diagnostic

A fold-honest histogram-gradient classifier predicts whether each of 37 variables was originally
absent from its five post-imputation summaries. Patients, imputers, and classifiers follow the same
outer folds. Across 296,000 patient-variable examples:

| imputer | micro AUROC | micro AP | median variable AUROC | incoherent imputed groups |
|---|---:|---:|---:|---:|
| median | 0.9994 | 0.9985 | 1.0000 | 0.0% |
| median-jitter | 1.0000 | 1.0000 | 1.0000 | 27.3% |
| empirical-marginal | 0.9808 | 0.9681 | 0.9719 | 74.1% |

Missingness is highly recoverable under every tested representation. Median does carry implicit
missingness, but empirical marginal does too. The mortality-performance gap cannot establish that
median-coded absence is responsible for most of the gap.

# Paired-Bootstrap Audit

The implementation draws patient indices with replacement, reuses the identical index matrix for
both models, recomputes the metric difference inside each of 2,000 replicates, and records seed
20260809. Independent code reproduced corrected XGBoost differences exactly:

- values+mask − values-only: `+0.0015823 [−0.0027744,+0.0059269]`;
- values-only − mask-only: `+0.0960026 [+0.0819327,+0.1104398]`.

Patient-level resampling is appropriate for these one-prediction-per-patient M2 comparisons. The
folds are an evaluation construction, not a second sampled cluster hierarchy. This does not
replace external-dataset or repeated-split uncertainty.

# Calibration Audit

The code clips probabilities to `[1e-15,1−1e-15]` and fits an effectively unpenalized logistic
regression of labels on predicted logits. Independent numerical maximum-likelihood fits agree
closely. The aggregate fit uses the same OOF labels on which it is reported, so it is descriptive,
not a corrected or independently evaluated calibration estimate.

Mask-only LR’s aggregate slope/intercept `1.000/−0.003` is not evidence of perfect calibration.
Its fold slopes range `0.884–1.168` and fold intercepts `−0.180–+0.214`. Per-fold metrics and
equal-mass OOF reliability bins/counts are now retained. A numerical edge case that gave the
constant prevalence predictor meaningless finite calibration values was fixed and regression
tested; constant-predictor calibration regression is now undefined. Proper post-hoc calibration
must isolate calibration data in M3.

# Baseline Audit

The prevalence reference now uses one global development-prevalence constant: AUROC 0.5 by
definition, AP=0.14025, Brier 0.12058, NLL 0.40542. The previous fold-specific constants caused
meaningless cross-fold ranking and AUROC 0.4994.

E-002 fixed `C=1` retained AUROC 0.7223745892. A current rerun produced 0.7222844; the 0.00009
drift cannot be resolved because the E-002 artifact lacks current M2-level environment and ID
provenance. Nested M2 selected `C={0.01,0.01,0.1,0.01,0.01}` and yields 0.7277881. The roughly
0.0055 increase is attributable to the changed model-selection protocol, so wording was changed
from exact reproduction to qualitative replication plus a tuned comparable estimate.

“GBDT” is XGBoost 3.4.0, not LightGBM or scikit-learn HGB. The manifest now records objective,
class weight, fixed and tuned parameters, selected round count per fold, seeds, package versions,
and the inner-only stopping procedure. Statics-only is official under the same protocol (LR
0.6310; XGBoost 0.6777). The compact mask LR decomposition gives ever 0.6898, counts/frequency
0.7059, recency 0.6802, ever+counts 0.7110, and full mask 0.7278.

# Leakage / Set-C Audit

The loader defaults to development sets a+b; neither runner requests set-c. Logs show exactly
4,000 a and 4,000 b records. The final record IDs exactly match the a+b cohort, are unique, and
number 8,000. Artifact provenance records sets `[a,b]` and excluded set `[c]`.

The v2 artifact includes git SHA and clean/dirty state, Python/platform/package versions, cohort,
split, config and NPZ hashes, fold parameters, seed, cutoff, representation, feature inventory,
model definition, labels, record IDs, and raw OOF predictions. No outer-test label or fitted cache
path was found.

# Claim Repairs

Withdrawn or narrowed:

- “values-only contains no missingness information” → it contains no explicit mask columns;
- “median-versus-stochastic proves shortcut learning” → the empirical control is incoherent and
  the gap is a sensitivity diagnostic only;
- “XGBoost gains +0.009 from explicit masks” → corrected gain +0.0016, interval includes zero;
- “calibration is perfect” → aggregate same-label descriptive regression only;
- “E-002 reproduced exactly” → qualitatively replicated under a different tuned protocol;
- any clinician-intent, causal, novelty, “model intelligence,” or ordering-policy language →
  associational measurement-presence and preprocessing language.

Safe statement: **Measurement-presence patterns are predictive, and preprocessing choices can
retain information about missingness even when explicit indicators are removed. Physiological
summaries dominate mask-only features; the incremental value of explicit masks is model- and
evaluation-protocol-dependent.**

# Corrected M2 Table

| model/view | AUROC | AP | Brier | NLL |
|---|---:|---:|---:|---:|
| prevalence | 0.5000 | 0.1403 | 0.1206 | 0.4054 |
| LR mask-only | 0.7278 | 0.2783 | 0.1114 | 0.3657 |
| XGBoost mask-only | 0.7319 | 0.2812 | 0.1111 | 0.3634 |
| LR values-only | 0.8095 | 0.4273 | 0.0997 | 0.3276 |
| XGBoost values-only | 0.8279 | 0.4471 | 0.0970 | 0.3151 |
| LR values+mask | 0.8240 | 0.4511 | 0.0969 | 0.3182 |
| XGBoost values+mask | 0.8295 | 0.4502 | 0.0968 | 0.3141 |

# Remaining Risks

- One public ICU dataset, one 24-hour cutoff, one outer split assignment, no external validation.
- Values-only cannot be made missingness-free by ordinary imputation; diagnostics do not separate
  physiology, acquisition process, treatment, and site workflow causally.
- Hyperparameter grids are compact and early-stopping round selection uses one inner split.
- Aggregate OOF calibration diagnostics reuse their labels and have no uncertainty intervals.
- E-002 lacks the artifact provenance required to explain its tiny current rerun drift exactly.
- Split-threshold/SHAP evidence was not collected; reconstructibility was prioritized.

# Supported Thesis

**Primary: Thesis D.** M2 itself demonstrates evaluation fragility: corrected nesting/refit removes
the XGBoost explicit-mask headline, and imputation construction changes both coherence and
performance. **Secondary: Thesis B.** Values dominate, while measurement presence and
preprocessing retain smaller, context-dependent signal. Thesis A is supported for LR but not as a
model-general claim; Thesis C is too strong because mask-only prediction survives.

# Recommended M3

Proceed, after sign-off, with **calibration robustness under structured information loss**, but
make the design stricter than the candidate:

- isolate training, calibration, and evaluation patients within each outer fold;
- remove coherent variable/panel groups, not only independent cells;
- compare mask-only, values-only, and values+mask with median and one defensible joint imputer;
- predeclare severity levels and primary contrasts;
- report AUROC, AP, Brier, NLL, calibration slope/intercept with uncertainty, reliability counts,
  and risk-coverage/AURC;
- distinguish natural missingness from synthetic deletion and do not call either deployment shift
  without evidence.

This ordering is scientifically optimal because M2 exposed calibration-estimation and imputation
fragility that should be resolved before interpreting policy rankings.

# Recommended M4

Later test **acquisition-policy ranking stability** across support-aware versus support-blind
protocol, natural versus structured synthetic masking, and predeclared cost regimes. Report paired
ranking changes and absolute utility, not a single favorable support/cost setting. M4 should use
the M3-vetted prediction/calibration pipeline; it should not precede M3.

# Quality Gate

- full pytest suite: pass;
- Ruff lint: pass;
- Ruff format check: pass;
- strict mypy over repository: pass;
- `git diff --check`: pass;
- corrected M2 rerun: pass;
- raw OOF independent rescoring and paired-bootstrap reproduction: pass;
- figures and diagnostic artifacts regenerated: pass.

# Final Recommendation

**PASS M2 WITH NONBLOCKING RISKS.** M3 is scientifically safe to begin only after the user accepts
the corrected thesis and M3 isolation requirements. Do not reuse the superseded `+0.0090`
XGBoost mask gain, the `+0.0145` missingness-attribution claim, or aggregate calibration values as
evidence of calibrated deployment performance.
