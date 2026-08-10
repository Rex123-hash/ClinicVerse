# Cliniverse — Experiment Log

**Rule: this file contains executed runs only.** No planned, expected, projected or illustrative
numbers may appear here. Every entry records the command, environment, and observed output.

---

## E-000 — Dataset access verification and structural statistics

- **Date:** 2026-08-09
- **Status:** COMPLETE, REPAIRED BY INDEPENDENT REVIEW #3 (M3-B)
- **Command:** `python scripts/verify_physionet2012.py --cache data/raw/physionet2012 --set a`
- **Environment:** Windows 11, Python 3.14.4, pandas 3.0.2, numpy 2.4.4
- **Purpose:** Confirm the primary dataset is obtainable without credentialing, and establish
  the real structural statistics used in `research_assessment.md` §5.1. Not a modelling run.

### Access (HTTP HEAD, unauthenticated)

| File | Bytes |
|---|---|
| `Outcomes-a.txt` | 79,219 |
| `Outcomes-b.txt` | 79,149 |
| `Outcomes-c.txt` | 79,191 |
| `set-a.tar.gz` | 6,632,372 |
| `set-b.tar.gz` | 6,652,690 |
| `set-c.tar.gz` | 6,600,293 |

All returned HTTP 200 with no authentication. License: ODC-BY v1.0.

### Outcomes

| Set | n | In-hospital deaths | Rate |
|---|---|---|---|
| set-a | 4,000 | 554 | 13.85% |
| set-b | 4,000 | 568 | 14.20% |
| set-c | 4,000 | 585 | 14.62% |

Columns: `RecordID, SAPS-I, SOFA, Length_of_stay, Survival, In-hospital_death`.

**Finding:** outcomes are published for all three sets, giving **12,000 labeled patients** —
three times the 4,000 commonly cited from the original challenge, where B and C were withheld.

### set-a time-series structure (n = 4,000)

- Time-series parameters: **36** (plus 6 static descriptors)
- Observation rows per record: median **392**, mean 402, min **0**, max 1,318
- Distinct timestamps per record: median **71**, mean 73.8, max 202
- Naive grid (4,000 × 48h × 36 vars) = 6,912,000 cells; 1,608,815 observation rows
  → occupancy **≤ 23.28%**, i.e. **≥ 76.72% missing**

### Data-quality findings

- **3 records with zero time-series observations:** `140501`, `140936`, `141264`
- **4 records with fewer than 20 observations:** `133628` (5), `136022` (13), `138477` (10),
  `139060` (1)

### Per-variable coverage (fraction of the 4,000 records where the variable is ever measured)

| Variable | Coverage | Obs/covered record | | Variable | Coverage | Obs/covered record |
|---|---|---|---|---|---|---|
| HR | 0.984 | 58.05 | | FiO2 | 0.679 | 11.92 |
| Creatinine | 0.984 | 3.55 | | MechVent | 0.632 | 12.32 |
| BUN | 0.984 | 3.54 | | Lactate | 0.546 | 3.68 |
| GCS | 0.984 | 15.64 | | SaO2 | 0.448 | 4.57 |
| HCT | 0.984 | 4.64 | | AST | 0.431 | 1.85 |
| Temp | 0.984 | 21.95 | | ALT | 0.430 | 1.85 |
| Platelets | 0.983 | 3.59 | | Bilirubin | 0.429 | 1.86 |
| WBC | 0.982 | 3.29 | | ALP | 0.422 | 1.83 |
| Na | 0.981 | 3.46 | | Albumin | 0.404 | 1.46 |
| HCO3 | 0.981 | 3.47 | | RespRate | 0.275 | 50.00 |
| K | 0.976 | 3.70 | | TroponinT | 0.216 | 2.46 |
| Mg | 0.974 | 3.49 | | Cholesterol | 0.076 | 1.03 |
| Glucose | 0.972 | 3.35 | | TroponinI | 0.051 | 2.12 |
| Urine | 0.971 | 35.26 | | | | |
| NISysABP | 0.873 | 28.15 | | | | |
| NIDiasABP | 0.871 | 28.20 | | | | |
| NIMAP | 0.870 | 27.83 | | | | |
| pH | 0.760 | 8.01 | | | | |
| PaO2 | 0.756 | 7.70 | | | | |
| PaCO2 | 0.756 | 7.71 | | | | |
| DiasABP | 0.700 | 52.01 | | | | |
| SysABP | 0.700 | 52.04 | | | | |
| MAP | 0.698 | 52.21 | | | | |

### Interpretation

Coverage splits into a **near-continuous monitoring tier** (HR, Temp, GCS, Urine, ABP variants;
15–58 observations per record) and a **discretely-ordered lab tier** (Creatinine, BUN, HCT,
Platelets, WBC, Na, HCO3, K, Mg, Glucose; **~3.3–3.6 observations per 48h stay**), plus a
selective/expensive tier (Lactate 0.55 down to TroponinI 0.051).

The lab tier's ~3.5 measurements per stay is a genuine discrete acquisition event, and the
variables group naturally into standard ordering panels. This is the empirical basis for the
panel-level acquisition framing in `research_assessment.md` §3.2(a).

---

## E-001 — Empirical derivation of the laboratory panel catalogue

- **Date:** 2026-08-09
- **Status:** COMPLETE
- **Command:** `python experiments/baselines/derive_panels.py --sets a b --threshold 0.35`
- **Git SHA:** recorded in commit for this milestone
- **Purpose:** Test whether laboratory variables have stable hourly-bin co-presence clusters.
  This does not recover orders, specimens, or clinical events.

### Method

For every `(patient, hour)` cell in which at least one of the 23 laboratory analytes was
measured (**86,559 active lab patient-hours** across sets a+b, n=8,000 patients), record which analytes
were measured together. Compute pairwise Jaccard co-measurement,
`P(i and j | i or j)`, then agglomeratively cluster (average linkage) on Jaccard distance.

**No clinical panel definitions were supplied to the clustering.** Similarity to recognised
panels supports cautious `*-like` labels; it is not evidence of actual orders.

### Result — clusters at Jaccard distance threshold 0.35

| Derived cluster | Members | Mean within-cluster Jaccard | Corresponds to |
|---|---|---|---|
| 1 | pH, PaCO2, PaO2 | **0.969** | ABG-like |
| 6 | ALP, ALT, AST, Bilirubin | **0.934** | hepatic-like |
| 4 | BUN, Creatinine, Glucose, HCO3, K, Mg, Na | **0.861** | BMP-like |
| 5 | HCT, Platelets, WBC | **0.782** | CBC-like |
| singletons | SaO2, Lactate, Albumin, TroponinT, TroponinI, Cholesterol | — | Individual sends |

The unsupervised clusters resemble four familiar panels at this threshold.

### Threshold sensitivity

| Threshold | Behaviour |
|---|---|
| 0.50 | BMP and CBC merge into one 10-analyte cluster (Jaccard 0.753) — consistent with a combined morning ICU draw |
| **0.35** | **BMP, CBC, LFT, ABG recovered exactly — adopted as the operating point** |
| 0.25 | Stable, except HCT peels off CBC leaving {Platelets, WBC} at 0.909 |
| 0.15 | BMP fragments: K and Mg separate, leaving {BUN, Creatinine, Glucose, HCO3, Na} at 0.919 |

The three primary panels (ABG, LFT, BMP-core) are stable across the whole 0.15–0.50 range.

### Strongest pairwise co-measurement

| Jaccard | Pair | | Jaccard | Pair |
|---|---|---|---|---|
| 0.995 | PaCO2–PaO2 | | 0.956 | PaO2–pH |
| 0.991 | BUN–Creatinine | | 0.953 | ALP–AST |
| 0.979 | ALT–AST | | 0.953 | Creatinine–HCO3 |
| 0.957 | PaCO2–pH | | 0.945 | ALP–ALT |
| 0.957 | BUN–HCO3 | | 0.929 | HCO3–Na |

### Findings that changed the design

1. **Feature grouping is empirically supported.** Within-group hourly co-presence is 0.78–0.97.
   That supports a feature-group sensitivity analysis, not a claim that the groups are real orders
   or that independent acquisition is clinically impossible.
2. **SaO2 is not part of ABG in this dataset.** It separated from the ABG cluster at every
   threshold (marginal draw frequency 0.176 vs 0.536 for pH). Assuming it belonged to ABG on
   clinical intuition would have been wrong; it is modelled as a singleton send.
3. **Albumin is not part of the hepatic panel here**, despite the clinical association. Also
   modelled as a singleton.

### Output

`experiments/baselines/results/panel_derivation.json` (full Jaccard and conditional matrices),
consumed by `configs/panels.yaml`.

---

## E-002 — Availability vs values: discrimination from measurement-presence patterns

- **Date:** 2026-08-09
- **Status:** COMPLETE
- **Command:** `python experiments/baselines/availability_ablation.py --cutoff 24 --folds 5`
- **Purpose:** The mandatory mask-only baseline (independent review #0, finding 11). Measures how much
  discrimination is associated with recorded measurement-presence patterns, with no measured
  value. It does not identify clinician intent or a causal process.
- **Setup:** Task T1 (in-hospital mortality), 24h decision point, 5-fold stratified CV on sets
  a+b, **n = 8,000**, prevalence **14.03%**. Imputation and scaling fit on training rows only.
  Bootstrap CIs, 1,000 resamples. Run **after** the Weight leakage fix (D-007), so no feature
  depends on post-cutoff data.

### Feature views

Three **disjoint** blocks, so each can be trained on alone:

- **availability** — per-variable observation counts, ever-measured flags, hours since last
  observation, plus totals. **Contains no measured value.**
- **values** — per-variable last / mean / min / max / slope. **Contains no mask indicator.**
- **statics** — Age, Gender, Height, ICUType, AdmissionWeight.

### Results

| Feature view | Model | #feat | AUROC | 95% CI | average precision (AP) | Brier |
|---|---|---|---|---|---|---|
| **availability only** | logreg | 113 | **0.7224** | [0.707, 0.738] | 0.2695 | 0.1127 |
| **availability only** | gbdt | 113 | 0.7187 | [0.703, 0.733] | 0.2588 | 0.1131 |
| values only | logreg | 185 | 0.8008 | [0.788, 0.814] | 0.4080 | 0.1023 |
| values only | gbdt | 185 | 0.8184 | [0.807, 0.831] | 0.4285 | 0.0992 |
| statics only | logreg | 5 | 0.6327 | [0.614, 0.651] | 0.2209 | 0.1171 |
| statics only | gbdt | 5 | 0.6692 | [0.653, 0.687] | 0.2237 | 0.1159 |
| availability + statics | logreg | 118 | 0.7514 | [0.736, 0.765] | 0.3033 | 0.1097 |
| availability + statics | gbdt | 118 | 0.7510 | [0.736, 0.765] | 0.3051 | 0.1093 |
| values + statics | gbdt | 190 | 0.8335 | [0.822, 0.844] | 0.4515 | 0.0966 |
| **all** | gbdt | 303 | **0.8363** | [0.825, 0.847] | 0.4565 | 0.0961 |

### Finding

**A model that sees no laboratory or vital value whatsoever — only measurement-presence patterns
(counts, ever-measured flags, recency) — reaches AUROC 0.7224 [0.707, 0.738].**

Adding measured values on top moves the all-blocks model from 0.7510 to 0.8363, so values carry
substantial independent signal. The point is not that values are useless; it is that
measurement-presence patterns alone support non-trivial discrimination.

**Reporting limit.** We do **not** state what fraction of full-model skill this represents. A ratio
of the form `(AUROC_avail − 0.5) / (AUROC_full − 0.5)` depends on an arbitrary chance-normalisation
and on an untuned full model. The comparable claim requires a properly tuned full-value baseline
on the identical cohort, split and protocol — deferred to M2.

### Consistency with prior work

Directionally consistent with published results and slightly stronger, as expected for a shorter
horizon: JAMA Netw Open (2019) reports AUROC ≈ 0.684 using missingness indicators alone for
30-day mortality. **The phenomenon is established prior art — we are not claiming to have
discovered it.** What it establishes *for this project* is that the acquirable support in a
retrospective acquisition benchmark is itself a strong predictor, which is why the support-blind
protocol (D-009, `BENCHMARK_SPEC.md` §5) is necessary rather than merely tidy.

### Caveats

- Availability features are computed on the 24h-truncated cohort; recency is measured to the
  cutoff, so no post-cutoff information is used.
- `HistGradientBoosting` handles NaN natively but is given the same imputation path as logistic
  regression so that views differ only in their features.
- These are out-of-fold predictions on development data (sets a+b). **set-c is quarantined.**

### Output

`experiments/baselines/results/availability_ablation.json` plus out-of-fold predictions in
`availability_ablation.oof.npz`, retained for later paired comparisons.

---

## E-003 — Disclosure protocol mechanics on real patients (M1 integration check)

- **Date:** 2026-08-09
- **Status:** RERUN AFTER ADVERSARIAL REVIEW #1; original ambiguous `random` rows superseded
- **Command:** `python experiments/baselines/disclosure_smoke.py --n 300 --budget 5 --rate 0.5 --seed 20260809 --cutoff 24`
- **Purpose:** Verify the M1 disclosure engine behaves as specified in
  `BENCHMARK_SPEC.md` on real data. **Mechanics only — no model is fitted and no predictive
  claim is made.**
- **Setup:** First 300 set-a patients, 24h cutoff, epochs (12, 18, 24), budget 5.0
  units, `group_hours@0.5#20260809` masking, `shared_plus_marginal` cost regime.
  `random_train_frequency` was fitted on the remaining 7,700 development patients, excluding
  every evaluation patient. Patient-specific policy seeds are `20260809 + patient_index`.

### Results (means per patient)

| Policy | Protocol | Spent | Requests | Values disclosed | Empty requests | Wasted spend |
|---|---|---|---|---|---|---|
| no_acquisition | support_aware | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| random_support_oracle | support_aware | 3.59 | 2.69 | **11.33** | 0.00 | 0.00 |
| fixed_order | support_aware | 2.84 | 2.02 | **10.82** | 0.00 | 0.00 |
| no_acquisition | support_blind | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| random_uniform_all | support_blind | 4.59 | 3.78 | **4.58** | 2.72 | 3.19 |
| random_train_frequency | support_blind | 4.46 | 3.45 | **7.32** | 1.96 | 2.44 |
| fixed_order | support_blind | 4.30 | 3.00 | **10.08** | 1.27 | 1.81 |

### Specification conformance

- Wasted spend is **exactly zero** under `support_aware`, as required: unavailable groups are not
  requestable there.
- Wasted spend is **positive** under `support_blind` — the budget cost of not being told what is
  available. This is the mechanism that makes hidden-bearing and empty groups indistinguishable
  before purchase (SO-2).
- `no_acquisition` spends nothing under both protocols, giving the zero-budget reference point.

### Interpretation

The old 10.89-versus-0.62 `random` contrast was not a comparison of the same information set:
the support-aware implementation sampled an availability-filtered oracle list, while the
support-blind implementation sampled all groups. Its labels and seventeen-fold interpretation
are superseded. With explicit baselines, the diagnostic oracle discloses 11.33 values,
support-blind uniform-all discloses 4.58, and the training-frequency support-blind baseline
discloses 7.32. Frequency weighting is materially stronger than uniform-all without using any
evaluation-patient support.

The fixed ordering remains 10.82 versus 10.08 because it targets commonly recorded groups.

**This is a mechanics observation, not a predictive result.** It confirms that baseline action
information must be named explicitly; `random_support_oracle` cannot be presented as deployable
or as the fair counterpart to a support-blind policy.

### Output

Printed to stdout; no artifacts written. Rerun to reproduce — the run is fully seeded.

---

## E-004 — M2 representation ablation (mask vs values vs both)

- **Date:** 2026-08-09
- **Status:** COMPLETE
- **Commands:**
  `python experiments/baselines/m2_representation_ablation.py --folds 5 --n-boot 2000`
  `python experiments/baselines/m2_figures.py`
- **Full report:** [`M2_MILESTONE_REPORT.md`](M2_MILESTONE_REPORT.md)
- **Artifacts:** `experiments/baselines/results/m2/{results.json,predictions.npz,figures/}`
- **Setup:** T1 in-hospital mortality, 24h boundary, 5-fold stratified CV, sets a+b, n=8,000,
  prevalence 14.03%. Nested hyperparameter selection on inner validation splits. set-c never
  loaded.

### Headline numbers

| run | AUROC [95% CI] | average precision (AP) | Brier | NLL |
|---|---|---|---|---|
| prevalence | 0.5000 [0.5000, 0.5000] | 0.1403 | 0.1206 | 0.4054 |
| LR mask-only | 0.7278 [0.7128, 0.7423] | 0.2783 | 0.1114 | 0.3657 |
| XGBoost mask-only | 0.7319 [0.7169, 0.7457] | 0.2812 | 0.1111 | 0.3634 |
| LR values-only | 0.8095 [0.7964, 0.8219] | 0.4273 | 0.0997 | 0.3276 |
| XGBoost values-only | 0.8279 [0.8162, 0.8395] | 0.4471 | 0.0970 | 0.3151 |
| LR values+mask | 0.8240 [0.8121, 0.8359] | 0.4511 | 0.0969 | 0.3182 |
| XGBoost values+mask | 0.8295 [0.8177, 0.8411] | 0.4502 | 0.0968 | 0.3141 |

### Paired differences (identical patients and folds)

| Comparison | LR | GBDT |
|---|---|---|
| VALUES+MASK − VALUES ONLY (AUROC) | +0.0146 [+0.0089, +0.0204] | +0.0016 [−0.0028, +0.0059] |
| VALUES ONLY − MASK ONLY (AUROC) | +0.0817 [+0.0655, +0.0975] | +0.0960 [+0.0819, +0.1104] |
| values-only median − median-jitter (AUROC) | +0.0014 [+0.0002, +0.0028] | +0.0064 [+0.0025, +0.0104] |
| values-only median − empirical-marginal (AUROC) | +0.0194 [+0.0142, +0.0247] | +0.0161 [+0.0105, +0.0220] |

The XGBoost explicit-mask interval includes zero; all other intervals shown exclude zero.

### Findings

1. **E-002 replicates qualitatively under a tuned protocol.** Its fixed-`C=1` artifact is 0.72237;
   nested M2 selects mostly `C=0.01` and reaches 0.72779. This is not an exact reproduction.
2. **Values dominate measurement patterns.** VALUES ONLY − MASK ONLY = +0.0960 AUROC (XGBoost).
3. **The XGBoost explicit-mask headline does not survive repair:** +0.0016 AUROC with a paired
   interval crossing zero. The LR gain remains +0.0146 and excludes zero.
4. **Values-only retains missingness cues, but the old imputation attribution is unsupported.**
   Empirical-marginal summaries are still highly missingness-reconstructible and are structurally
   incoherent, so their mortality gap is a sensitivity result rather than a missingness estimate.
5. **SAPS-I and SOFA omitted**: PhysioNet documents no time window for them and warns its own
   calculator disagrees with the outcomes file, so cutoff-safety is unverifiable.

### Interpretation

**Thesis D is primary:** nesting, final refit, representation and imputation materially affect the
conclusion. **Thesis B is secondary:** values dominate, while measurement presence and
preprocessing retain smaller model-dependent signal. The strong shortcut framing is unsupported.

---

## E-005 - M3 calibration robustness under structured information loss

- **Date:** 2026-08-09
- **Status:** COMPLETE
- **Design:** predeclared in [`M3_DESIGN.md`](M3_DESIGN.md), committed before execution
- **Full report:** [`M3_MILESTONE_REPORT.md`](M3_MILESTONE_REPORT.md)
- **Commands:**
  `python experiments/robustness/m3_calibration_under_loss.py --n-boot 2000`
  `python experiments/robustness/m3_figures.py`
- **Artifacts:** `experiments/robustness/results/m3/`
- **Setup:** T1 mortality, 24h boundary, 5-fold, sets a+b, n=8,000. Isolated three-way partition
  per fold (model-train 4,800 / calibration 1,600 / outer test 1,600). Imputer fitted once per fold
  on clean model-train data, never refitted under stress. Calibrators fitted on clean calibration
  data only. set-c never loaded.

### Realized severity (group loss is indivisible)

| requested | realized mean | cells removed | matched per patient |
|---|---|---|---|
| 0.25 | 0.284 | 86,192 | yes |
| 0.50 | 0.519 | 157,625 | yes |
| 0.75 | 0.779 | 231,414 | yes |

### Primary result (values_mask / XGBoost / Platt)

| severity | condition | AUROC | NLL | cal. intercept | mean predicted risk |
|---|---|---|---|---|---|
| 0.00 | none | 0.8270 | 0.3151 | -0.010 | 0.1397 |
| 0.284 | group | 0.8204 | 0.3212 | +0.204 | 0.1224 |
| 0.519 | group | 0.8118 | 0.3308 | +0.360 | 0.1094 |
| 0.779 | group | 0.8002 | 0.3445 | **+0.573** | **0.0944** |
| 0.779 | variable-matched scattered | 0.8002 | 0.3445 | **+0.573** | **0.0944** |
| 0.779 | cell_random | 0.8016 | 0.3345 | +0.115 | 0.1184 |

True prevalence is 0.1403 throughout.

### Paired contrast, group minus matched cell (Platt)

| severity | NLL | Brier |
|---|---|---|
| 0.284 | +0.0029 [-0.0005, +0.0065] | +0.0007 [-0.0005, +0.0019] |
| 0.519 | +0.0076 [+0.0028, +0.0122] | +0.0021 [+0.0005, +0.0036] |
| 0.779 | +0.0100 [+0.0057, +0.0145] | +0.0027 [+0.0012, +0.0042] |

### Findings

1. Discrimination is robust; calibration is not. AUROC falls 0.827 to 0.800 across a 78% loss,
   while the calibration intercept moves from -0.010 to +0.573 and mean predicted risk falls to
   0.0944 against an unchanged 14.03% prevalence.
2. The old count-matched comparison does not isolate structure. Exact per-patient/per-analyte
   matching is mask-identical to group loss under the implemented whole-window variable-removal
   semantics, giving NLL and Brier differences exactly 0.0000 [0.0000, 0.0000]. The old excess
   over count-random combines analyte identity with amount.
3. Clean-data Platt improves NLL/Brier and moves slope closer to one, but does not remove the
   intercept/risk-level drift. Under this clean-calibration/shifted-test protocol, isotonic has
   worse stress NLL and slope than Platt; no general harmfulness claim is made.
4. AURC barely separates the conditions - a negative result for that metric as a stress readout.
5. The entropy decrease is not independent evidence of overconfidence; at a 14% base rate it
   follows mechanically from predicted risks falling.

### Reproducibility note

An intermediate lint auto-fix changed behaviour in `information_loss.py` mid-milestone, so an
earlier draft table differed. Review #3 now generates every loss mask twice, pins semantics in
regression tests, and verified two reduced end-to-end runs across 155 arrays as bit-identical
(max difference 0.000e+00). Repaired artifacts use schema v2 and clean source provenance.

---

## Planned next runs

**E-004** — T1 clinical baselines: prevalence, SAPS-I, SOFA as single-feature predictors, for
comparison against the model tiers above.
**E-005** — support-aware vs support-blind disclosure, paired ΔAUBC across policies and cost
regimes (the primary pre-registered comparison in `BENCHMARK_SPEC.md` §3).

No numbers will be recorded here until executed.
