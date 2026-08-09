# Cliniverse — Experiment Log

**Rule: this file contains executed runs only.** No planned, expected, projected or illustrative
numbers may appear here. Every entry records the command, environment, and observed output.

---

## E-000 — Dataset access verification and structural statistics

- **Date:** 2026-08-09
- **Status:** COMPLETE
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
- **Purpose:** The panel-level acquisition framing rests on a factual claim — that
  laboratory analytes are ordered *in groups, as single events*. This run tests that claim
  instead of assuming it.

### Method

For every `(patient, hour)` cell in which at least one of the 23 laboratory analytes was
measured (**86,559 ordering events** across sets a+b, n=8,000 patients), record which analytes
were measured together. Compute pairwise Jaccard co-measurement,
`P(i and j | i or j)`, then agglomeratively cluster (average linkage) on Jaccard distance.

**No clinical panel definitions were supplied to the clustering.** Recovery of recognised
panels is therefore evidence, not construction.

### Result — clusters at Jaccard distance threshold 0.35

| Derived cluster | Members | Mean within-cluster Jaccard | Corresponds to |
|---|---|---|---|
| 1 | pH, PaCO2, PaO2 | **0.969** | Arterial blood gas (ABG) |
| 6 | ALP, ALT, AST, Bilirubin | **0.934** | Hepatic function panel (LFT) |
| 4 | BUN, Creatinine, Glucose, HCO3, K, Mg, Na | **0.861** | Basic metabolic panel (BMP) |
| 5 | HCT, Platelets, WBC | **0.782** | Complete blood count (CBC) |
| singletons | SaO2, Lactate, Albumin, TroponinT, TroponinI, Cholesterol | — | Individual sends |

The unsupervised clustering recovers the four standard clinical panels exactly.

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

1. **The panel claim is supported.** Within-panel co-measurement of 0.78–0.97 means treating
   these analytes as independently acquirable features — as all surveyed AFA benchmarks do —
   misrepresents how the data is generated. This is the empirical basis for
   `research_assessment.md` §3.2(a).
2. **SaO2 is not part of ABG in this dataset.** It separated from the ABG cluster at every
   threshold (marginal draw frequency 0.176 vs 0.536 for pH). Assuming it belonged to ABG on
   clinical intuition would have been wrong; it is modelled as a singleton send.
3. **Albumin is not part of the hepatic panel here**, despite the clinical association. Also
   modelled as a singleton.

### Output

`experiments/baselines/results/panel_derivation.json` (full Jaccard and conditional matrices),
consumed by `configs/panels.yaml`.

---

## E-002 — Availability vs values: how much of ICU mortality prediction is clinician behaviour?

- **Date:** 2026-08-09
- **Status:** COMPLETE
- **Command:** `python experiments/baselines/availability_ablation.py --cutoff 24 --folds 5`
- **Purpose:** The mandatory mask-only baseline (independent review #0, finding 11). Measures how much
  discrimination is available from *which tests were ordered* alone, with no measured value.
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

| Feature view | Model | #feat | AUROC | 95% CI | AUPRC | Brier |
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

## Planned next runs

**E-003** — T1 clinical baselines: prevalence, SAPS-I, SOFA as single-feature predictors, for
comparison against the model tiers above.
**E-004** — support-aware vs support-blind disclosure, paired ΔAUBC across policies and cost
regimes (the primary pre-registered comparison in `BENCHMARK_SPEC.md` §3).

No numbers will be recorded here until executed.
