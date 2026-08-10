# Cliniverse — Status

**Updated:** 2026-08-11
**Current milestone:** M5 discovery + confirmation complete and classified M5-C; recovery arm pending.

## Where we are

| Milestone | State | Notes |
|---|---|---|
| Research assessment | **Done** | `docs/research_assessment.md` |
| Dataset selection + access verification | **Done** | PhysioNet/CinC 2012 verified downloadable, statistics reproduced by `scripts/verify_physionet2012.py` |
| Scope sign-off on reframing | **Done** | D-001 **superseded by D-008** after independent review #0 |
| M0 repo/tooling/CI | **Done** | See exit criteria below |
| independent review #0 response | **Done** | `docs/REVIEW_RESPONSE_0.md`, `docs/NOVELTY_REASSESSMENT.md`, `docs/BENCHMARK_SPEC.md` |
| M1 TwinBench v0 | **Done + repaired** | Adversarial audit in `docs/ADVERSARIAL_REVIEW_1.md` |
| M2 Baselines | **Done + repaired** | E-004; `docs/M2_MILESTONE_REPORT.md`, audit in `docs/ADVERSARIAL_REVIEW_2.md` |
| M3 Calibration robustness | **Done + repaired** | E-005; `docs/M3_MILESTONE_REPORT.md`, `docs/ADVERSARIAL_REVIEW_3.md`. M3-B |
| M4 Acquisition ranking stability | **Done + repaired** | `docs/M4_MILESTONE_REPORT.md`, `docs/ADVERSARIAL_REVIEW_4.md`. Verdict **M4-C**, ACCEPT |
| M5 Discrimination-silent failure search | **Discovery + confirmation done** | `docs/M5_DESIGN.md` (predeclared), `docs/M5_MILESTONE_REPORT.md`. Verdict **M5-C**: primary test T1 failed, transfer test T4 passed. Recovery arm not yet run |
| M6 API + minimal UI | Not started | |
| M7 Final + review response | Not started | |

## Verified facts (executed, not assumed)

- PhysioNet/CinC Challenge 2012 is openly downloadable with **no credentialing** (ODC-BY v1.0), ~20 MB.
- Outcomes are available for **all three sets** → **12,000 labeled patients** (not 4,000).
- Mortality: set-a 13.85%, set-b 14.20%, set-c 14.62%.
- **37** time-series variables (`Weight` is longitudinal, not a static descriptor).
- Binned grid occupancy **20.25%**, i.e. **79.75% missing** (production parser).
  The raw row-count bound of 24.46% is a loose upper bound only: it counts `-1` sentinel
  rows and within-hour collisions, and is never the reported missingness statistic.
- 3 records in set-a contain zero *valid* observations (only `Weight,-1`).

Reproduce with:

```bash
python scripts/verify_physionet2012.py
```

## M0 exit criteria — all met

| Criterion | Evidence |
|---|---|
| `uv` environment pinned to Python 3.12 | `pyproject.toml` (`requires-python >=3.12,<3.13`); torch 2.13, xgboost 3.4, sklearn 1.9 installed |
| Lint / format / type / test in CI | `.github/workflows/ci.yml`; ruff + `ruff format --check` + mypy strict + pytest |
| `pytest` green | 214 passed after repair #1 |
| ruff clean | `All checks passed!` |
| mypy strict clean | Repair #1 checks 21 source files |
| Parser reproduces verified statistics | `tests/test_physionet2012.py::TestSetA` pins 4,000 records / 554 deaths / 37 variables / 3 degenerate records |
| Binned occupancy below raw upper bound | 20.25% binned vs 24.46% raw row-count bound |
| Leakage test passes | `tests/test_splits.py`, `tests/test_physionet2012.py::TestLeakageGuards` |
| set-c quarantined | `load_cohort()` defaults to a+b and requires an explicit final-holdout flag for set-c; `final_holdout()` also requires an unlock token |

## M1 exit criteria — all met

| Criterion | Evidence |
|---|---|
| Seeded, reproducible case generation | Manifest v2 hashes dataset, schema, catalogue, mechanism, cases, and Git SHA |
| Manifests store no patient values | Specifications, provenance, and content fingerprints only; asserted by test |
| Non-inferability of the support oracle | `tests/test_disclosure.py::TestIndistinguishability` — under support-blind, two patients differing only in hidden availability produce byte-identical views |
| Information boundary enforced | A purchase cannot disclose beyond the epoch boundary; the cell becomes purchasable only after `advance_epoch()` |
| Unavailable requests charged in full | `test_successful_and_empty_requests_cost_the_same` |
| Works end-to-end on real data | Repaired E-003: explicit oracle, uniform-all, and training-frequency baselines |

## M2 exit criteria — all met

| Criterion | Evidence |
|---|---|
| Three binding representations on identical splits | `mask_only` / `values_only` / `values_mask`, E-004 |
| Prevalence floor | constant reference: AUROC 0.5000, AP 0.14025 |
| LR and XGBoost for each representation | corrected runs in `results/m2/results.json` |
| Tuning on development data only | fold-honest nested preprocessing/selection; set-c never loaded |
| Paired inference, not overlapping CIs | `paired_bootstrap_difference`, 2,000 patient resamples |
| Calibration slope/intercept + reliability | descriptive aggregate and per-fold OOF diagnostics retained |
| Machine-readable artifacts with provenance | git/package versions, feature inventory, cohort/split/config/NPZ hashes, fold parameters, raw OOF predictions |
| Figures generated from artifacts | `results/m2/figures/` |
| E-002 reconciled | fixed-`C` artifact 0.72237; nested/tuned M2 LR 0.72779 |

## Key M2 result

Mask-only reaches **AUROC 0.7319** with no clinical values, but values dominate
(VALUES ONLY − MASK ONLY = **+0.0960** AUROC, XGBoost). After correcting nesting and final refit,
explicit masks add only **+0.0016 [−0.0028,+0.0059]** to XGBoost values-only. Missingness remains
highly reconstructible after every tested imputer, but the empirical-marginal control is
structurally incoherent, so its mortality gap cannot quantify missingness contribution. Primary
**Thesis D**, secondary **Thesis B**.

## Key M3 result

Under whole-window removal of selected co-measurement-group variables, AUROC falls modestly from
0.8270 to 0.8002 while the calibration intercept moves from -0.010 to **+0.573** and mean predicted
risk falls to 0.0944 against 14.03% prevalence: systematic risk underestimation. The stronger
per-patient/per-analyte control is mask-identical, so structured-minus-variable-matched NLL and
Brier are exactly zero. The old excess over count-random is an analyte-identity effect, not an
identified coherence effect. **M3-B.** Platt improves proper scores/slope but not intercept drift.

## Key M4 result

Review #4 repaired a result-invalidating batched fixed-order bug and reran M4. Under the fair
support-blind protocol, `fixed_domain_order` now wins **8/8** conditions (mean Kendall tau-b
**+0.776**, min +0.600), with **zero fair winner changes and zero supported fair reversals**.
Classification **M4-C**, not M4-A/B.

In the primary condition, the surrogate expected-entropy-reduction heuristic discloses 2.428 new
cells at a 94.59% zero-new-cell request rate versus 11.556 cells at 75.99% for the training-frequency
random baseline. This is benchmark-specific descriptive evidence, not a general availability-versus-
information claim.

## Key M5 result

Exhaustive enumeration of all 1,023 non-empty co-measurement-group subsets against the frozen model,
each scored as an **excess over an amount-matched random control**, selected on discovery folds 0-2
and confirmed on folds 3-4. The primary test **failed**: rank-1 confirmation excess NLL +0.00587
[-0.00174, +0.01365]. The transfer test **passed** decisively: Spearman tau between discovery and
confirmation excess across all 1,023 configurations is **+0.865**, permutation p = 1.0e-4.

Descriptively (post-hoc, not confirmatory), a single group carries the effect: mean confirmation
excess is +0.01005 for the 512 configurations containing `BMP_like` and -0.00007 for the 511 without
it, and `BMP_like` appears in 50 of the top 50. Withholding `BMP_like` alone moves AUROC only
0.8179 -> 0.8078 while the calibration intercept moves -0.141 -> **+0.520** and mean predicted risk
falls 0.1432 -> **0.0984** against ~14% prevalence. **M5-C.**

## Superseded M3 gate note

**STOP at the M3 decision gate.** M4 begins only after sign-off on the contract in
`docs/ADVERSARIAL_REVIEW_3.md`: predeclare NLL-vs-budget primary, Brier co-primary, direct calibration
diagnostics secondary, and a fixed AUBC integration rule. No M4 work has started.

## M3 exit criteria — all met

| Criterion | Evidence |
|---|---|
| Design provenance | Original predeclaration retained; Review #3 amendments explicitly post-hoc |
| Isolated calibration partition | model-train 4,800 / calibration 1,600 / outer test 1,600 per fold |
| Imputer never refitted under stress | fitted once per fold on clean model-train data |
| Loss applied before feature construction | `evaluation/information_loss.py` operates on the cohort |
| Amount and variable matching | per-patient totals and per-patient/per-variable counts retained and asserted |
| Paired contrasts, not standalone CIs | 2,000 patient-level resamples, identical patients and seeds |
| Calibration ladder | uncalibrated / Platt / isotonic, all fitted on calibration data only |
| Figures from artifacts | `results/m3/figures/` |
| Demonstration patient by repaired rule | record 142380; median deterioration among 866 eligible deaths |
| Determinism verified | two reduced runs, 155 arrays, maximum difference 0.0; masks regenerated twice |
| Clean artifact provenance | schema v2; source `df18f97`; `git_dirty=false`; fold/source arrays retained |
