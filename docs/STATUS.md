# Cliniverse — Status

**Updated:** 2026-08-09
**Current milestone:** M2 complete — decision gate reached, awaiting sign-off before M3.

## Where we are

| Milestone | State | Notes |
|---|---|---|
| Research assessment | **Done** | `docs/research_assessment.md` |
| Dataset selection + access verification | **Done** | PhysioNet/CinC 2012 verified downloadable, statistics reproduced by `scripts/verify_physionet2012.py` |
| Scope sign-off on reframing | **Done** | D-001 **superseded by D-008** after independent review #0 |
| M0 repo/tooling/CI | **Done** | See exit criteria below |
| independent review #0 response | **Done** | `docs/REVIEW_RESPONSE_0.md`, `docs/NOVELTY_REASSESSMENT.md`, `docs/BENCHMARK_SPEC.md` |
| M1 TwinBench v0 | **Done + repaired** | Adversarial audit in `docs/ADVERSARIAL_REVIEW_1.md` |
| M2 Baselines | Ready to start | Must follow the contract in `docs/ADVERSARIAL_REVIEW_1.md` |
| M3 Uncertainty | Not started | |
| M4 Acquisition (core) | Not started | |
| M5 Ablations/robustness/OOD | Not started | |
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
| Prevalence floor | AUROC 0.4994 [0.4840, 0.5152] |
| LR and GBDT for each representation | 13 runs in `results/m2/results.json` |
| Tuning on development data only | Nested inner-validation selection; set-c never loaded |
| Paired inference, not overlapping CIs | `paired_bootstrap_difference`, 2,000 patient resamples |
| Calibration slope/intercept + reliability | Reported per run; reliability figure from artifacts |
| Machine-readable artifacts with provenance | git SHA, cohort fingerprint, split hash, config hash, per-fold hyperparameters, raw OOF predictions |
| Figures generated from artifacts | `results/m2/figures/` |
| E-002 reproduced under a comparable protocol | mask-only 0.7278/0.7280 vs 0.7223745892 |

## Key M2 result

Mask-only reaches **AUROC 0.7278** with no clinical values. But values dominate
(VALUES ONLY − MASK ONLY = **+0.095** AUROC, GBDT), and explicit mask features add only
**+0.0090** on top of values. Meanwhile median-versus-stochastic imputation is worth **+0.0145** —
larger than the entire mask block. Pre-declared **Outcome B**; the strong measurement-shortcut
framing is not supported and has been restated.

## Next actions

**STOP at the M2 decision gate.** M3 (uncertainty/calibration) begins only after sign-off. The
report recommends carrying imputation strategy as an experimental axis into M3/M4, and identifies
calibration under budget pressure (H3) as the strongest remaining open question.
