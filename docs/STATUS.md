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

## Next actions

**STOP at the M2 decision gate.** M3 (uncertainty/calibration) begins only after sign-off. The
report recommends calibration robustness under structured group-level information loss with an
isolated calibration split before later policy-ranking stability work.
