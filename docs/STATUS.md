# Cliniverse — Status

**Updated:** 2026-08-09
**Current milestone:** M2 (baselines) — M0 and M1 complete.

## Where we are

| Milestone | State | Notes |
|---|---|---|
| Research assessment | **Done** | `docs/research_assessment.md` |
| Dataset selection + access verification | **Done** | PhysioNet/CinC 2012 verified downloadable, statistics reproduced by `scripts/verify_physionet2012.py` |
| Scope sign-off on reframing | **Done** | D-001 **superseded by D-008** after independent review #0 |
| M0 repo/tooling/CI | **Done** | See exit criteria below |
| independent review #0 response | **Done** | `docs/REVIEW_RESPONSE_0.md`, `docs/NOVELTY_REASSESSMENT.md`, `docs/BENCHMARK_SPEC.md` |
| M1 TwinBench v0 | **Done** | Disclosure engine, masking, manifests, episode runner |
| M2 Baselines | In progress | Needs the directly comparable tuned full-value baseline |
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
| `pytest` green | 173 passed |
| ruff clean | `All checks passed!` |
| mypy strict clean | `Success: no issues found in 18 source files` |
| Parser reproduces verified statistics | `tests/test_physionet2012.py::TestSetA` pins 4,000 records / 554 deaths / 37 variables / 3 degenerate records |
| Binned occupancy below raw upper bound | 20.25% binned vs 24.46% raw row-count bound |
| Leakage test passes | `tests/test_splits.py`, `tests/test_physionet2012.py::TestLeakageGuards` |
| set-c quarantined | Quarantined from model fitting and model selection following an aggregate cohort audit (n=4,000, 585 deaths, read once during dataset assessment). `final_holdout()` requires an explicit unlock token |

## M1 exit criteria — all met

| Criterion | Evidence |
|---|---|
| Seeded, reproducible case generation | `CaseManifest.content_hash` is order-independent and seed-stable; `tests/test_cases.py::TestManifestReproducibility` |
| Manifests store no patient data | Specifications and hashes only; asserted by test |
| Non-inferability of the support oracle | `tests/test_disclosure.py::TestIndistinguishability` — under support-blind, two patients differing only in hidden availability produce byte-identical views |
| Information boundary enforced | A purchase cannot disclose beyond the epoch boundary; the cell becomes purchasable only after `advance_epoch()` |
| Unavailable requests charged in full | `test_successful_and_empty_requests_cost_the_same` |
| Works end-to-end on real data | E-003: 300 real patients, both protocols, spec-conformant waste behaviour |

## Next actions

M2 — baselines on T1 with the full metric suite, including the **directly comparable tuned
full-value baseline** required before any ratio against the E-002 availability-only figure may
be stated.
