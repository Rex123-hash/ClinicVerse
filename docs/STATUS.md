# Cliniverse — Status

**Updated:** 2026-08-09
**Current milestone:** M1 (TwinBench v0) — M0 complete.

## Where we are

| Milestone | State | Notes |
|---|---|---|
| Research assessment | **Done** | `docs/research_assessment.md` |
| Dataset selection + access verification | **Done** | PhysioNet/CinC 2012 verified downloadable, statistics reproduced by `scripts/verify_physionet2012.py` |
| Scope sign-off on reframing | **Done** | D-001 ACCEPTED — panel-level AFA framing |
| M0 repo/tooling/CI | **Done** | See exit criteria below |
| M1 TwinBench v0 | In progress | |
| M2 Baselines | Not started | |
| M3 Uncertainty | Not started | |
| M4 Acquisition (core) | Not started | |
| M5 Ablations/robustness/OOD | Not started | |
| M6 API + minimal UI | Not started | |
| M7 Final + review response | Not started | |

## Verified facts (executed, not assumed)

- PhysioNet/CinC Challenge 2012 is openly downloadable with **no credentialing** (ODC-BY v1.0), ~20 MB.
- Outcomes are available for **all three sets** → **12,000 labeled patients** (not 4,000).
- Mortality: set-a 13.85%, set-b 14.20%, set-c 14.62%.
- 36 time-series variables; ≥76.7% missing on a naive hourly grid.
- 3 records in set-a contain zero time-series observations.

Reproduce with:

```bash
python scripts/verify_physionet2012.py
```

## M0 exit criteria — all met

| Criterion | Evidence |
|---|---|
| `uv` environment pinned to Python 3.12 | `pyproject.toml` (`requires-python >=3.12,<3.13`); torch 2.13, xgboost 3.4, sklearn 1.9 installed |
| Lint / format / type / test in CI | `.github/workflows/ci.yml`; ruff + `ruff format --check` + mypy strict + pytest |
| `pytest` green | 69 passed |
| ruff clean | `All checks passed!` |
| mypy strict clean | `Success: no issues found in 8 source files` |
| Parser reproduces verified statistics | `tests/test_physionet2012.py::TestSetA` pins 4,000 records / 554 deaths / 36 variables / 3 degenerate records |
| Binned occupancy below raw upper bound | 19.35% binned vs 23.28% raw row-count bound — correct direction (within-hour collapsing + implausible-value removal) |
| Leakage test passes | `tests/test_splits.py`, `tests/test_physionet2012.py::TestLeakageGuards` |
| set-c untouched | `final_holdout()` requires an explicit unlock token; every other split path raises `LeakageError` on set-c |

## Next actions

M1 — TwinBench v0: panel catalogue derived from empirical co-measurement structure, case
schema, masking mechanisms, seeded case generation with content-hash manifests.
