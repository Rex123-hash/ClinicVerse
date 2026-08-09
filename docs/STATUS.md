# Cliniverse — Status

**Updated:** 2026-08-09
**Current milestone:** M0 (pre-implementation) — research assessment delivered, awaiting scope sign-off.

## Where we are

| Milestone | State | Notes |
|---|---|---|
| Research assessment | **Done** | `docs/research_assessment.md` |
| Dataset selection + access verification | **Done** | PhysioNet/CinC 2012 verified downloadable, statistics reproduced by `scripts/verify_physionet2012.py` |
| Scope sign-off on reframing | **Blocked — awaiting owner decision** | See `docs/DECISIONS.md` D-001 |
| M0 repo/tooling/CI | Not started | |
| M1 TwinBench v0 | Not started | |
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

## Open questions for the project owner

1. **D-001** — Accept the reframing from "we invented active acquisition for patient world
   models" to "reproducible benchmark + evaluation of panel-level cost-aware acquisition"?
   The original framing is not defensible (`research_assessment.md` §3.1).
2. **Deadline** — the hackathon submission date determines how much of M5–M6 is reachable.

## Next actions once unblocked

M0: `uv` environment pinned to Python 3.12, ruff/mypy/pytest CI, P12 parser + hourly binning +
patient-level splits with set-c locked, and the leakage regression test.
