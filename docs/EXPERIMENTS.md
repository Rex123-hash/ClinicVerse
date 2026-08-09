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

## Planned next run

**E-001** — baseline tier 0/1 on task T1 (in-hospital mortality at the 24h decision point).
No numbers will be recorded here until executed.
