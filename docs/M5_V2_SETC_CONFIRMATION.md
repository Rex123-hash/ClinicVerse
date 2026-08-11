# M5-v2 One-Shot Set-C Confirmation

**Date:** 2026-08-11
**Contract:** [`M5_V2_DESIGN.md`](M5_V2_DESIGN.md) §9 and the machine-readable
`set_c_evaluation_contract` block of the final freeze artifact
**Freeze source:** `01bc036145e22c1821de8aae8233c2bc4a75b7a0` · **Repo HEAD at execution:** `e2120eb`
**Executed:** once. **Result: CONFIRMED.**

---

## 1. Result

| criterion | value | verdict |
|---|---:|---|
| **Primary** — one-sided 95% lower bound on `Delta_C` | **+0.012421** | **PASS** (`LB > 0`) |
| **Discrimination-silent** — AUROC drop | **+0.011461** | **PASS** (`≤ 0.02`) |
| **CONFIRMATION** — both required | — | **PASS** |

| quantity | value |
|---|---:|
| `Delta_C` (mean paired excess NLL) | **+0.018347** |
| one-sided 95% LOWER bound | **+0.012421** |
| clean AUROC | **0.834994** |
| withheld AUROC | **0.823534** |
| AUROC drop | **+0.011461** |
| bootstrap | 10,000 resamples, seed 20260809, 10,000 valid |
| pattern | `BUN + Glucose + Na` |
| cohort | set-c, n = 4,000, 585 deaths, prevalence 0.14625 |

The observed set-c outcome counts (4,000 records, 585 deaths, 14.625%) reproduce the aggregate
figures recorded during the earlier dataset assessment, as expected.

## 2. What this confirms

A failure pattern selected entirely on development data — three analytes, `BUN + Glucose + Na` —
degrades this frozen model's probability reliability on a cohort that played no part in fitting it,
selecting it, or setting any threshold, **by more than an amount-matched random removal of the same
number of cells from the same patients**, and it does so while discrimination moves only 0.011 AUROC.

The confirmed effect (`Delta_C` = +0.018347) is **larger** than the development out-of-selection
estimate (+0.012013) and more than twice the predeclared minimum detectable effect (+0.008044). The
detectability gate was not optimistic.

## 3. Secondary descriptive diagnostics

**Not part of the frozen decision rule.** Reported for interpretation only.

| quantity | clean | withheld | change |
|---|---:|---:|---:|
| NLL | 0.31862 | 0.33746 | +0.01884 |
| Brier | 0.09890 | 0.10469 | +0.00579 |
| calibration intercept | +0.02629 | **+0.60641** | +0.58012 |
| calibration slope | 0.96970 | 1.07191 | +0.10221 |
| mean predicted risk | 0.13989 | **0.10370** | −0.03619 |
| mean removed cells / patient | 0 | 5.93 | — |

Against a set-c prevalence of 14.625%, mean predicted risk falls to 10.4% while AUROC moves 0.011.
This is the discrimination-silent signature — systematic under-prediction of mortality risk that
ordinary discrimination monitoring would not flag — reproduced on the holdout.

## 4. Mandatory limitation — restate verbatim wherever this result is reported

> The R = 5 control draws are FIXED across all 10,000 bootstrap replicates. The interval propagates
> patient-sampling uncertainty but NOT control-draw Monte-Carlo uncertainty, and is mildly optimistic
> on that account.

Equivalently, and in the required wording: the bootstrap propagates patient-sampling uncertainty but
does not separately propagate control-draw Monte-Carlo uncertainty, because the five control draws
are fixed across bootstrap replicates.

## 5. Historical Set-C disclosure — required wording

> **No Set-C patient-level information was retained or used for model fitting, model selection,
> failure-pattern selection, or any M5-v2 statistic after the aggregate audit.**

Set C must **not** be described as historically untouched. The aggregate cohort audit recorded at
commit `610f614` observed the set-c record count (4,000), in-hospital deaths (585), mortality rate
(14.62%) and two file sizes. No set-c per-patient data, record IDs, time series or per-variable
statistics were read before this evaluation.

## 6. Execution integrity

**Pre-flight, before the unlock.** The three frozen fitted artifacts were verified against their
recorded hashes; the frozen contract, pattern and five control seeds were read from the artifact and
checked; and the deserialised pipeline was required to reproduce the freeze's recorded development
diagnostic exactly — recorded `0.13257001340389252`, observed `0.13257001340389252`. The run aborts
on any mismatch. Every pre-flight step used development data only.

**Nothing was refitted.** The model, imputer and calibrator were deserialised from the freeze package
and applied as-is. The calibrator was never refitted under withholding.

**Exactly one analysis was run.** One pattern, one control configuration, one bootstrap, one decision.
No alternative pattern, R, control pool, delta, calibration or bootstrap was executed. The artifact
records `alternative_analyses_run: {count: 0}`.

**Amount matching was enforced per draw.** Each of the five controls was asserted to remove exactly
the frozen pattern's realised per-patient cell counts; a mismatch aborts the run.

### Disclosed provenance limitation

The result artifact records **`git_dirty: true`**. The sole uncommitted item at execution time was the
runner script `experiments/robustness/m5_v2_setc_oneshot.py`, which was untracked when it ran; repo
HEAD was `e2120eb` and every other tracked file was clean. The frozen fitted artifacts were
hash-verified before use, so the dirty flag does not affect which model or contract was executed.

**This will not be repaired by re-running.** Re-executing the holdout to obtain cleaner provenance
would be a second look at set-c, which the frozen contract forbids. The blemish is disclosed and left
in place.

## 7. Artifacts

`experiments/robustness/results/m5v2_setc/`

| file | sha256 |
|---|---|
| `results.json` | `7179a5744e5d9034a735fb6bcd1652a96e850e285fc60b8de61983a7d192a907` |
| `setc_oneshot_predictions.npz` | `b8ed025b4a3ed037a07e6351240aa84b5240d9432e84c0639529a21701e38783` |

The NPZ retains record IDs, labels, per-patient `d_i`, clean and withheld prediction vectors, all
five control prediction vectors, the five per-patient control log-loss rows, the per-patient removed
cell counts, and the five frozen control seeds — so every reported number is independently
recomputable from raw predictions.

Frozen fitted artifacts used, verified before the unlock:

| file | sha256 |
|---|---|
| `final_model.json` | `669ef8d158e4cdace73fd7d34a4397dba1ee6e2a909a9dfd99b24e3f467954d1` |
| `final_imputer.npz` | `0e84c7bc647521ced39c906a205d8b178d6a49a0d5680fc2de24bee2239193d2` |
| `final_calibrator.json` | `3d38a73417c2416ecd957a08ee1e79ad312a31d7753edf623dee6bec4ecc01f7` |

Set-c evaluation provenance: cohort fingerprint
`d3287171b76c4803cc6af4acf17c3118578362267ad5b948e6a8a5e2c0ea496e`, record-id hash
`dc6b25c8ed062234e4952c6aca7c413eeaed444ae251f8f0aa798e19ac1b4218`, `sets: ['c']`.

## 8. Limitations that survive confirmation

- **The 11/20 selection margin travels with this pattern.** Confirmation validates that *this* pattern
  damages reliability out-of-cohort; it does not establish that the exact three-analyte membership is
  the uniquely correct minimal core. The `BUN`-centred family remains better supported than the exact
  membership.
- **Analyte-set identity, never coherence.** Removal is whole-window, so per M3-B this identifies
  which analytes were withheld and cannot test co-occurrence structure or order events.
- Groups remain **reconstructed co-measurement clusters** (`*_like`), never verified laboratory orders.
- Synthetic withholding is not natural missingness and not deployment shift. One historical ICU
  dataset, one cutoff, no external validation beyond this holdout.
- The finding is not biologically surprising; the contribution is the procedure that found it
  automatically, controlled it against amount-matched removal, and confirmed it once on a quarantined
  cohort.
- No causal, deployment-utility, clinical-validation or clinician-intent claim.

## 9. Set C is now spent

The single pre-registered use is consumed. There is no second test, and no further set-c experiment is
authorised by this result.
