# M5-v2 Final Model Freeze

**Date:** 2026-08-11
**Contract:** [`M5_V2_DESIGN.md`](M5_V2_DESIGN.md) §8, executed exactly as predeclared
**Generated from:** `08f767a`, clean tree (`git_dirty=false`)
**Stage:** model freeze only. **No evaluation was performed. Set C was not accessed.**

This stage fits one final pipeline on A+B and freezes it, together with the set-c evaluation
contract. It does not test anything, and it does not authorise testing anything.

---

## 1. Frozen failure pattern

**`BUN + Glucose + Na`** — cohort columns `[4, 11, 23]`.

Carried in as a constant from the M5-v2 A+B development phase (verdict **v2-STABLE**, selected in
**11/20** resplits, exactly the majority threshold). **This stage performs no selection**; the
pattern cannot change here.

**The 11/20 margin travels with this pattern.** One resplit flipping to `BUN+Glucose` would have
produced 10/20 and a v2-DIFFUSE verdict. The exact three-analyte membership is knife-edge; the
**`BUN`-centred family is more stable than the exact membership**, and every selection in every
resplit contained `BUN`.

## 2. The final model contract, as executed

| element | value |
|---|---|
| source sets | **a + b only** (set-c never loaded) |
| patients | 8,000, prevalence 0.14025 |
| partition | **6,400 final-training / 1,600 final-calibration**, disjoint, stratified by mortality |
| partition seed | **20260809** |
| train prevalence | 0.14031 |
| calibration prevalence | 0.14000 |
| representation | `values_mask`, 24-hour cutoff, 298 features |
| imputer | median, fitted on **the 6,400 clean training rows only**, seed 20260809 |
| model | `XGBClassifier`, `random_state=20260809`, fitted on **the 6,400 only** |
| hyperparameters | `max_depth=5, learning_rate=0.05, min_child_weight=10, n_estimators=200, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0` — unchanged from M2/M3/M4/M5 |
| features split on | **230** (degenerate-booster guard passed) |
| calibrator | Platt, fitted on **the isolated 1,600 clean calibration rows only** |
| calibrator parameters | slope **0.991581**, intercept **0.070116** |
| calibrator refitting under withholding | **never** |

Training on all 8,000 without isolation is not permitted, because the calibrator would then see its
own training data and the frozen probabilities would be self-calibrated. The 6,400/1,600 partition is
what prevents that, and `tests/test_final_freeze.py` pins sizes, disjointness, coverage,
stratification and seed reproducibility.

**Fitting diagnostic, not evaluation.** Mean raw prediction on the calibration rows 0.13257, mean
calibrated 0.13999. In-sample for the calibrator by construction; recorded only so the freeze can be
reproduced and verified, never as evidence of performance.

## 3. Artifact hashes

| file | sha256 |
|---|---|
| `final_model.json` | `669ef8d158e4cdace73fd7d34a4397dba1ee6e2a909a9dfd99b24e3f467954d1` |
| `final_imputer.npz` | `0e84c7bc647521ced39c906a205d8b178d6a49a0d5680fc2de24bee2239193d2` |
| `final_calibrator.json` | `3d38a73417c2416ecd957a08ee1e79ad312a31d7753edf623dee6bec4ecc01f7` |

The package is serialised transparently rather than pickled — XGBoost native JSON, imputer arrays as
NPZ, Platt slope/intercept as JSON — so a reviewer can inspect every fitted quantity without
executing our code.

Provenance and split fingerprints:

| field | value |
|---|---|
| git SHA | `08f767a748381431f1824183a2dc3911bae408ac` |
| git dirty | **false** |
| source sets | `['a', 'b']` |
| cohort fingerprint | `f59c44f07556b7a606623b928df12770af795abccff4b5771d5fe26fa25a2e34` |
| config hash | `40e192fe9139cd04c0b6647e738c4d85ed4de5b29c27b2ee951e4ee90de18b55` |
| training index hash | `8feae9cee60a1c2d48557a24d44a8f98ad8107986eb36f63a2d1e90696541ca5` |
| calibration index hash | `193a3e806e1b4a6d5ed3bf7c23a0fff0bd0b9c507dd4dfefd9c7936a9fb6311f` |
| training record-id hash | `e62187cbfd85ca8115b796d054165a38a91f66eb49d5385bb8c057dee5e5db41` |
| calibration record-id hash | `3a7914d77f56c290fed61099f983fd376be512c6fa1c9dd29bf6db2bbb9969a5` |

The cohort fingerprint matches the one recorded by M4 and M5, confirming the same development cohort.

## 4. Frozen set-C evaluation contract — NOT EXECUTED

Frozen now so it cannot be designed after seeing anything. **Running it requires separate explicit
approval.**

| element | frozen value |
|---|---|
| pattern | `BUN + Glucose + Na` — one pattern, no alternatives |
| expected n | 4,000 |
| controls | **R = 5** amount-matched random draws |
| control seeds | **963394647, 118547003, 817200064, 959170045, 1019676579** |
| statistic | `d_i` = per-patient log loss under withholding minus the mean per-patient log loss over the 5 controls; `Delta_C = mean_i d_i` |
| interval | paired patient-level **percentile bootstrap** on `{d_i}` |
| resamples | **10,000** |
| bootstrap seed | **20260809** |
| bound | one-sided **95% LOWER** confidence bound = 5th percentile of replicate means |
| single-class resamples | skipped, matching project convention |
| **primary decision rule** | **PASS if and only if `LB > 0`** |
| constraint | `AUROC(clean) − AUROC(pattern) ≤ 0.02` on set-c |
| direction | fixed in advance by development; the hypothesis is strictly `Delta_C > 0` |

**Forbidden in that run:** any search or enumeration; any tuning of model, calibrator, imputer or
pattern; any alternative or substitute pattern; any second look, retest, or retest under a different
delta; any refitting of the calibrator under withholding.

**Monte-Carlo limitation — must be restated verbatim in any report of the set-c result:**

> The R = 5 control draws are FIXED across all 10,000 bootstrap replicates. The interval propagates
> patient-sampling uncertainty but NOT control-draw Monte-Carlo uncertainty, and is mildly optimistic
> on that account.

## 5. Historical Set-C exposure — precise disclosure

**The defensible wording, to be used everywhere:**

> Set C was **quarantined from model fitting and model selection after an earlier aggregate cohort
> audit.**

It must **not** be described as "historically untouched", "never seen" or "never read". The following
aggregate information was observed once, during dataset assessment, and is recorded in the repository.

**What was observed, exactly.** Established from git history; first introduced at commit `610f614`
("docs: research assessment, dataset verification, project charter") and carried since in
`docs/EXPERIMENTS.md` (E-000) and `docs/BENCHMARK_SPEC.md`:

| observed quantity | value | where recorded |
|---|---|---|
| set-c record count | 4,000 | `EXPERIMENTS.md` E-000 outcomes table |
| set-c in-hospital deaths | **585** | same |
| set-c mortality rate | **14.62%** | same; also `STATUS.md` |
| `Outcomes-c.txt` size | 79,191 bytes | E-000 access table (HTTP HEAD) |
| `set-c.tar.gz` size | 6,600,293 bytes | E-000 access table (HTTP HEAD) |

**What was NOT observed.** No set-c per-patient data, record IDs, time series, per-variable coverage,
missingness structure or feature statistics were ever read. The E-000 time-series structure and
per-variable coverage tables are computed on **set-a only** (E-000 states this explicitly). No set-c
quantity has ever entered a fitted object, a hyperparameter choice, a split, a threshold, a metric, or
a pattern selection.

**Current local cache state**, established from filesystem metadata without reading contents:

- `data/raw/physionet2012/Outcomes-c.txt` — **present** (the labels file was downloaded).
- `data/raw/physionet2012/set-c/` — **absent**. The set-c record archive was never materialised, so
  the time-series data does not exist locally.

The honest summary is therefore: **set-c aggregate outcome counts were read once during dataset
assessment and its labels file is present on disk; its records have never been materialised, and
nothing from set-c has ever influenced any modelling decision.**

The lock is enforced in code, not by convention: `load_cohort()` defaults to sets a+b and raises
`DataError` when set-c is requested without `allow_final_holdout=True`; `final_holdout()` requires a
separate unlock token. `tests/test_final_freeze.py` asserts both.

## 6. Confirmation: Set C was not accessed during this stage

- `allow_final_holdout` is **never passed** anywhere in `m5_v2_final_freeze.py`; a source-level test
  asserts this and fails even on a machine with no dataset present.
- The realised source sets were asserted at runtime to be exactly `('a', 'b')`, and the run aborts
  otherwise.
- The artifact records `set_c_access: {loaded_during_freeze: false, scored_during_freeze: false,
  allow_final_holdout_passed: false, realised_source_sets: ["a", "b"]}`.
- Provenance independently records `sets: ['a', 'b']`.

## 7. Reproduce

```bash
python experiments/robustness/m5_v2_final_freeze.py
```

Artifacts in `experiments/robustness/results/m5v2_final_freeze/`: `final_freeze.json` (the full
contract, split provenance, hashes and the frozen set-c contract), `final_model.json`,
`final_imputer.npz`, `final_calibrator.json`.

## 8. What is not done

- **The set-c test has not been run.** Set C remains locked.
- No M6 work has started.
- Nothing has been pushed to any remote.
