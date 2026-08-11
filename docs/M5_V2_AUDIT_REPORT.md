# M5-v2 Adversarial Scientific Audit

**Date:** 2026-08-11  
**Scope:** targeted audit of M5-v2 only; A+B development data and committed M5-v2 artifacts  
**Audit verdict:** **REPAIR**  
**Post-repair scientific verdict:** **v2-STABLE**  
**Implementation repair:** `91262fd`  
**Repaired artifact provenance:** `git_sha=91262fd5c53c56665a042d11225abdc7b5c85777`,
`git_dirty=false`

## Decision

One P0 defect existed in the nested out-of-selection estimator: four-fold nested selection used
five-fold pooled AUROC eligibility, allowing the held-out fold to influence candidate eligibility.
The repair computes pooled AUROC on exactly the four selection folds. It changes 16 of 100 nested
choices and changes the development estimate from +0.012120478724049692 to
**+0.012013168205102115**, but it does not change the frozen pattern, G1-G3 or the G4 decision.

M5-v2 is safe to freeze before the single final-holdout evaluation under the repaired artifacts.
The exact three-analyte identity remains a knife-edge development result at 11/20.

## Exact post-repair gates

| item | independently reproduced value | result |
|---|---:|---|
| frozen pattern | `BUN + Glucose + Na` | — |
| G1 | 0/20 null-control selections | PASS |
| G2 | 11/20 exact-pattern selections; threshold 11 | PASS at minimum |
| clean AUROC, reference run | 0.8270462696167844 | — |
| withheld AUROC, reference run | 0.8113249172359208 | — |
| AUROC drop | 0.015721352380863585 | G3 PASS (`<= 0.02`) |
| naive development mean | +0.01378706584739147 | descriptive only |
| `Delta_hat_oos` | +0.012013168205102115 | development only |
| `sigma_Delta` | 0.20461633739548665 | one reference OOF prediction per patient |
| MDE | +0.00804441345228621 | one-sided alpha 0.05, power 0.80, n=4000 |
| G4 comparison | 0.00804441345228621 <= 0.012013168205102115 | PASS |

The MDE was independently recomputed as
`(z_0.95 + z_0.80) * 0.20461633739548665 / sqrt(4000)`.

## Provenance and quarantine

The git order is clean: predeclaration `5562120` precedes implementation `1de9feb`, which precedes
result commit `4ce67f7`. The design document is unchanged across the implementation and result
commits. The repair at `91262fd` restores the original four-fold nested-selection contract without
changing a threshold, candidate space, model, seed or gate.

The M5-v2 loader defaults mechanically to sets A+B, `development_cohort` rejects a materialised
locked-set patient, and neither the runner nor its dependencies request the final-holdout flag or
unlock token. Both original and repaired logs contain only set A and set B loads. Repaired provenance
records `sets=[a,b]`, `n_patients=8000`, and `excluded_sets=[c]`. The selector, models, controls,
scoring, detectability arithmetic, tests and artifact production did not load final-holdout data.

Repository history already discloses that an aggregate audit of the final holdout occurred during
earlier dataset assessment. Therefore “never touched” is not literally correct historically; the
accurate description is “quarantined from model fitting and model selection after an aggregate
cohort audit.” No final-holdout feature row or label row was loaded in this audit, and no such
quantity was used in any M5-v2 statistic or repair choice.

## Search space and withholding

Independent reconstruction produced exactly 127 non-empty `BMP_like` subsets, 7 `CBC_like` subsets
and 7 `ABG_like` subsets: 141 candidates. All 141 names, analyte sets and effective A+B removal masks
are unique. All analytes map to intended cohort variables. The eligibility pool is exactly these 23
catalogued laboratory analytes:

`ALP, ALT, AST, Albumin, BUN, Bilirubin, Cholesterol, Creatinine, Glucose, HCO3, HCT, K, Lactate,
Mg, Na, PaCO2, PaO2, Platelets, SaO2, TroponinI, TroponinT, WBC, pH`.

No vital, ventilator or static variable enters the candidate or control pool. Candidate withholding
is deterministic whole-window removal on the truncated 24-hour cohort before feature construction;
both value cells and observation masks are cleared. The analyte-addressed path is bit-identical to
the earlier group-addressed path when given all members of a group.

## Amount-matched controls

The repaired full run exercised all 141 x 5 = 705 control draws. Each draw passes the runtime hard
assertion that its per-patient removed-cell vector is exactly equal to the candidate's realised
per-patient count vector. The 705 hashed draw seeds are unique and deterministic. The sampling code
draws without replacement only from that patient's observed cells in the fixed 23-lab pool; it has no
label input, no outcome-dependent seed and no cross-patient sampling state. A mismatch would stop the
run rather than produce an artifact.

## Splits, fitting and selector

The resplit seeds are exactly 20260809 through 20260828. Every resplit contains five stratified outer
folds of 1,600 patients, and every patient receives exactly one OOF prediction. For each outer fold,
the outer training partition alone is split into model-training and calibration rows. The median
imputer and XGBoost fit only model-training rows; Platt scaling fits only isolated calibration rows;
the outer-test fold enters none of those fitted objects. Feature construction is per-patient, and all
population-level preprocessing is fold-local.

The selector independently reproduces the predeclared leader, leader-dispersion band, sparsity
preference, mean-score tie-break and lexical tie-break. The complete nonzero 20-resplit frequency
table is:

| pattern | count |
|---|---:|
| `BUN+Glucose+Na` | 11 |
| `BUN+Glucose` | 4 |
| `BUN` | 1 |
| `BUN+Glucose+HCO3` | 1 |
| `BUN+Glucose+Mg` | 1 |
| `BUN+Glucose+HCO3+Mg` | 1 |
| `BUN+Glucose+HCO3+Na` | 1 |
| all other 134 candidates | 0 |

Per-resplit choices, in seed order, are:

| b / seed | selected pattern |
|---|---|
| 0 / 20260809 | `BUN+Glucose+Na` |
| 1 / 20260810 | `BUN+Glucose+Na` |
| 2 / 20260811 | `BUN+Glucose+Mg` |
| 3 / 20260812 | `BUN+Glucose+Na` |
| 4 / 20260813 | `BUN+Glucose+Na` |
| 5 / 20260814 | `BUN+Glucose+Na` |
| 6 / 20260815 | `BUN+Glucose+Na` |
| 7 / 20260816 | `BUN+Glucose+Na` |
| 8 / 20260817 | `BUN+Glucose` |
| 9 / 20260818 | `BUN+Glucose+HCO3` |
| 10 / 20260819 | `BUN+Glucose+Na` |
| 11 / 20260820 | `BUN+Glucose+Na` |
| 12 / 20260821 | `BUN+Glucose` |
| 13 / 20260822 | `BUN+Glucose` |
| 14 / 20260823 | `BUN` |
| 15 / 20260824 | `BUN+Glucose+HCO3+Mg` |
| 16 / 20260825 | `BUN+Glucose+Na` |
| 17 / 20260826 | `BUN+Glucose` |
| 18 / 20260827 | `BUN+Glucose+HCO3+Na` |
| 19 / 20260828 | `BUN+Glucose+Na` |

Zero top-level selections come from the development-derived null-control regions. This is a sanity
check against one failure mode, not proof that the method is valid and not an independent biological
negative-control result.

## Corrected nested out-of-selection summary

All 100 nested selections use ranking, tie-band dispersion and pooled AUROC eligibility from the
other four folds only. The held-out fold is read only after the pattern is fixed.

| selected pattern | count |
|---|---:|
| `BUN+Glucose+Na` | 32 |
| `BUN+Glucose` | 24 |
| `BUN` | 11 |
| `BUN+Mg` | 6 |
| `BUN+Glucose+HCO3` | 4 |
| `BUN+Glucose+Mg` | 4 |
| `BUN+Na` | 3 |
| `BUN+Glucose+HCO3+Mg` | 3 |
| `BUN+Glucose+HCO3+Na` | 3 |
| `BUN+Creatinine` | 2 |
| `BUN+Mg+Na` | 2 |
| `BUN+Glucose+HCO3+Mg+Na` | 2 |
| `BUN+HCO3+Na` | 1 |
| `BUN+Creatinine+Glucose+Na` | 1 |
| `BUN+Glucose+Mg+Na` | 1 |
| `BUN+HCO3+Mg+Na` | 1 |

Every selected pattern contains BUN. Held-out excess is positive in 99/100 evaluations, with mean
+0.012013168205102115, minimum -0.002439086624919069 and maximum +0.023789761660853052. No interval
or p-value is attached to these repeated-development quantities.

## AUROC eligibility and artifact rescoring

All 141 candidates satisfy the AUROC-drop constraint on the reference run only. Reference maximum
drop is +0.0181501742360749 and median drop is +0.0028803506387620192. Across the 20 resplits,
eligible counts are:

`141, 138, 140, 141, 141, 141, 141, 140, 141, 141, 141, 137, 138, 141, 130, 141, 140, 121,
141, 140`.

The constraint changes 2 of 20 top-level selections relative to unrestricted selection, so it was
not non-binding during the search.

The repaired NPZ pins labels, record IDs, candidate names, fold assignments, the 141 x 20 x 5 delta
table, top-level and nested AUROCs, one reference OOF clean prediction per patient, all 141 reference
candidate prediction vectors and one reference per-patient paired difference for the frozen pattern.
Record IDs are unique; labels are binary; candidate and fold indices agree with JSON. Independent
raw prediction rescoring with scikit-learn reproduces clean, candidate and nested reference AUROCs
exactly. The maximum numerical disagreement across all independently rescored stored quantities is
**0.0**.

Post-repair SHA-256 hashes are:

- `m5v2_tables.npz`: `a97e256103df1c584bc1d934509cc869e7128766cb3556d4d856ff53037a8af6`
- `results.json`: `db657e1e4b68c0138da143c97e165b83e09fc41e4ff1c69e446cdc1e210b9ed8`
- `frozen_pattern.json`: `5d921e9d7b2512a3486da078464c6fd07414604d1ff943c5185a0a11d05fa3bf`
- `m5v2_run.log`: `825be12d2127186b1c0aab267946972cbb9512dfc5c4db46798454267d95c811`

## Claim corrections

- Remove “genuine interaction” and “superadditive interaction.” The supported statement is:
  “Glucose and Na provide additional development-stage damage when combined with BUN despite weak
  singleton excesses.”
- Do not compare v1's 58% and v2's 12.9% as like-for-like estimands or as an exact reduction in
  selection bias. State only that apparent shrinkage was substantially smaller under the v2
  stability-aware development procedure than under v1's naive selection procedure.
- Preserve the 11/20 knife edge and distinguish the more stable BUN-centred family from exact
  three-analyte membership.
- Describe all A+B quantities as development findings. Make no causal, clinical-utility,
  laboratory-order, coherence or natural-missingness claim.

## Quality gates

- Targeted M5-v2 tests: 37 passed.
- Ruff check: passed.
- Ruff format check: passed.
- Mypy strict on the normal 30-file project scope: passed.
- Full pytest: 415 passed.
- Git diff check: passed.

The final model was not fitted, the final holdout was not scored, M6 was not started, and nothing was
pushed.
