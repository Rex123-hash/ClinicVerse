# M5 Milestone Report — Discrimination-Silent Reliability Failure Search

**Date:** 2026-08-11
**Design:** predeclared in [`M5_DESIGN.md`](M5_DESIGN.md) at commit `7749ff7`, before any search ran
**Implementation:** `f6b6e6b` · **Artifact provenance:** `git_sha=f6b6e6b`, `git_dirty=false`
**Classification:** **M5-C by the predeclared rule — the primary test did not pass**
**Status:** discovery + confirmation complete. Recovery arm not yet run. set-c not used.

---

## 1. Headline

The exhaustive search **did** find a large, reproducible, discrimination-silent reliability failure,
and it **did** show that the failure ranking transfers to unseen patients. It **did not** resolve the
selected configuration's excess from zero on the confirmation set. Under the predeclared rule, T1 is
the primary test and T1 failed, so **M5 is classified M5-C**.

| test | result | value |
|---|---|---|
| **T1 (primary)** — confirmation ΔNLL_excess of rank-1 | **FAIL** | +0.00587 **[−0.00174, +0.01365]**, p = 0.134 |
| T2 — confirmation ΔBrier_excess of rank-1 | FAIL | p = 0.507 |
| T3 — AUROC preservation on confirmation | **PASS** | drop +0.0104 ≤ δ = 0.02 |
| T4 — transfer of configuration identity | **PASS** | Spearman ρ = **+0.8648**, permutation p = 1.0×10⁻⁴ |

All 1,023 configurations were enumerated (`complete_enumeration: true`). n = 8,000, sets a+b,
discovery = 4,800 patients (folds 0–2), confirmation = 3,200 patients (folds 3–4), disjoint.

## 2. The predeclared taxonomy does not fit the observed combination

`M5_DESIGN.md` §15 defined M5-C as "T1 fails", glossed as *"after matching the amount of information
removed, which information is withheld does not measurably change reliability damage."*

**The classification rule is honoured — T1 failed, so this is M5-C — but that gloss is contradicted by
the data and must not be quoted.** T4 passed at ρ = +0.865 with the permutation p-value at its
10,000-permutation floor, and the descriptive contrast in §4 is large and one-sided. Which
information is withheld clearly does matter. What failed is the *magnitude* test for the *selected*
configuration at this sample size.

The predeclared outcome set anticipated "T1 passes, T4 fails" (M5-B) but not "T1 fails, T4 passes".
That is a gap in the predeclaration, recorded here rather than repaired retroactively. No test,
endpoint, constraint or selection rule was changed after seeing results.

## 3. What the primary test actually says

The locked top-5, selected on discovery folds 0–2 alone and written to `locked_selection.json`
before any confirmation statistic was computed:

| rank | configuration | discovery excess | confirmation excess | interval | AUROC | intercept |
|---:|---|---:|---:|---|---:|---:|
| 1 | `BMP_like+TroponinI` | +0.01385 | **+0.00587** | [−0.00174, +0.01365] (95%) | 0.8075 | +0.523 |
| 2 | `BMP_like+Cholesterol` | +0.01373 | +0.00843 | [−0.00117, +0.01784] (99%) | 0.8079 | +0.523 |
| 3 | `BMP_like+Cholesterol+SaO2` | +0.01361 | +0.00914 | [−0.00064, +0.01872] (99%) | 0.8068 | +0.517 |
| 4 | `BMP_like` | +0.01326 | +0.00713 | [−0.00238, +0.01668] (99%) | 0.8078 | +0.520 |
| 5 | `BMP_like+Cholesterol+TroponinI` | +0.01304 | +0.00830 | [−0.00165, +0.01819] (99%) | 0.8076 | +0.526 |

Ranks 2–5 carry Bonferroni-adjusted 99% intervals as predeclared, so their width is not comparable
to rank 1's 95% interval.

Every point estimate is positive and they cluster tightly around +0.006 to +0.009. None resolves from
zero. At n = 3,200 the paired bootstrap half-width is ≈ ±0.0077, which is the same order as the
effect being estimated. **M5 is underpowered for a per-configuration magnitude claim, not
uninformative about the phenomenon.**

### 3.1 The winner's curse is visible, and the search bought nothing

Rank-1 discovery excess +0.01385 shrank to +0.00587 on confirmation — a 58% reduction. That gap is
the cost of maximising over 1,023 dependent statistics, paid honestly by the split rather than
hidden.

More pointedly: **plain `BMP_like` (rank 4) has a *higher* confirmation excess (+0.00713) than the
"optimised" rank-1 selection (+0.00587).** The four configurations that outranked it on discovery all
consist of `BMP_like` plus a near-empty singleton — `TroponinI` (severity 0.002), `Cholesterol`
(0.003), `SaO2` (0.024) — which cannot plausibly add damage and instead jitter the discovery estimate
upward. The optimisation over the enumerated space added noise and cost interpretability while
finding nothing beyond its single-group core.

## 4. What the search did find — descriptive, and strong

**One co-measurement group accounts for the entire effect.**

| | mean confirmation excess NLL |
|---|---:|
| 512 configurations containing `BMP_like` | **+0.01005** |
| 511 configurations without `BMP_like` | **−0.00007** |

`BMP_like` appears in **50 of the top 50** configurations ranked by confirmation excess. Every other
singleton group is null: `ABG_like` −0.00076, `CBC_like` −0.00191, `hepatic_like` +0.00083,
`Lactate` +0.00158, and the four rare singletons within ±0.0003.

`BMP_like` is the empirically derived BMP-like co-measurement subset: BUN, Creatinine, Glucose,
HCO3, K, Mg, Na. Withholding it alone, on the 3,200 confirmation patients:

| quantity | clean | `BMP_like` withheld | change |
|---|---:|---:|---:|
| AUROC | 0.8179 | 0.8078 | **−0.0101** |
| calibration intercept | −0.141 | **+0.520** | +0.661 |
| calibration slope | — | 1.018 | — |
| mean predicted risk | 0.1432 | **0.0984** | −0.0448 |
| realized severity | 0 | 0.416 | — |

Against a confirmation prevalence near 14%, mean predicted risk falls to 9.8%. This reproduces the
M3-B signature — discrimination barely moves while predicted risk drifts systematically downward —
but here it is produced by withholding **one** co-measurement group, it is **amount-matched** against
equal-count random cell removal, and it is measured on patients that played no part in selecting it.

**This is the discrimination-silent failure the milestone set out to find.** What M5 could not do is
prove, at n = 3,200, that its *excess over amount-matched random removal* is non-zero.

## 5. Is the damage just "how much was removed"?

Partly, and the two metrics separate cleanly:

- Spearman(realized severity, **AUROC drop**) = **+0.942** — discrimination loss is almost entirely a
  function of how much was removed.
- Spearman(realized severity, **excess NLL**) = **+0.609** — the reliability excess is only moderately
  explained by amount, and it is already amount-matched by construction, so the residual reflects
  larger removals amplifying an identity effect rather than amount acting alone.

74.1% of all 1,023 configurations show positive confirmation excess (median +0.00284), so the typical
configuration damages probability reliability more than an equal-amount random removal — by a small
margin.

## 6. Constraint behaviour

The AUROC-preservation constraint bound as intended and was not vacuous: **517 of 1,023**
configurations were eligible on discovery, **755 of 1,023** on confirmation. The constraint therefore
excluded roughly half the space from selection, which is what makes the surviving failures *silent*
rather than merely large.

## 7. Exploratory, not confirmatory — read with care

Section 4's `BMP_like` contrast, the top-50 tabulation, the severity correlations and the
confirmation-ranked table in the artifact are **post-hoc analyses of the confirmation set**. The
confirmation patients were reserved to test one locked selection; using them to rank all 1,023
configurations and read off the best is precisely what the lock exists to prevent. These statements
are descriptive characterisations of a completed enumeration, **not** confirmed effects, and none of
them may be reported as a confirmatory result or given a p-value.

An honest confirmatory claim about `BMP_like` specifically would require a patient set not used here.

## 8. Limitations

- **Underpowered for the primary endpoint.** The confirmation half-width (±0.0077) is comparable to
  the effect (+0.006 to +0.009). A larger confirmation set, or a design that spends less of the
  cohort on discovery, is the obvious remedy.
- **Declared in advance and restated here:** the paired bootstrap resamples patients but treats the
  R = 3 control draws as fixed, so it does not propagate control-draw Monte-Carlo noise and is mildly
  optimistic.
- **Analyte-set identity, never coherence.** Removal is whole-window, so per M3-B this identifies
  which analytes were withheld and cannot test co-occurrence structure or order events.
- The groups are **reconstructed co-measurement clusters** (`*_like`), not verified laboratory orders.
- Synthetic withholding is not natural missingness and not deployment shift. One historical ICU
  dataset, one cutoff, one split assignment, no external validation.
- No causal, deployment-utility, clinical-validation or clinician-intent claim is made or supported.

## 9. Reproduce

```bash
python experiments/robustness/m5_failure_search.py
```

Artifacts in `experiments/robustness/results/m5/`: `results.json` (both full 1,023-row tables, all
intervals, all four tests, provenance), `locked_selection.json` (the lock, written before any
confirmation statistic), `m5_predictions.npz` (raw per-patient predictions and excesses for the clean
condition and every locked configuration). Every reported number is recomputable from the NPZ alone.

## 10. Decision

**M5-C by the predeclared rule.** The primary magnitude test failed; the transfer test passed
decisively; the discrimination-silent phenomenon is present and large in descriptive terms and is
concentrated almost entirely in a single co-measurement group.

The predeclared **recovery arm has not been run** and is not reported here.
