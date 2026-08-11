# M5-v2 Milestone Report — Stability-Aware Adversarial Failure Search

**Date:** 2026-08-11
**Design:** predeclared in [`M5_V2_DESIGN.md`](M5_V2_DESIGN.md) at commit `5562120`, before any v2 code ran
**Implementation:** `1de9feb`, repaired at `91262fd` · **Artifact provenance:** `git_sha=91262fd`, `git_dirty=false`
**Verdict:** **v2-STABLE — all four gates pass**
**Status:** A+B development phase complete. **Set C not loaded.** No confirmatory claim is made here.

M5-v1 is untouched and remains **M5-C**.

---

## 1. Result

**Frozen pattern: `BUN + Glucose + Na`**, selected in **11 of 20** development resplits.

| gate | result | value |
|---|---|---|
| **G1 — null-control sanity** | **PASS** | frozen pattern is in the target region; **zero** null-control patterns selected in 20 resplits |
| **G2 — majority stability** | **PASS (at the minimum)** | `Pi = 0.55`, exactly **11/20**, threshold 11 |
| **G3 — discrimination-silent** | **PASS** | reference-run AUROC drop **+0.01572** ≤ delta 0.02 |
| **G4 — detectability** | **PASS** | MDE **+0.00804** ≤ out-of-selection development estimate **+0.01201** |

All 141 candidates enumerated, 20 resplits, R = 5 amount-matched controls, n = 8,000 sets a+b.

**Every number in this report is a development estimate.** The 20 resplits reuse the same 8,000
patients and are not independent. No interval, p-value or effect claim is attached to any of them.

## 2. G2 passed by one resplit — read this before anything else

`Pi(BUN+Glucose+Na) = 11/20` against a threshold of 11. **One resplit flipping to
`BUN+Glucose` — which took 4/20 — would have produced 10/20 and a v2-DIFFUSE verdict.**

This is the narrowest possible pass and must be carried into every downstream statement. The gate
was predeclared as a strict majority and it was met; it was not met comfortably. The full
selection-frequency table:

| pattern | selections | region |
|---|---:|---|
| **`BUN+Glucose+Na`** | **11/20** | target |
| `BUN+Glucose` | 4/20 | target |
| `BUN+Glucose+Mg` | 1/20 | target |
| `BUN+Glucose+HCO3` | 1/20 | target |
| `BUN` | 1/20 | target |
| `BUN+Glucose+HCO3+Mg` | 1/20 | target |
| `BUN+Glucose+HCO3+Na` | 1/20 | target |

Every selection in every resplit contained `BUN`. The instability is entirely about *which
companions* join it, not about the core.

## 3. Apparent shrinkage under the stability-aware procedure

| | M5-v1 | M5-v2 |
|---|---:|---:|
| selection rule | maximise pooled discovery excess | 1-SE parsimony on per-fold means |
| apparent shrinkage | **58%** | **12.9%** |
| naive estimate | +0.01385 | +0.01379 |
| held-out estimate | +0.00587 (confirmation) | +0.01201 (out-of-selection procedure) |

The two columns are not like-for-like estimands: v1 evaluates one discovery winner on a separate
development partition, whereas v2 repeatedly re-selects a pattern on four folds and evaluates the
selected procedure on the fifth. Apparent shrinkage was substantially smaller under the v2
stability-aware development procedure than under v1's naive selection procedure; the percentages
must not be interpreted as an exact quantitative reduction in selection bias.

Across all 100 (resplit, held-out fold) pairs, the out-of-selection excess was positive in **99/100**,
mean +0.01201, range -0.00244 to +0.02379. The most common nested selections were
`BUN+Glucose+Na` 32/100, `BUN+Glucose` 24/100, `BUN` 11/100 and `BUN+Mg` 6/100. Every selected
pattern contained `BUN`; exact companion membership remained variable.

## 4. What the frozen pattern does

On the reference run `b = 0` (seed 20260809), which reproduces the M3 clean baseline exactly
(AUROC 0.8270, NLL 0.3151):

| quantity | clean | `BUN+Glucose+Na` withheld | change |
|---|---:|---:|---:|
| AUROC | 0.8270 | **0.8113** | **−0.0157** |
| NLL | 0.3151 | 0.3302 | +0.0151 |
| Brier | 0.0968 | 0.1009 | +0.0041 |
| calibration intercept | −0.010 | **+0.488** | +0.498 |
| calibration slope | 0.988 | 1.065 | +0.077 |
| mean predicted risk | 0.1397 | **0.1067** | −0.0330 |
| mean realized severity | 0 | 0.176 | — |
| mean cells removed / patient | 0 | 5.93 | — |

Against a development prevalence of 14.03%, mean predicted risk falls to 10.7%. **AUROC moves 0.016
— a change routine monitoring would not flag — while the model systematically under-predicts
mortality risk.** This is the M3-B signature reproduced from **three analytes and roughly six cells
per patient**, rather than from a seven-analyte panel.

## 5. `BUN` carries most of the development signal

Single-analyte development excesses on the reference run:

| analyte | excess NLL | AUROC drop | region |
|---|---:|---:|---|
| **BUN** | **+0.01021** | +0.0155 | target |
| Creatinine | +0.00168 | +0.0022 | target |
| K | +0.00068 | +0.0005 | target |
| Glucose | −0.00026 | −0.0004 | target |
| HCO3 | −0.00016 | +0.0003 | target |
| Na | −0.00011 | −0.0004 | target |
| Mg | −0.00052 | −0.0002 | target |
| PaCO2 / Platelets / WBC / pH / PaO2 / HCT | +0.00046 to −0.00113 | ≈ 0 | **null control** |

`BUN` alone yields +0.01021, about **74%** of the frozen pattern's +0.01379. `Glucose` and `Na`
alone are indistinguishable from the null controls (−0.00026 and −0.00011). Yet together with `BUN`
they reach +0.01379, against an additive prediction of +0.00984.

This arithmetic does not establish a statistical interaction or a valid additive decomposition,
because candidate and control losses share fitted models, patients and amount-matched random draws.
The supported statement is narrower: **Glucose and Na provide additional development-stage damage
when combined with BUN despite weak singleton excesses.** In v1 the appended analytes
(`TroponinI`, `Cholesterol`) were near-empty; here `Glucose` and `Na` are routinely measured
(severity 0.057 and 0.059 each). Exact three-analyte membership remains provisional at 11/20.

Descriptively, the v2 procedure estimate is numerically larger than the full seven-analyte panel's v1
confirmation estimate (+0.01201 versus +0.00713). These are different estimands on different
development partitions and do not support the claim that withholding fewer analytes is intrinsically
more damaging.

## 6. The null-control sanity gate did real work

Zero null-control patterns were selected in any of the 20 resplits, and on the reference run all six
null analytes sit within ±0.0011 of zero excess while `BUN` sits at +0.0102 — roughly a tenfold
separation. The procedure distinguished the target region from a region M5-v1 had shown to be null.

**This is necessary, not sufficient.** Passing G1 rules out one specific failure mode — a procedure
that finds "signal" everywhere. It does not validate the method, and it is not evidence that the
frozen pattern will replicate on unseen data.

## 7. Two predeclared expectations that were wrong, recorded as such

**`M5_V2_DESIGN.md` §7 predicted G4 was "more likely to fail than to pass."** It passed. The
prediction assumed the minimal core would carry an effect similar to full `BMP_like` (about +0.007),
which would have sat below the MDE. It did not: the sharper pattern concentrates more damage
(+0.01201), which is precisely the escape route the design named but declined to bank on. The
variance prediction was accurate — design estimated `sigma_Delta ≈ 0.21`, measured **0.2046**, giving
MDE **+0.00804** against a predicted ≈0.008.

On the **reference resplit only**, all **141/141** candidates were eligible at delta = 0.02 (maximum
drop +0.0182, median +0.0029). Across all 20 resplits, eligibility ranged from 121 to 141 candidates,
and the constraint changed the top-level choice in 2 resplits compared with unrestricted selection.
It therefore did bite during v2, although the frozen pattern and its net 11/20 count were unchanged.
G3 is both a property of the frozen reference result and an active predeclared search constraint.

## 8. Limitations

- **G2 passed by a single resplit.** Treat the specific three-analyte pattern as provisional; the
  `BUN`-containing *family* is far better supported than the exact membership.
- **Nothing here is confirmatory.** All 8,000 A+B patients are development, the resplits reuse them,
  and no interval or p-value is attached to any development quantity.
- The detectability analysis uses `sigma_Delta` from one reference run and a shrunken effect from
  nested re-selection. It is an honest projection, **not** a guarantee that the set-c test will
  resolve.
- **Analyte-set identity, never coherence.** Removal is whole-window, so per M3-B this identifies
  which analytes were withheld and cannot test co-occurrence structure or order events.
- Groups remain **reconstructed co-measurement clusters** (`*_like`), never verified laboratory
  orders.
- Synthetic withholding is not natural missingness and not deployment shift. One historical ICU
  dataset, one cutoff, no external validation.
- **The finding is not biologically surprising.** That withholding blood urea nitrogen degrades
  mortality-risk calibration will not astonish a clinician. The contribution is the **procedure** —
  stability-aware, parsimony-constrained, amount-matched, out-of-selection-calibrated — which found
  it automatically, separated it from null controls, and produced an out-of-selection development
  estimate before the final holdout test.
- No causal, deployment-utility, clinical-validation or clinician-intent claim.

## 9. Reproduce

```bash
python experiments/robustness/m5_v2_stability_search.py
```

Artifacts in `experiments/robustness/results/m5v2/`: `results.json` (full reference-run table for all
141 candidates, the per-resplit selections, the selection-frequency table, all 100 out-of-selection
components, detectability and the gate verdicts), `frozen_pattern.json`, and `m5v2_tables.npz` (the
full 141 x 20 x 5 delta table, top-level and nested AUROC tables, candidate identities, fold
assignments, reference-run predictions for all candidates, and reference-run per-patient differences
for the frozen pattern).

## 10. Decision and what is not done

**v2-STABLE.** A minimal, amount-controlled, discrimination-silent failure pattern —
`BUN + Glucose + Na` — is frozen, and the predeclared detectability analysis projects that set-c
could resolve it.

**Not done, and not authorised by this report:**

- The final frozen model of `M5_V2_DESIGN.md` §8 has **not** been fitted.
- **Set C has not been loaded, scored, or referenced by any statistic here, and remains locked.**
- Unlocking set-c for the single frozen test of §9 requires separate explicit approval.

## 11. Quality gate

Recorded here after the the reviewer M5-v2 audit; the earlier revision of this report omitted the counts.

| check | result |
|---|---|
| targeted M5-v2 tests (`tests/test_failure_search_v2.py`) | **37 passed** |
| full suite (`pytest`) | **415 passed** |
| `ruff check .` | clean |
| `ruff format --check .` | clean |
| mypy strict (`cliniverse twinbench`) | clean |
| working tree at artifact generation | clean, `git_dirty=false`, source `91262fd` |

### Audit repair recorded

The nested four-fold selection previously allowed the held-out fold to influence AUROC eligibility.
the reviewer repaired it at `91262fd`. The repair changed **16 of 100** nested choices. It did **not** change
the frozen pattern (`BUN+Glucose+Na`), the selection frequency (11/20), or any gate verdict. The
post-repair nested positivity is **99/100**, not the 100/100 stated in the pre-repair revision.
