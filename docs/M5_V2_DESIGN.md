# M5-v2 Design — Stability-Aware Adversarial Failure Search

**Status:** FINAL PREDECLARATION. Committed before any v2 result exists.
**Date:** 2026-08-11
**Does not touch M5-v1**, which is closed at `5bfac1d` and remains **M5-C** exactly as reported.
**Inherits:** M1 disclosure semantics, the M2 model contract, the M3 calibration architecture
including the M3-B correction, and the M5-v1 withholding and amount-matched-control semantics.

Nothing below — search space, withholding semantics, control, scoring, selection rule, gates,
detectability analysis, final model definition or the set-c test — may be revised after inspecting
any v2 result.

---

## 1. Question and scope

M5-v1 established that configuration ranking transfers (Spearman rho = 0.8648) and that the
vulnerable region is concentrated in `BMP_like`, but its selection rule rewarded estimator noise: the
rank-1 pick shrank 58% from discovery to confirmation, and plain `BMP_like` outperformed the
"optimised" winner. v2 asks a sharper question under a selection rule that cannot be gamed the same
way:

> **Is there a minimal, reproducible core of analytes inside `BMP_like` whose withholding drives the
> discrimination-silent reliability failure — and is it stable enough to freeze as a single
> pre-registered hypothesis?**

v2's job on A+B is to **produce one frozen failure pattern and an honest detectability verdict.** It
is not to measure an effect.

## 2. Statistical status of A+B — binding

**All 8,000 A+B patients are development.** M5-v1 consumed folds 0-2 as discovery and folds 3-4 as
confirmation; no unused A+B patient exists, and re-splitting does not create one.

**Repeated resplits are a development-stability device, not an inferential one.** The 20 x 5 fold
estimates reuse the same 8,000 patients, so they are **not independent**, and no quantity derived
from them is a confirmatory standard error, confidence interval or p-value.

- Within one resplit, the five fold estimates are computed on **disjoint patient sets** but share
  heavily overlapping *training* data. Their dispersion is a **heuristic tolerance scale, not a valid
  standard error.** It is written `D(c)` and is used only to define a tie band.
- Across resplits, patients are fully reused. Resplits are aggregated **only** by counting how often
  each candidate is selected. Dispersions are never pooled across resplits and no interval is ever
  formed from the 100 estimates.
- The name "1-SE rule" is retained as the standard name of the procedure (Breiman et al., CART 1984).
  The quantity is a fold dispersion; the label is not a claim.

The restriction of the search space to `BMP_like` is **development-derived from M5-v1's exploratory
section** and carries no confirmatory status.

A+B emits exactly four things: a frozen pattern, a frozen test statistic, a frozen decision rule, and
a detectability verdict.

## 3. Search space

**141 candidates**, all competing in one pool:

| region | analytes | subsets |
|---|---|---|
| **target** — `BMP_like` | BUN, Creatinine, Glucose, HCO3, K, Mg, Na | 2^7 - 1 = **127** |
| **null control** — `CBC_like` | HCT, Platelets, WBC | 7 |
| **null control** — `ABG_like` | PaCO2, PaO2, pH | 7 |

Null-control regions were null in M5-v1 (confirmation excess -0.00191 and -0.00076). They are
**development-derived null controls**, placed in the same selection pool so that the procedure's
ability to separate signal from null is observable rather than assumed.

**Temporal windows are dropped from v2.** They may return only as a secondary ablation after the main
pattern is frozen, under a separate predeclaration.

## 4. Withholding and controls — unchanged from v1

- Deterministic **whole-window removal**: every observed cell of every named analyte across the
  truncated 24-hour window, values to NaN and observation-mask cells to false, applied to the cohort
  **before** feature construction. No RNG, no severity target.
- The control is the existing tested `LossCondition.CELL_RANDOM` path with `match_counts` set to the
  candidate's realized per-patient counts. **No new sampling logic is introduced for the control.**
- The control pool remains **all 23 eligible laboratory analytes**, including those withheld — M3 and
  v1 semantics, fixed here.
- **R = 5** independent control draws per candidate; seeds derived by hashing the base seed, the
  analyte names and the repetition index. The control arm's per-patient loss is the mean over the
  five draws.
- Every reported quantity is an **excess over this control**, never raw damage.

## 5. Scoring and selection

**Plain language.** For each candidate, in each fold: withhold it, and separately delete the same
number of randomly chosen laboratory cells from the same patients. Measure how much worse the
probabilities get beyond that random deletion. Average over the folds. Among all candidates
effectively tied with the best, keep the **smallest**. Repeat under 20 resplits and see which pattern
keeps winning.

**Mathematically.** For candidate `c`, resplit `b`, fold `k` with outer-test patients `P_{k,b}`:

```
Delta_{k,b}(c) = mean over i in P_{k,b} of [ l_i(c) - (1/R) * sum_r l_i(ctrl_r(c)) ]
```

with `l_i` the per-patient log loss. Within resplit `b`:

```
Deltabar_b(c) = (1/5) * sum_k Delta_{k,b}(c)
D_b(c)        = sd_k( Delta_{k,b}(c) ) / sqrt(5)
```

**Eligibility** (discrimination-silent constraint, delta = 0.02, inherited from v1), evaluated on the
resplit's pooled out-of-fold predictions — one number per candidate per resplit:

```
E_b = { c : AUROC_b(clean) - AUROC_b(c) <= delta }
```

**Selection within resplit `b` — one rule, no second penalty:**

1. `c_b^max = argmax over c in E_b of Deltabar_b(c)`
2. tie band `T_b = { c in E_b : Deltabar_b(c) >= Deltabar_b(c_b^max) - D_b(c_b^max) }`
3. `c_b* =` the member of `T_b` with the **fewest analytes**; ties broken by higher `Deltabar_b`, then
   lexicographically.

**Across resplits.** The 20 resplit seeds are `20260809 + b` for `b = 0..19`; **`b = 0` is the
reference run** used by §6 G3 and §7.

```
Pi(c)  = (1/20) * count of b with c_b* == c
c_star = argmax_c Pi(c)
```

The full selection-frequency table is reported.

## 6. Gates — the minimum set

Evaluated in order. A failure at any gate stops promotion and leaves set-c locked.

| gate | requirement | why essential |
|---|---|---|
| **G1 — null-control sanity gate** | `c*` must not lie in a null-control region | If the winner is drawn from a region v1 showed to be null, the procedure is not separating signal from noise on this data. Membership test, no numeric cutoff. **Passing G1 is necessary but does not by itself validate the method** — it rules out one specific failure mode, nothing more. |
| **G2 — majority stability** | `Pi(c*) >= 11/20` | A strict majority of the **20 predefined development resplits** must select the same pattern. These resplits reuse the same 8,000 A+B patients and are therefore **not independent samples**; the count is a development-stability measure, not an inferential one. Not a tuned cutoff — it is the definition of a majority. |
| **G3 — discrimination-silent** | `AUROC(clean) - AUROC(c*) <= delta = 0.02`, evaluated on the **reference run `b = 0` (seed 20260809)** using its **pooled 8,000-patient out-of-fold predictions** | This *is* the phenomenon under study; without it the finding is a loud failure, not a silent one. Pinning it to `b = 0` means G3 and `sigma_Delta` are read off the same single reference run, so neither can be shopped across resplits. |
| **G4 — detectability** | `MDE(n = 4000) <= Delta_oos` (§7) | Set C is single-use. A test that cannot detect the effect must not be run. |

Surviving constants: **delta = 0.02** (inherited from v1), **lambda = 1** (the standard 1-SE rule),
**11/20** (a majority), **B = 20** and **R = 5** (compute choices, not thresholds), **alpha = 0.05
one-sided** and **power = 0.80** (conventions).

**G1 failure is a null-control sanity failure, not a discarded run.** It is reported as a methods
finding, the pattern is not promoted, set-c stays locked, and the run, its artifact and the full
selection-frequency table are retained and published.

## 7. Set-C detectability analysis — conservative by construction

**Variance term — one reference run only.** `sigma_Delta` is computed from **exactly one predeclared
reference 5-fold out-of-fold run, seed 20260809** (resplit `b = 0`). That run assigns every A+B
patient exactly one out-of-fold prediction, giving **exactly one paired difference per patient**:

```
d_i          = l_i(c_star) - (1/R) * sum_r l_i(ctrl_r(c_star))
sigma_Delta  = sd( { d_i : i = 1..8000 } )
```

Predictions from the other 19 resplits are **never pooled into `sigma_Delta`**; doing so would
duplicate patients and understate the dispersion.

**Minimum detectable effect** at set-c's `n = 4,000`, one-sided alpha = 0.05, power 0.80:

```
MDE = (z_0.95 + z_0.80) * sigma_Delta / sqrt(4000) = 2.486 * sigma_Delta / 63.25
```

**Effect term — shrunken, never the naive winner.** `Deltabar(c*)` is the maximum of 141 correlated
estimates and is upward-biased; using it would make set-c look better powered than it is, which is
precisely the v1 failure. `Delta_oos` is computed by **nested out-of-selection evaluation**: within
each resplit, for each held-out fold `k`, re-run the §5 selection using only the other four folds,
then evaluate *that* pattern's excess on fold `k` alone. Average over all 20 x 5 = 100 (fold,
resplit) pairs.

This costs no extra model or prediction work — the `Delta_{k,b}(c)` table already exists and
re-selection is arithmetic on it.

`Delta_oos` is a **development quantity with no confidence interval and no p-value**, per §2. Both
`Delta_oos` and the naive `Deltabar(c*)` are reported side by side so the shrinkage is visible
**before** set-c is touched; v1 measured 58% shrinkage after the fact, v2 measures it in advance.

**G4 passes iff `MDE <= Delta_oos`.**

**Expected outcome, stated in advance so it cannot be spun later.** From v1's confirmation interval,
`sigma_Delta` is approximately 0.21, giving `MDE` approximately 0.008; v1's `BMP_like` confirmation
excess was +0.00713. **G4 is more likely to fail than to pass.** The correct response to failure is to
leave set-c locked and report that it cannot answer the question at this effect size. G4 can pass
only if the minimal core concentrates more damage per removed cell than the full seven-analyte panel
— a real possibility, and the main scientific reason to run v2, but not one to bank on.

## 8. The final frozen model, defined exactly

If and only if G1-G4 pass, one final pipeline is fitted **on A+B only**, before set-c is loaded.
Training on all of A+B without isolation is not permitted: the calibrator must never see its own
training data.

1. Partition all 8,000 A+B patients **once**, stratified by mortality, seed 20260809: **6,400
   final-model-training / 1,600 final-calibration.** Disjoint.
2. Fit the median imputer on the **6,400 clean** model-training rows only.
3. Fit XGBoost with the frozen M2/M3/M4 hyperparameters (`max_depth=5, learning_rate=0.05,
   min_child_weight=10, n_estimators=200, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0`),
   `random_state=20260809`, on the **6,400** only. The degenerate-booster guard must pass.
4. Fit the Platt calibrator on the **1,600 clean** calibration rows only, transformed by the
   train-fitted imputer. It is **never refitted** under any withholding condition, exactly as in M3,
   M4 and M5-v1.
5. Nothing is fitted on set-c, ever. Set-c is pure test.
6. The fitted pipeline, the frozen pattern `c*`, the frozen test statistic and the pass/fail rule are
   committed to git **before** set-c is loaded, with artifact hashes recorded.

**Circularity check, recorded now.** `c*` was selected using A+B labels and the final model is
trained on A+B labels. That is not circular for the confirmatory claim, because the claim concerns
`c*`'s effect on **set-c patients**, whose labels entered neither the selection nor any fitted object.

## 9. The set-C primary test — frozen now, not authorised here

**Set C remains completely locked. v2's A+B phase does not load it and this document does not
authorise loading it.** Unlocking requires separate explicit approval after the gate report. The test
is frozen here so that it cannot be designed after seeing anything.

**Statistic.** Per set-c patient `i`, with `c*` and the same R = 5 amount-matched control
construction:

```
d_i       = l_i(c_star) - (1/5) * sum_r l_i(ctrl_r(c_star))
Delta_C   = mean_i d_i,   n = 4000
```

**Interval.** Paired patient-level **percentile bootstrap** on `{d_i}`: resample set-c patient
indices with replacement, recompute the mean within each replicate, and take the **5th percentile**
of the replicate distribution as the **one-sided 95% lower confidence bound** `LB`. Single-class
resamples are skipped, matching project convention.

- **resamples: 10,000** (frozen; larger than v1's 2,000 because this is a single one-shot test)
- **seed: 20260809** (frozen)
- direction fixed in advance by the v1 development finding: the hypothesis is strictly `Delta_C > 0`

**Decision rule.** The primary test **passes iff `LB > 0`**, together with the constraint check
`AUROC(clean) - AUROC(c*) <= 0.02` on set-c. One load, one pattern, one bootstrap, one bound, one
pass/fail. No search, no selection, no re-test under a different delta, no second look.

**Monte-Carlo limitation, preserved explicitly.** The **R = 5 control draws are fixed across all
10,000 bootstrap replicates.** The interval therefore propagates patient-sampling uncertainty but
**not** control-draw Monte-Carlo uncertainty, and is mildly optimistic on that account. R = 5
averaging limits that term rather than the interval. This limitation is declared here and **must be
restated verbatim in any report of the set-c result.**

## 10. Artifact contract

The A+B run retains: schema id; git SHA and dirty flag; cohort fingerprint; split hashes; config
hash; the full candidate table with region tags; the `Delta_{k,b}` table for all 141 candidates,
20 resplits and 5 folds; per-resplit pooled AUROC and eligibility; the per-resplit selections; the
selection-frequency table `Pi`; the nested out-of-selection estimate and its 100 components; the
reference-run per-patient `d_i` for `c*`; `sigma_Delta`, `MDE` and the G1-G4 verdicts; and the
frozen pattern.

## 11. Claim rules — binding

- No v2 quantity computed on A+B is confirmatory. Say **"development estimate"**, never "effect" or
  "confirmed".
- Never attach an interval or p-value to `Deltabar`, `D`, `Pi`, or `Delta_oos`.
- Say **"analyte-set identity"**, never "structure", "coherence" or "event" (M3-B).
- Say **"discrimination-silent"** only with delta stated alongside.
- The groups remain **reconstructed co-measurement clusters** (`*_like`), never verified laboratory
  orders.
- Passing G1 rules out one failure mode; it is **not** a validation of the method.
- If the minimal core is clinically unsurprising, say so. The contribution is the **procedure** —
  stability-aware, parsimony-constrained, amount-matched, out-of-selection-calibrated — not the
  biology.
- No causal, deployment-utility, clinical-validation or clinician-intent claim.

## 12. Predeclared outcomes

- **v2-STABLE:** G1-G4 all pass. A frozen minimal pattern is promoted and set-c unlocking is proposed
  under §9.
- **v2-UNDERPOWERED:** G1-G3 pass, G4 fails. The pattern is frozen and published; **set-c stays
  locked**; the finding is that the effect is real enough to name but too small to confirm at
  n = 4,000.
- **v2-DIFFUSE:** G2 fails — no pattern reaches 11/20. The vulnerable region has no stable minimal
  core.
- **v2-SANITY-FAILURE:** G1 fails. Reported as a null-control sanity failure with the full
  selection-frequency table; no pattern is promoted.

**No outcome will be manufactured into another.** v2-UNDERPOWERED, v2-DIFFUSE and v2-SANITY-FAILURE
are legitimate results and will be reported in those words.
