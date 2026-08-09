# M2 Milestone Report — Representation Ablation

**Date:** 2026-08-09
**Task:** T1, in-hospital mortality, 24h information boundary
**Cohort:** PhysioNet/CinC 2012 sets a+b, n = 8,000, prevalence 14.03%. **set-c never loaded.**
**Recommendation:** **MODIFY** (§10)

Every number below is read from `experiments/baselines/results/m2/results.json`. None was typed
by hand. Raw out-of-fold predictions are retained in `predictions.npz` for independent
recomputation.

---

## Protocol

Outer 5-fold stratified CV over patients gives each patient exactly one out-of-fold prediction.
Hyperparameters are selected **inside** each outer training fold on a held-out inner validation
split (20%), from a compact grid fixed before any result was seen. Imputation and scaling are
fitted on training rows only. Comparisons use **paired patient-level bootstrap** (2,000 resamples)
on identical out-of-fold predictions — never overlapping standalone intervals.

| Provenance | Value |
|---|---|
| git SHA | `4bfed43a1b85` (working tree dirty: M2 code was uncommitted at run time) |
| cohort fingerprint | `f59c44f07556b7a6` |
| split hash | `21cbeab1b5bc308f` |
| config hash | `a80e2d2829a09a95` |
| seed | 20260809 |

**Search spaces.** LR: `C ∈ {0.01, 0.1, 1.0, 10.0}`. XGBoost:
`max_depth ∈ {3,5} × learning_rate ∈ {0.05, 0.1} × min_child_weight ∈ {1,10}`, with
`n_estimators=600` and early stopping at 50 rounds. Selection metric: inner-validation AUROC.

**SAPS-I / SOFA are omitted.** The PhysioNet documentation specifies no time window for the
outcome-file scores, and explicitly warns that its own SAPS calculator's output "do not always
match those given in the outcomes file." Cutoff-safety is therefore unverifiable. Including them
as 24h predictors would be scientifically invalid, so they are excluded rather than reported with
a caveat.

---

## Results

| run | #feat | AUROC [95% CI] | AUPRC [95% CI] | Brier | NLL | slope | intercept |
|---|---|---|---|---|---|---|---|
| prevalence | 0 | 0.4994 [0.4840, 0.5152] | 0.1401 [0.1319, 0.1481] | 0.1206 | 0.4054 | — | — |
| **LR mask-only** | 113 | **0.7278** [0.7128, 0.7423] | 0.2783 [0.2570, 0.3018] | 0.1114 | 0.3657 | 1.000 | −0.003 |
| **GBDT mask-only** | 113 | **0.7280** [0.7133, 0.7418] | 0.2734 [0.2533, 0.2965] | 0.1116 | 0.3652 | 1.002 | +0.014 |
| **LR values-only** | 185 | 0.8095 [0.7964, 0.8219] | 0.4273 [0.3981, 0.4564] | 0.0997 | 0.3276 | 0.979 | −0.037 |
| **GBDT values-only** | 185 | 0.8233 [0.8109, 0.8352] | 0.4455 [0.4151, 0.4783] | 0.0972 | 0.3174 | 0.954 | −0.008 |
| **LR values+mask** | 298 | 0.8240 [0.8121, 0.8359] | 0.4511 [0.4205, 0.4806] | 0.0969 | 0.3182 | 0.965 | −0.057 |
| **GBDT values+mask** | 298 | **0.8323** [0.8206, 0.8442] | **0.4627** [0.4331, 0.4933] | 0.0956 | 0.3116 | 1.018 | +0.104 |

Supplementary (not part of the three-way contract):

| run | #feat | AUROC [95% CI] | AUPRC | Brier | NLL |
|---|---|---|---|---|---|
| statics-only LR | 5 | 0.6310 [0.6127, 0.6493] | 0.2183 | 0.1173 | 0.3932 |
| statics-only GBDT | 5 | 0.6789 [0.6625, 0.6951] | 0.2343 | 0.1150 | 0.3818 |
| values+mask+statics LR | 303 | 0.8309 [0.8188, 0.8424] | 0.4586 | 0.0959 | 0.3141 |
| values+mask+statics GBDT | 303 | **0.8397** [0.8286, 0.8506] | 0.4654 | 0.0950 | 0.3081 |
| values-only **stochastic-imputation** LR | 185 | 0.7984 [0.7853, 0.8115] | 0.4174 | 0.1008 | 0.3321 |
| values-only **stochastic-imputation** GBDT | 185 | 0.8088 [0.7962, 0.8221] | 0.4206 | 0.0997 | 0.3266 |

## Paired differences (identical patients, identical folds)

| Comparison | Model | AUROC Δ [95% CI] | AUPRC Δ [95% CI] |
|---|---|---|---|
| VALUES+MASK − VALUES ONLY | LR | **+0.0146** [+0.0089, +0.0204] * | +0.0238 [+0.0139, +0.0346] * |
| VALUES+MASK − VALUES ONLY | GBDT | **+0.0090** [+0.0043, +0.0137] * | +0.0172 [+0.0030, +0.0301] * |
| VALUES ONLY − MASK ONLY | LR | +0.0817 [+0.0655, +0.0975] * | +0.1490 [+0.1203, +0.1764] * |
| VALUES ONLY − MASK ONLY | GBDT | +0.0953 [+0.0810, +0.1101] * | +0.1721 [+0.1427, +0.2009] * |
| VALUES+MASK − MASK ONLY | LR | +0.0962 [+0.0828, +0.1095] * | +0.1728 [+0.1457, +0.1973] * |
| VALUES+MASK − MASK ONLY | GBDT | +0.1042 [+0.0909, +0.1187] * | +0.1893 [+0.1618, +0.2163] * |

`*` = 95% paired interval excludes zero.

## Residual-missingness control — the most consequential result

Median imputation writes the training median into every unmeasured cell, so a model can detect
"exactly the median" and reconstruct the missingness indicator. Stochastic imputation samples from
the training marginal, removing that signature. The gap bounds how much of values-only performance
is recoverable missingness information rather than physiology.

| Comparison | Model | AUROC Δ [95% CI] | AUPRC Δ [95% CI] |
|---|---|---|---|
| values-only median − values-only stochastic | LR | **+0.0111** [+0.0059, +0.0162] * | +0.0098 [+0.0008, +0.0193] * |
| values-only median − values-only stochastic | GBDT | **+0.0145** [+0.0081, +0.0207] * | +0.0249 [+0.0098, +0.0401] * |

**For GBDT, the implicit missingness signal leaking through median imputation (+0.0145 AUROC) is
larger than the explicit gain from adding the whole 113-feature mask block (+0.0090 AUROC).**

A "values-only" baseline that uses median imputation — standard practice throughout this
literature — is therefore already absorbing more measurement-pattern information than it would
gain from being handed explicit indicators. This is a methodological finding about how
values-versus-mask ablations are conducted, and we did not anticipate it.

---

## Decision-gate answers

**1. Is mask-only AUROC 0.7224 reproduced and directly comparable?**
Yes, and slightly exceeded. E-002 reported **0.7223745892** with a fixed `C=1.0`. Under the M2
protocol with `C` selected inside each training fold, mask-only reaches **0.7278** (LR) and
**0.7280** (GBDT). The difference is attributable to hyperparameter selection, not to a change in
the data or the boundary. **The finding survives directly comparable, leakage-safe evaluation.**

**2. Best mask-only model.** GBDT at 0.7280 [0.7133, 0.7418]; LR is statistically
indistinguishable at 0.7278. Mask-only is essentially linear — trees buy nothing here.

**3. Best values-only model.** GBDT at 0.8233 [0.8109, 0.8352], clearly above LR's 0.8095.

**4. Best values+mask model.** GBDT at 0.8323 [0.8206, 0.8442]. Adding statics reaches 0.8397.

**5. How large is VALUES+MASK − VALUES ONLY?**
**+0.0090 AUROC** (GBDT) and **+0.0146** (LR). Both exclude zero, so the effect is real — but it
is small, and for GBDT it is smaller than the imputation artefact in §Residual-missingness.

**6. How much predictive information remains without any values?**
Substantial. Mask-only reaches AUROC 0.728 and AUPRC 0.278 against a prevalence floor of 0.4994 /
0.1401, using zero clinical measurements. It also beats statics-only (0.6310 / 0.6789), i.e.
measurement patterns carry more than age, sex, weight, height and ICU type combined. We report no
ratio to full-model skill.

**7. Is the measurement-policy-shortcut framing still strong enough to be the central hook?**
**Partly — and it must be restated more precisely.**

- *Supported:* a model with no clinical values at all reaches 0.728. Measurement presence is a
  genuine, large signal.
- *Supported and sharpened:* models exploit measurement patterns **even when you try to withhold
  them**, via imputation. That is the more interesting and less obvious claim.
- *Not supported:* the strong form — "measurement patterns are what these models mostly run on".
  Given values, explicit patterns add only ~0.009 AUROC. Values dominate
  (VALUES ONLY − MASK ONLY = +0.095).

This maps to **Outcome B** in the pre-declared interpretation rules, with an unanticipated
addition from the imputation control. We are not forcing the shortcut story.

**8. Does GBDT dominate?**
For values-bearing representations, yes (+0.014 to +0.008 AUROC over LR). For mask-only, no —
LR and GBDT are identical to three decimal places. No deep model is justified: nothing indicates
the tabular ceiling has been reached by architecture rather than by information.

**9. Any result that weakens the thesis?**
Yes, two, both reported rather than buried:
- The explicit values+mask gain (+0.0090 GBDT) is small. A referee could fairly say measurement
  patterns are largely redundant once physiology is available.
- Values-only is much stronger than mask-only. Any framing implying models are "mostly reading the
  care process" is not supported.

**10. GO / MODIFY / PIVOT.** **MODIFY.**

---

## What this means for M3/M4

The acquisition experiment becomes *more* relevant, not less, and its emphasis shifts.

The interesting quantity is no longer "how much does the mask add" — it is **how a model's
performance and calibration behave when the measurement-pattern shortcut is disrupted**, which is
exactly what support-blind replay tests. M2 shows models absorb that shortcut even through
imputation, so an acquisition policy evaluated under support-aware replay is being scored in a
regime where the shortcut is freely available.

Two concrete changes to the M3/M4 plan:

1. **Imputation strategy becomes an experimental axis, not a fixed detail.** Median versus
   stochastic imputation changes GBDT AUROC by more than the entire mask block does. Acquisition
   results must be reported under both, or the imputation choice silently sets the conclusion.
2. **Calibration is where the story may be strongest.** GBDT values+mask has calibration slope
   1.018 and intercept +0.104 — the worst intercept among the core three — while mask-only is
   near-perfect (1.000 / −0.003). Whether calibration degrades faster than discrimination under
   budget pressure (H3) is now the most promising open question.

## Honest limitations

- One dataset, one cutoff, one binning. No external validation.
- Values-only cannot be made perfectly free of missingness information; the stochastic variant
  bounds but does not eliminate it.
- The XGBoost early-stopping refit reuses the inner split for its stopping signal, so the final
  model is fitted on 80% of each outer training fold rather than 100%. This is conservative — it
  slightly understates GBDT — and is applied identically to every representation.
- Development cohort only. **set-c remains quarantined.**
- Associational throughout. Nothing here is causal, and nothing claims a model infers clinician
  intent.

## Artifacts

```
experiments/baselines/results/m2/
  results.json        metrics, intervals, per-fold hyperparameters, provenance
  predictions.npz     raw out-of-fold predictions, labels, record ids
  figures/            baseline comparison, representation contrast, reliability
```

Reproduce:

```bash
python experiments/baselines/m2_representation_ablation.py --folds 5 --n-boot 2000
python experiments/baselines/m2_figures.py
```
