# M5 Design — Discrimination-Silent Reliability Failure Search

**Status:** PREDECLARED. Written and committed **before** any M5 search result exists.
**Date:** 2026-08-11
**Inherits:** repaired M1 disclosure semantics, the M2 model contract, the M3 calibration
architecture *including the M3-B correction*, and the M4 acquisition evaluator.

Nothing below — search space, loss semantics, control, primary endpoint, constraint, split,
selection rule, statistics or success/null definitions — may be revised after inspecting any
comparative result. If a decision turns out to be wrong, it is reported as a limitation, not
silently changed.

---

## 1. Central question

> Are there combinations of withheld clinical information that damage the model's **probability
> reliability** far more than an equal *amount* of randomly withheld information, **while ordinary
> discrimination stays close to its clean value** — and does the identity of those combinations
> generalize to patients that were not used to find them?

M5 is the first Cliniverse milestone in which the failure condition is **discovered by search rather
than predeclared by hand**. Handoff §16 records that no automated failure search exists in this
repository; M5 is that stage.

### 1.1 Motivation from M3

M3 found, at its highest severity, AUROC 0.8270 → 0.8002 (a drop a monitoring dashboard would
plausibly ignore) while the calibration intercept moved −0.010 → **+0.573** and mean predicted risk
fell to 0.0944 against 14.03% prevalence. That condition was chosen by hand. M5 asks whether such
**discrimination-silent** reliability failures can be found systematically, and whether *which*
information is withheld matters after the *amount* is controlled for.

M3-B is the standing warning: an earlier structural hypothesis collapsed once an exact control was
applied. M5 is designed so that the same collapse, if it happens again, is visible and reportable.

## 2. Hypothesis

> **H5.** There exist analyte-group withholding configurations whose **excess** reliability damage
> over an amount-matched random control is greater than zero on patients not used to select them,
> while AUROC remains within a predeclared margin of its clean value on the same patients; and the
> *identity* of damaging configurations transfers from discovery patients to confirmation patients.

> **H5₀ (null).** After matching the amount of information removed, reliability damage does not
> depend on which information was removed. Operationally: the selected configuration's confirmation
> excess CI includes zero, and discovery/confirmation excess rankings are uncorrelated.

**Both outcomes are reportable.** H5₀ is a useful finding: it would mean a single volume-of-
missingness monitor is sufficient and no configuration-specific audit is needed. Instability and
fragility will **not** be manufactured.

## 3. What M5 is NOT — binding

- **Not** a claim about prospective clinical test ordering, recommendation, or intervention.
- **Not** an adversary model. No attacker is postulated. This is a **pre-deployment audit** of a
  frozen model against information configurations the deployment environment could plausibly produce.
- **Not** an event-structure experiment. Removal is **whole-window analyte removal**, exactly as in
  M3. Per M3-B, this identifies **analyte-set identity**, never co-occurrence coherence.
- **Not** a causal, deployment-utility or clinical-validation claim.
- **Not** evidence about real missingness. Removal is synthetic, specified and seeded by us.
- The groups remain **reconstructed co-measurement clusters** (`*_like`), never verified orders.

## 4. Frozen model contract

Unchanged from M2/M3/M4 and **not re-tuned, re-fitted or re-selected for M5**:

- XGBoost on `values_mask`: `max_depth=5, learning_rate=0.05, min_child_weight=10,
  n_estimators=200, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0`.
- Cutoff 24 h. 5 patient-level stratified outer folds, seed 20260809.
- Per fold: **4,800 model-train / 1,600 calibration / 1,600 outer test**.
- Median imputer fitted once per fold on clean model-train rows; **never refitted** under any
  withholding configuration.
- Static Platt calibrator fitted on the **clean** calibration partition; **never recalibrated** on
  any stressed condition.
- Information is removed from the **cohort**, before feature construction, so a removed cell is
  indistinguishable from one never measured. No post-hoc feature deletion.
- **set-c is never loaded.** `load_cohort()` defaults to sets a+b.

The degenerate-booster guard (`FoldModel.n_features_used`) remains active.

## 5. Search space — exhaustive

The action catalogue partitions the 23 eligible laboratory analytes into **10 co-measurement
groups**: `BMP_like, CBC_like, hepatic_like, ABG_like, Lactate, SaO2, Albumin, TroponinT, TroponinI,
Cholesterol`.

The search space is **every non-empty subset of those 10 groups: 2¹⁰ − 1 = 1,023 configurations.**

It is **enumerated exhaustively**, not sampled and not optimized. This is a deliberate design
choice: with the whole space evaluated, the discovered configuration's standing is an **exact rank
within a fully known distribution**, so no stochastic-search or early-stopping artefact can be
mistaken for a finding, and the multiplicity story (§10.3) requires no assumption.

Vitals and ventilator settings are never eligible, matching M3: they are near-continuously monitored
and removing them would not correspond to a group-level information event.

## 6. Withholding semantics — deterministic

For a configuration `S`, every **observed** cell of every member analyte of every group in `S` is
removed across the entire truncated 24-hour window: values set to NaN, observation-mask cells set to
false.

This is **deterministic** — no RNG, no severity target, no per-patient draw. It is the M3
`group_structured` semantics with the group set **specified** instead of drawn to hit a severity.
Realized severity is therefore a *consequence* of `S`, recorded per patient, never a target.

## 7. Amount-matched random control — the load-bearing comparison

"Configuration `S` is damaging" is uninteresting if `S` simply removed more cells. Every
configuration is therefore compared against a control that removes **exactly the same number of
cells from the same patient**, drawn uniformly from that patient's observed eligible laboratory
cells.

- The control is the existing, tested, review-audited `LossCondition.CELL_RANDOM` path with
  `match_counts` set to the configuration's realized per-patient counts. **No new sampling code is
  introduced for the control.**
- The control draws from **all 23 eligible analytes**, including those in `S`. This is the M3
  count-matched semantics and is retained unchanged.
- **R = 3** independent control draws per configuration. Each draw's seed is derived by hashing the
  base seed together with the configuration's group names and the repetition index, so the seeds are
  reproducible across machines and well separated; a linear stride would risk colliding with the
  per-patient seed arithmetic inside `information_loss` once a thousand configurations are in play,
  which would silently correlate control draws across configurations.
- The control arm's per-patient loss is the **mean over the 3 draws**. Three draws damp control-draw
  Monte-Carlo noise, which is a genuine search-overfitting channel when 1,023 configurations compete
  for the maximum.

Every reported primary quantity is an **excess over this control**, never raw damage.

## 8. Endpoints

Let `L(·)` denote a per-patient loss and let `control(S)` be the mean over the R control draws.

**Primary — ΔNLL_excess:**

> `ΔNLL_excess(S) = NLL(S) − NLL(control(S))`
>
> **Higher is worse** (more damaging than an equal amount of random removal). Configurations are
> ranked in **descending** ΔNLL_excess.

**Co-primary — ΔBrier_excess**, defined identically.

**Constraint — AUROC preservation.** A configuration is **eligible** only if

> `AUROC(clean) − AUROC(S) ≤ δ`, with **δ = 0.02**, evaluated on the same patient set.

δ is fixed now, before any result. The constraint is what makes a discovered failure *silent* rather
than merely damaging: a configuration that destroys discrimination would be caught by ordinary
monitoring and is not the phenomenon under study. Ineligible configurations are still evaluated and
retained in the artifact; they are excluded only from selection.

**Direct secondary diagnostics** (reported, never ranked, never used for selection): calibration
intercept, calibration slope, mean predicted risk, realized severity (removed cells per patient),
AUROC, AP, and the corresponding control-arm values.

AUROC is deliberately **not** an endpoint. M3 established that discrimination can stay nearly flat
while probability reliability degrades; ranking by AUROC would search for the wrong failure.

## 9. Discovery and confirmation

Patients are split by **outer fold**, so the two sets are disjoint by construction:

| set | folds | patients | role |
|---|---|---|---|
| **discovery** | outer-test partitions of folds **0, 1, 2** | ~4,800 | rank all 1,023 configurations; select |
| **confirmation** | outer-test partitions of folds **3, 4** | ~3,200 | estimate the effect for the locked selection |

Each patient is always scored by **its own fold's** frozen model, imputer and calibrator, exactly as
in M3/M4. No patient is ever scored by a model trained on itself.

**Disclosed nuance, recorded now rather than discovered later.** The fold-3 and fold-4 models were
trained on patients that include discovery-set patients. That is unavoidable in a 5-fold design and
it does **not** bias the confirmation estimate: the frozen models are identical across all 1,023
configurations, so they cannot favour any particular configuration, and the confirmation patients'
labels are never used to select one. What confirmation establishes is that the *selected
configuration* damages reliability on **patients not used to select it** — not that it does so under
a model trained without them.

### 9.1 Selection and lock — binding

1. Compute the primary endpoint for all 1,023 configurations on the **discovery** set only.
2. Restrict to configurations satisfying the §8 AUROC constraint **on the discovery set**.
3. Rank by descending discovery ΔNLL_excess and take the **top 5**.
4. **Write the locked top-5 to the artifact before any confirmation statistic is computed.** The
   run does this as an explicit ordered step, and the locked list is stored under its own artifact
   key so that the lock is auditable after the fact.
5. Only then compute confirmation statistics.

The **rank-1 configuration is the single primary confirmatory test.** The remaining four are
supporting evidence.

## 10. Statistics

### 10.1 Paired patient-level bootstrap

Confirmation intervals use a **paired** patient-level bootstrap: within each replicate the same
resampled patient indices are used for the configuration arm and the control arm, and the difference
is recomputed inside the replicate. **2,000 replicates, seed 20260809**, percentile 2.5/97.5.

NLL and Brier are patient means, so per-patient losses are differenced once and averaged on each
resample; this is algebraically identical to recomputing both arms inside every replicate and is the
same optimization already regression-tested in M4. Aggregated quantities are never resampled
independently.

**Declared limitation of this interval.** The bootstrap resamples patients and treats the R control
draws as fixed. It therefore propagates patient-level uncertainty but **not** the Monte-Carlo
uncertainty of the control draw itself, so the interval is mildly optimistic. R = 3 averaging is what
limits that term rather than the interval. This is stated now, in advance, and must be restated in
the milestone report rather than discovered by a reviewer.

### 10.2 Predeclared confirmatory tests

| # | test | role | passes when |
|---|---|---|---|
| **T1** | confirmation ΔNLL_excess of the **rank-1** configuration | **primary** | 95% percentile CI excludes zero, in the damaging (positive) direction |
| **T2** | confirmation ΔBrier_excess of the rank-1 configuration | secondary confirmatory | 95% CI excludes zero, positive |
| **T3** | AUROC constraint for the rank-1 configuration on the **confirmation** set | constraint check | `AUROC(clean) − AUROC(S) ≤ 0.02` |
| **T4** | Spearman ρ between discovery and confirmation ΔNLL_excess across **all 1,023** configurations | secondary confirmatory | ρ > 0 with one-sided permutation p < 0.05 |

T4 is the strongest single test of H5₀: it asks whether configuration *identity* transfers, using the
entire enumerated space rather than the selected tail, so it cannot be gamed by the selection rule.
Its inference is a **one-sided permutation test with 10,000 permutations** (seed 20260809), which is
exact under the null of no association. A configuration-level bootstrap CI is also reported but is
flagged as **approximate**, because configurations share groups and are therefore not independent.

**Multiplicity mechanics.** T1 is the single primary test and takes no correction. T2 and T4 are
reported as a two-member secondary family with **Holm–Bonferroni** correction. The four supporting
configurations in the locked top-5 are reported at **Bonferroni-adjusted 99% intervals**
(percentile 0.5/99.5, i.e. 0.05/5), so that the whole locked set can be read at a family-wise 5%
level without recomputing at five different levels. T3 is a constraint check, not a hypothesis test,
and is never counted in the family.

### 10.3 Multiplicity

Because the space is enumerated exhaustively, the selection is a **maximum over 1,023 dependent
statistics on the discovery set**. No correction is applied to the *discovery* ranking, because no
inference is drawn from it — it is a selection device only. All inference is drawn on the
**confirmation** set for a selection that was locked before those numbers existed, which is the
standard and honest way to pay for a search. The full 1,023-row discovery and confirmation
distributions are retained in the artifact so that any reader can locate the selected configuration
within them.

## 11. Recovery arm — predeclared, runs after the discovery checkpoint

For the rank-1 confirmed configuration only, the discovered withheld information becomes the
**hidden support** handed to the existing M4 acquisition evaluator: a deterministic masking
mechanism whose hidden mask is exactly the configuration's removed cells. The existing M4 policies
(`no_acquisition`, `random_uniform_all`, `random_train_frequency`, `fixed_domain_order`,
`greedy_eig`, `greedy_eig_per_cost`) then attempt to buy the information back under the predeclared
M4 budget grid `β ∈ {0, .1, .2, .3, .4, .5, .75, 1}`, `support_blind`, `shared_plus_marginal`.

Reported: NLL, Brier, calibration intercept and mean predicted risk versus budget, and the budget at
which the confirmation excess ceases to exclude zero. **No new policy is introduced and no policy is
re-tuned.** All disclosure, costing, boundary enforcement and budget accounting remain inside the
tested `DisclosureEngine`; M5 supplies only the hidden mask.

This arm is predeclared here so it cannot be designed after seeing the discovery result, but it is
**executed and reported after** the discovery + confirmation checkpoint.

## 12. Set C — remains locked

**set-c is not used in M5.** It is not loaded, not scored, and not referenced by any M5 statistic.
The `load_cohort()` default and the `final_holdout()` unlock token both remain in force. Any future
use requires a separate, explicit, pre-registered decision that is not part of this design.

## 13. Artifact contract

The run writes a machine-readable artifact retaining: schema id; git SHA and dirty flag; cohort
fingerprint; split hash; config hash; the full 1,023-row discovery table and the full 1,023-row
confirmation table (both arms, all endpoints and diagnostics); realized per-patient severity
summaries; the **locked top-5 under its own key**; every confirmatory interval; the Spearman
generalization statistic; the clean-reference metrics for both patient sets; and raw per-patient
predictions for the clean condition and for every locked configuration and its control draws, so
that every reported number is independently recomputable from predictions alone.

## 14. Claim rules — binding

- Report **excess over amount-matched random**, never raw degradation, as the headline.
- Say **"analyte-set identity"**, never "structure", "coherence" or "event" (M3-B).
- Say **"discrimination-silent"** only when the δ constraint is actually satisfied on the
  confirmation set, and state δ alongside it.
- Say **"withheld"** or **"removed"**, never "the model was attacked".
- Never present a discovery-set number as a result. Discovery numbers are selection inputs.
- Never claim the discovered configuration is what a real hospital would lose.
- No causal, deployment-utility, clinical-validation or clinician-intent claim.
- If H5₀ holds, report it as the finding, in those words.

## 15. Predeclared outcomes

- **M5-A (strong):** T1, T2 and T4 all pass (T2/T4 after Holm), and T3 holds. A named,
  amount-controlled, out-of-sample-confirmed, discrimination-silent reliability failure exists and
  configuration identity generalizes.
- **M5-B (partial):** T1 passes but T4 fails, or T1 passes with T3 violated. A specific damaging
  configuration exists but either its identity does not transfer across the space or it is not
  silent. Reported as a narrower result with the failing test named.
- **M5-C (null):** T1 fails. After amount matching, which information is withheld does not
  measurably change reliability damage. Reported as-is: a single volume-of-missingness monitor
  suffices, and no configuration-specific audit is warranted.

**The null will not be manufactured into a finding, and a finding will not be manufactured out of a
null.**
