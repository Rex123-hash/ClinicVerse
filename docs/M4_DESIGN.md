# M4 Design — Acquisition-Policy Ranking Stability

**Status:** PREDECLARED. Written and committed **before** any substantive M4 policy evaluation.
**Date:** 2026-08-09
**Inherits:** repaired M1 disclosure semantics, M2 model contract, M3 calibration architecture,
and the M4 contract in `ADVERSARIAL_REVIEW_3.md`.

Nothing below — policies, budget grid, cost regimes, primary metric, ranking statistic, bootstrap
protocol, success/null definitions — may be revised after inspecting comparative results.

> **Review #4 audit note (2026-08-11).** This remains the historical predeclaration. The
> artifact key `greedy_eig` is retained, but the implemented score is described after audit as a
> **surrogate expected-entropy-reduction heuristic**, not true EIG. The original batched
> `fixed_domain_order` implementation incorrectly reapplied its first priority instead of advancing
> through this document's declared sequence; Review #4 repaired that implementation defect and
> reran M4 without changing the sequence or any scientific choice below.

---

## 1. Central question

> Does the acquisition policy judged "best" remain best when the retrospective evaluation
> assumptions change?

M4 is **not** an attempt to invent a better acquisition algorithm. The contribution under test is
an **evaluation** result: whether benchmark design choices determine the scientific conclusion
about which acquisition strategy is preferable.

**We do not assume ranking reversals exist.** Outcome C (stable rankings) is a legitimate and
reportable result, and would reposition Cliniverse as a robustness certification rather than a
discovery of fragility.

## 2. Retrospective semantics — binding

M4 uses **sequential selective disclosure / retrospective replay**, exactly as repaired in M1.

A policy requests disclosure of eligible hidden historical information. This is **not** prospective
clinical test ordering, not a test recommendation, not a clinical intervention, and does not
generate future test results. Language implying any of those is forbidden in code, docs and
figures.

Action groups (`BMP_like`, `CBC_like`, `ABG_like`, `hepatic_like`, and singleton sends) are
**reconstructed co-measurement feature groups**, not verified clinical orders.

### 2.1 Action semantics (explicit)

| aspect | definition |
|---|---|
| what an action requests | disclosure of all hidden cells of one catalogue group, at timestamps at or before the current epoch boundary |
| what becomes visible | those cells only; values and mask both update |
| overlaps | none — the catalogue partitions the eligible variables (enforced by `PanelCatalogue`) |
| repeated action | permitted; charged in full; discloses nothing new |
| partially disclosed group | remaining hidden members of that group are disclosed; already-visible members are unaffected |
| failed / unavailable request | **charged in full, discloses nothing** — no free probing |
| already fully disclosed | charged in full, discloses nothing |
| beyond boundary | never disclosed; becomes eligible only after `advance_epoch()` |

## 3. Policy information boundary — binding

Under **support-blind**, before choosing an action a policy must never receive: the hidden support,
`S_hidden`, hidden values, future availability, the number of hidden values, an availability-filtered
candidate list, the natural-vs-synthetic distinction, or the outcome label.

The policy chooses **before** the result is returned.

**Leakage test (mandatory).** A test perturbs hidden values while holding all policy-visible state
constant and asserts the selected action sequence is unchanged. Any policy that fails this is not
admissible.

## 4. Support protocols

| protocol | role |
|---|---|
| **`support_blind`** | **PRIMARY.** The fair protocol. Any action may be requested; unavailable requests cost full price. |
| `support_aware` | **DIAGNOSTIC ONLY.** An availability oracle. Never described as deployable, and never used for a headline claim. |

The scientific question includes whether policy ordering changes when the availability oracle is
removed.

## 5. Policies

### 5.1 Mandatory references
1. **`no_acquisition`** — budget-zero reference.
2. **`random_uniform_all`** — support-blind, uniform over all legal actions.
3. **`random_train_frequency`** — support-blind; action probabilities fitted on **training folds
   only**. The serious random comparator.
4. **`fixed_domain_order`** — predeclared deterministic order, authored by the engineering team
   from observed measurement frequency. Called **domain-motivated**, never clinician-derived.
   Order: `BMP_like, CBC_like, ABG_like, Lactate, hepatic_like, SaO2, Albumin, TroponinT,
   TroponinI, Cholesterol`.
5. **`random_support_oracle`** — support-aware diagnostic only.
6. **`full_information_ceiling`** — all eligible information disclosed; diagnostic upper reference,
   **not** budget-comparable and excluded from ranking tables.

### 5.2 Strong adaptive policies (at least one must survive)

7. **`greedy_eig`** — myopic expected information gain about the outcome.
   For each affordable action `a`, the policy integrates over a **predictive distribution for the
   unknown hidden value** using a fixed 3-point quadrature at the training 25th/50th/75th
   percentiles of each member variable (equal weights), computes the model's outcome probability
   under each imputed completion, and scores
   `EIG(a) = H(p_now) − mean_k H(p_k)` where `H` is binary entropy.
   It then selects `argmax EIG`. **All quantiles come from training folds only.** The hidden value
   is never consulted.

8. **`greedy_eig_per_cost`** — the same score divided by the action's cost under the active regime.
   This is the cost-sensitive greedy comparator.

### 5.3 Yu et al. / SM-DDPO — feasibility assessment, declared in advance

Yu et al. (ICLR 2023) is the closest group-acquisition prior work and is treated seriously.

**A faithful reproduction is not feasible within TwinBench, and we will not fake one.**

| aspect | Yu et al. | TwinBench | verdict |
|---|---|---|---|
| action semantics | prospective panel ordering | retrospective selective disclosure | **not equivalent** |
| state | learned encoder over acquired panels, MIMIC-IV | disclosed-view summary features, PhysioNet 2012 | adaptable in spirit only |
| objective | F1-shaped reward via RL (SM-DDPO) | NLL-vs-budget, no RL training | **not equivalent** |
| cost | time-cost from MIMIC-IV timestamps | declared dimensionless regimes | different basis |

**What can be faithfully adapted:** the *evaluation setting* — sequential group-level actions with
shared per-group cost, and cost-vs-performance curves. That is adopted.
**What cannot:** the SM-DDPO policy itself, its reward shaping, and its learned encoder.
**Closest justified analogue implemented:** `greedy_eig_per_cost`, a cost-normalized myopic
utility policy. **It is labelled a closest analogue, never a reproduction of Yu et al.**

## 6. Budget grid — predeclared, normalized

Costs are dimensionless and differ across regimes, so budgets are expressed as a **fraction of the
total cost of acquiring every catalogue action once under the active regime**:

```
β ∈ {0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00}
```

Covering zero → low → medium → high → ceiling. The same grid is used for every policy comparison
within a regime, and normalization makes the grid comparable across regimes. Chosen before any
policy result was inspected.

## 7. Cost regimes

Existing repository regimes are reused; no new names are invented.

| regime | role |
|---|---|
| `shared_plus_marginal` | **reference regime** (catalogue default) |
| `uniform_group` | every action costs the same |
| `ordinal_tier` | coarse routine / targeted / specialised tiers |
| `per_analyte` | cost proportional to member count |

`uniform_group` and `per_analyte` are the two most materially different and are the primary
ranking-sensitivity contrast.

## 8. Disclosure (masking) conditions

Hidden support is generated by the repaired `group_hours` mechanism at two predeclared rates:
**0.3** and **0.6**.

**M3's `group_structured` and `variable_matched_scattered` are NOT imported as separate
mechanisms.** Repair #3 established they are mask-identical under whole-window semantics; treating
them as distinct conditions here would be invalid.

## 9. Primary metric — AUNLLC

**Primary endpoint: NLL versus budget.**

Integrated scalar, defined before results:

> **AUNLLC** (Area Under the NLL-vs-Budget Curve) — the trapezoidal integral of NLL over the
> predeclared normalized budget grid β ∈ [0, 1], divided by the grid width (1.0), i.e. the
> budget-weighted mean NLL.
>
> `AUNLLC = ∫₀¹ NLL(β) dβ ≈ Σᵢ ½(NLLᵢ + NLLᵢ₊₁)(βᵢ₊₁ − βᵢ)`
>
> **Lower is better.**

The grid is fixed in §6 and the weighting is uniform in β, so no weighting choice can be made after
seeing results.

**Co-primary:** Brier-vs-budget, integrated identically (**AUBSC**, lower is better).

## 10. Secondary outcomes

AUROC-vs-budget and AP-vs-budget (integrated as AUAUROCC / AUAPC, **higher** is better —
orientation stated explicitly); calibration intercept-vs-budget and slope-vs-budget as **direct
diagnostics**, not ranked; mean predicted risk-vs-budget; realized spend; successful disclosures;
failed-request rate.

AUROC is deliberately **not** primary: M3 showed discrimination can stay nearly flat while
probability reliability degrades.

## 11. Ranking-stability statistic — predeclared

**Primary: Kendall tau-b** between policy rankings (ordered by AUNLLC, ascending) across each pair
of evaluation conditions. Chosen because the policy set is small and pairwise inversions are
directly interpretable.

Also reported descriptively: **winner identity** per condition, and the **number of pairwise rank
inversions**.

`full_information_ceiling` and `random_support_oracle` are **excluded from ranking tables** — the
first is not budget-comparable, the second is an oracle.

### 11.1 Ranking-uncertainty rule — binding

A rank reversal is only reported as *supported* when **both** hold:

1. the winner identity changes between conditions, **and**
2. the paired bootstrap CI for the ΔAUNLLC between the two competing policies excludes zero in at
   least one of the two conditions.

Otherwise the reversal is reported as **statistically unresolved**, in those words. A reversal
arising from effectively tied policies swapping by numerical noise will be stated as such.

> **Review #4 terminology correction.** The binding at-least-one-condition flag is retained in the
> artifact as `predeclared_one_condition_evidence`. It is not, by itself, called a statistically
> supported *reversal* when the other relevant paired interval includes zero. The stricter
> `SUPPORTED REVERSAL` label requires both condition-specific paired intervals to exclude zero.

## 12. Statistics

Identical patients, masks and seeds across every paired comparison.

**Bootstrap protocol (binding).** Within each replicate:
1. resample patients with replacement (patient-level);
2. recompute each policy's metric-vs-budget curve **on that resample**;
3. integrate the curve to its AUNLLC;
4. take the paired policy difference.

Already-aggregated budget points are **never** bootstrapped independently. 1,000 replicates,
seed 20260809.

## 13. Model and calibration — frozen

No new model search. Frozen from M2/M3: **XGBoost on `values_mask`**, `max_depth=5,
learning_rate=0.05, min_child_weight=10, n_estimators=200, subsample=0.8, colsample_bytree=0.8,
reg_lambda=1.0`, with the M3 three-way fold split (model-train / calibration / outer test).

A **static Platt calibrator fitted on the clean calibration partition** is applied to every
prediction. Its role is explicit: it is fitted once on clean data and never refitted under any
acquisition state. **No test-condition recalibration.** If acquisition shifts the input
distribution enough that calibration semantics become questionable, that is reported — as it was
in M3 — rather than silently corrected.

Imputation: median, fitted on clean model-train rows only, never refitted.

## 14. Training-boundary rule

Every learned quantity — action frequencies, quantile tables for `greedy_eig`, any surrogate — is
fitted **only on the training portion of the relevant fold**. No outer-test adaptation.
**set-c is never loaded.**

## 15. Compute scope — declared in advance

| scope | patients | conditions |
|---|---|---|
| **Primary condition** — `support_blind` × `shared_plus_marginal` × mask 0.6 | all 8,000 | full budget grid |
| **Ranking-stability grid** — 2 protocols × 4 cost regimes × 2 mask rates = 16 | a seeded 2,000-patient subsample, **identical across all conditions** | full budget grid |

The subsample is declared here, before results, purely for compute tractability. All paired
comparisons within the stability grid use those same patients, so pairing is preserved.

## 16. Artifact contract

Every run retains: git SHA and dirty flag; cohort fingerprint; split hash; policy-training split
hash; mask seed and mechanism id; action-catalogue config hash; cost regime; support protocol;
budget grid; policy id and config; patient ids; actions selected; success/failure per action;
realized costs; disclosed counts; predictions at every budget; labels; metrics; integrated
summaries; paired differences; ranking table; and the ranking-stability statistic.

**Action traces** are retained for the final policies: step, policy-visible state hash, action,
cost, success, remaining budget, newly disclosed component count, and model prediction before and
after. Traces record **counts and hashes, never hidden values.**

## 17. Claim rules — binding

- No prospective-ordering, clinical-recommendation or intervention language.
- `support_aware` results are never presented as deployable.
- `fixed_domain_order` is domain-motivated, never clinician-derived.
- `greedy_eig_per_cost` is a closest analogue, never a reproduction of Yu et al.
- The expected contribution is a **new evaluation of policy-ranking stability**, not a new policy.
  Before any novelty claim, primary literature is searched around the exact claim.
- No causal or deployment-utility claims.

## 18. Predeclared outcomes

- **A (very strong):** rankings materially change across support/cost/masking assumptions, and the
  reversals satisfy §11.1.
- **B (moderate):** the top policy is stable, but margins, calibration behaviour or cost-efficiency
  change substantially.
- **C (null):** rankings stable across assumptions. Reported as-is; Cliniverse becomes a robustness
  certification.

**Instability will not be manufactured.** If the evidence supports C, C is what is reported.
