# TwinBench — Formal Specification

**Version:** 0.3 (repair #1; supersedes version 0.2)
**Date:** 2026-08-09

This document fixes the estimand, the information boundary, and the policy/evaluator information
split **before** implementation, so that the protocol cannot be quietly reshaped to suit results.

---

## 1. What this benchmark is, and is not

TwinBench is **sequential retrospective selective disclosure of values in panel-like feature
groups under a budget.**

It is **not** prospective test ordering. We do not use that phrase. A policy here cannot cause a
test to be performed that was never performed. An action reveals all synthetically hidden,
historically recorded values in the chosen feature group at hourly bins before the current
boundary. It does not reconstruct an order, specimen, or co-measurement event.

This restriction is forced. Estimating what would have happened had a test genuinely been ordered
requires the No-Unobserved-Confounding assumption of the AFAPE literature, and NUC is false in ICU
data: tests are ordered because of clinician judgement that the record does not contain.

---

## 2. Timeline, epochs and the information boundary

For a patient with horizon `H = 48` hours:

- **Decision epochs** `k = 1 … K` at boundaries `t_1 < t_2 < … < t_K`.
  Default: `K = 3` at `t = 12, 18, 24` hours.
- **Prediction time** `T = t_K = 24` hours.
- **Information boundary at epoch k** is `t_k`. Any quantity with timestamp `> t_k` is outside the
  boundary at that epoch.

**Boundary rule (invariant BR-1).** At epoch `k`, no value with timestamp `> t_k` may enter the
policy's view, the model's input, or any feature derived from either.

**Target rule (invariant BR-2).** Every prediction target has support strictly `> T`, or is an
end-of-episode outcome unavailable at any epoch.

`Cohort.truncate(hours)` is the single enforcement point for BR-1, and
`tests/test_leakage.py` asserts the property generally rather than per-variable.

---

## 3. Estimand

Fix a patient population `P`, a disclosure protocol `Π`, a cost regime `C`, a masking mechanism
`m` and seed `s`.

For an acquisition policy `π` and budget `B`:

```
V(π, B; Π, C, m, s) = E_{patient ~ P} [ L( ŷ_π(patient, B), y(patient) ) ]
```

where `ŷ_π` is produced by a **fixed** predictive model (identical across all policies) applied to
whatever `π` disclosed within budget `B`, and `L` is the task metric.

**Headline scalar — AUBC.** Area under the budget–performance curve over a fixed budget grid
`B ∈ {0, 1, 2, 3, 4, 5, 8, 10}` cost units, normalised by grid width.

**Primary comparison (pre-registered).**

```
Δ AUBC(π) = AUBC(π ; Π = support_blind) − AUBC(π ; Π = support_aware)
```

evaluated on **identical patients, identical masks, identical seeds, identical epochs, identical
predictive model**. Reported as a **paired patient-level bootstrap CI** (≥1,000 resamples over
patients). Standalone non-overlapping CIs are not used for policy comparison.

**Secondary (pre-registered).** Rank-reversal count for the ordering of policies by AUBC across the
four cost regimes of §6.

**Scope limit — binding.** `V` is a property of *this protocol*. It is not an estimate of clinical
utility or deployment value, and no document, figure or API response may present it as one.

---

## 4. Policy-visible vs evaluator-only information

The benchmark's validity rests on this split. Availability is informative (E-002: availability
alone gives AUROC 0.7224), so any leak of the support turns the benchmark into a test of how well a
policy reads clinician behaviour.

### 4.1 Policy-visible (`PolicyView`)

| Field | Description |
|---|---|
| `disclosed_values` | Values disclosed so far, `NaN` elsewhere |
| `disclosed_mask` | `True` where a value has been disclosed |
| `epoch`, `n_epochs` | Current and total epoch index |
| `boundary_hour` | `t_k` for the current epoch |
| `spent`, `remaining` | Budget consumed and left |
| `catalogue` | Detached immutable action names, members, and costs — identical for every patient |
| `statics` | Admission descriptors within the boundary |

### 4.2 Evaluator-only (never reachable from `PolicyView`)

- `S_hidden` — the synthetically hidden set
- The historical support (which cells were ever recorded)
- Counts or existence of hidden values, globally or per panel
- Whether a given gap is natural or synthetic
- Future availability, timestamps beyond `t_k`, and all targets

**Invariant SO-1.** `PolicyView` holds no reference to evaluator state. Arrays are read-only
copies; its action catalogue is a separate immutable policy type, not the evaluator catalogue.

**Invariant SO-2 (indistinguishability).** For any policy, a panel with hidden values and a panel
with none must be indistinguishable *prior to purchase*.

### 4.3 Unavailable requests

Under `support_blind`, a policy may request **any** feature group at **any** epoch.

| Situation | Cost charged | Disclosed |
|---|---|---|
| Group has hidden observed values within the boundary | full group cost | all such values across earlier hourly bins |
| Group has none (never recorded, or only outside boundary) | **full group cost** | nothing |
| Group already fully disclosed | full group cost | nothing new |

Charging full price for an empty result is deliberate and load-bearing. Free failed requests would
let a policy probe availability at no cost, reconstructing the historical ordering pattern — the
exact leak SO-2 forbids. After paying, `Purchase.n_disclosed == 0` and the unchanged next view
legitimately reveal that this action produced nothing *within that boundary*. That acquired
absence information is part of the estimand; it is never free.

---

## 5. Disclosure protocols

| Protocol | Requestable set | Purpose |
|---|---|---|
| **support_aware** | Only groups with at least one hidden value within the boundary | Availability oracle that diagnoses standard replay practice; **not deployable or a fair standalone comparator** |
| **support_blind** | The entire catalogue | Availability is no longer free; a wasted request costs full budget |

The paired protocol difference diagnoses sensitivity to free historical support. It must not be
described as a comparison between two deployable policies.

---

## 6. Cost regimes

Costs are **dimensionless relative units**. PhysioNet 2012 contains no prices; no monetary figure
is used or implied.

| Regime | Definition |
|---|---|
| `uniform_group` | Every feature-group action costs 1.0 — isolates grouping from pricing |
| `shared_plus_marginal` | `cost = 1.0 + 0.1 × (group members)` — a shared action cost plus a marginal per-member cost |
| `ordinal_tier` | Routine = 1, targeted = 2, specialised = 3, by development-data presence frequency |
| `per_analyte` | `cost = number of analytes` — the implicit assumption of feature-level AFA; included as the comparison that tests whether grouping matters |

A conclusion holding only under one regime is a conclusion about that regime, and is reported as
such.

Repeated actions pay the full declared cost and disclose only values not already disclosed.
Groups partition the laboratory features, so overlap has no incremental-cost ambiguity. Budgets
and every action cost must be finite; costs are strictly positive.

### 6.1 Random baseline contract

- `random_uniform_all`: support-blind, uniform over all affordable legal action types.
- `random_train_frequency`: support-blind, weighted by group presence fitted on training
  patient-hours only; no evaluation-patient support enters the weights.
- `random_support_oracle`: support-aware diagnostic oracle over available groups only. Never
  label this deployable, fair, or simply `random`.

### 6.2 Masking mechanism contract

| Name | Mechanism | Represents | Does not represent |
|---|---|---|---|
| `mcar_cells` | Independently samples observed analyte-hour cells | Synthetic cellwise ablation | Clinical ordering or realistic missingness |
| `group_hours` | Samples active hourly bins per feature group and hides observed members in each selected bin | Synthetic preservation of binned within-group co-presence | Orders, specimens, assays, prospective events, or causal interventions |
| `time_blocks` | Samples contiguous hour blocks across variables | Synthetic monitoring gaps | A clinical decision or patient-state process |

All mechanisms are outcome-free, require only the cutoff-truncated observation mask, and are
deterministic in `(mechanism parameters, seed, patient index)`. Paired protocol comparisons reuse
the identical case and mask.

---

## 7. Tasks

| Task | Type | Definition | Status |
|---|---|---|---|
| **T1** | Binary | In-hospital mortality, predicted from data ≤ 24h | **Primary** |
| **T3** | Scalar regression | Mean creatinine over hours (24, 48], for patients with ≥1 creatinine in that window, predicted from data ≤ 24h | **Secondary** |
| ~~T2~~ | — | LOS > 3 days | **Dropped** — arbitrary threshold, and confounded by the discharge/death process |

T3 is restricted to patients with an observed target. This is a **selection bias** — those patients
had a reason to be measured — and it is reported as a limitation, not silently absorbed.

---

## 8. Metrics

**T1.** AUROC, AUPRC, Brier, log-loss, **reliability curve, calibration slope, calibration
intercept**, risk–coverage curve and AURC for selective prediction.
Brier and log-loss are described as **proper scoring rules** (calibration *and* refinement), never
as calibration metrics on their own.

**T3.** MAE, RMSE, **split-conformal** interval coverage and mean width at 90% nominal, reported
**stratified by missingness pattern and budget**. No CP-MDA: it is a regression method for missing
covariates and we have not justified transferring it to classification.

**Acquisition.** Budget–performance curves, AUBC, paired ΔAUBC, cost-normalised utility,
rank-reversal counts across cost regimes.

---

## 9. Reproducibility contract

Manifest version 2 records dataset and sets, record ID and patient index, cutoff and epochs,
mechanism ID/rate/seed, protocol, cost regime, finite budget, catalogue version/content hash,
schema/config hash, cutoff-safe dataset fingerprint, and Git SHA. Every field contributes to the
content hash; regeneration from identical state reproduces it, and tampering is rejected.

**set-c** is **quarantined from model fitting and model selection following an aggregate cohort
audit** (n=4,000, 585 deaths, 14.62%, read once during dataset assessment). It is unlocked exactly
once, at the end. `load_cohort()` defaults to sets a+b and requires the explicit
`allow_final_holdout=True` flag even to materialise set-c; `final_holdout()` separately requires
its unlock token. That use must be logged in `EXPERIMENTS.md`.
