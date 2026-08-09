# TwinBench — Formal Specification

**Version:** 0.2 (supersedes the informal description in `research_assessment.md` §13.3)
**Date:** 2026-08-09

This document fixes the estimand, the information boundary, and the policy/evaluator information
split **before** implementation, so that the protocol cannot be quietly reshaped to suit results.

---

## 1. What this benchmark is, and is not

TwinBench is **sequential selective disclosure (replay) of historically recorded panel-like
events under a budget.**

It is **not** prospective test ordering. We do not use that phrase. A policy here cannot cause a
test to be performed that was never performed; it can only cause an already-recorded value to be
disclosed, or spend budget discovering that nothing is available.

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
| `catalogue` | Panel names, members, costs — identical for every patient |
| `statics` | Admission descriptors within the boundary |

### 4.2 Evaluator-only (never reachable from `PolicyView`)

- `S_hidden` — the synthetically hidden set
- The historical support (which cells were ever recorded)
- Counts or existence of hidden values, globally or per panel
- Whether a given gap is natural or synthetic
- Future availability, timestamps beyond `t_k`, and all targets

**Invariant SO-1.** `PolicyView` holds no reference to evaluator state. It is constructed by copy,
and policies receive nothing else.

**Invariant SO-2 (indistinguishability).** For any policy, a panel with hidden values and a panel
with none must be indistinguishable *prior to purchase*.

### 4.3 Unavailable requests

A policy may request **any** panel in the catalogue at **any** epoch.

| Situation | Cost charged | Disclosed |
|---|---|---|
| Panel has values within the boundary in `S_hidden` | full panel cost | those values |
| Panel has none (never recorded, or only outside boundary) | **full panel cost** | nothing |
| Panel already fully disclosed | full panel cost | nothing new |

Charging full price for an empty result is deliberate and load-bearing. Free failed requests would
let a policy probe availability at no cost, reconstructing the historical ordering pattern — the
exact leak SO-2 forbids.

---

## 5. Disclosure protocols

| Protocol | Requestable set | Purpose |
|---|---|---|
| **support_aware** | Only panels with at least one hidden value within the boundary | Reproduces standard AFA replay practice, where the acquirable set is derived from what was historically recorded — so availability is a free signal |
| **support_blind** | The entire catalogue | Availability is no longer free; a wasted request costs full budget |

The paired difference between these is the primary result.

---

## 6. Cost regimes

Costs are **dimensionless relative units**. PhysioNet 2012 contains no prices; no monetary figure
is used or implied.

| Regime | Definition |
|---|---|
| `uniform_event` | Every panel event costs 1.0 — isolates grouping from pricing |
| `shared_plus_marginal` | `cost = 1.0 + 0.1 × (analytes in panel)` — a shared draw cost plus a marginal per-analyte cost |
| `ordinal_tier` | Routine = 1, targeted = 2, specialised = 3, by observed ordering frequency |
| `per_analyte` | `cost = number of analytes` — the implicit assumption of feature-level AFA; included as the comparison that tests whether grouping matters |

A conclusion holding only under one regime is a conclusion about that regime, and is reported as
such.

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

Every case carries `(mechanism_id, seed, protocol, cost_regime, git_sha, config_hash)`.
Generated case manifests ship as **content hashes**, not blobs. Regeneration from a seed must
reproduce identical hashes; this is asserted in CI.

**set-c** is **quarantined from model fitting and model selection following an aggregate cohort
audit** (n=4,000, 585 deaths, 14.62%, read once during dataset assessment). It is unlocked exactly
once, at the end, via `final_holdout()` with an explicit token, and that use is logged in
`EXPERIMENTS.md`.
