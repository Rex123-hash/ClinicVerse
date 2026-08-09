# Response to Independent Scientific Review #0

**Date:** 2026-08-09
**Reviewer:** the reviewer (independent)
**Status:** All P0 findings resolved or formally specified. Recommendation at the end: **MODIFY**.

Every finding below was independently verified against the repository, the actual data, the
PhysioNet documentation, or primary literature before being classified. Where the reviewer is right I
say so plainly; where it is partly right I separate the correct part from the overreach.

---

## Summary table

| # | Finding | Classification | Resolved |
|---|---|---|---|
| 1 | Yu et al. (ICLR 2023) already does panel-level acquisition — our novelty claim is invalid | **ACCEPT** | Yes — claim retracted, D-001 superseded |
| 2 | Weight leakage: longitudinal Weight treated as static | **ACCEPT** | Yes — fixed, tested, statistics recomputed |
| 3 | Dataset has 37 time-series variables, not 36 | **ACCEPT** | Yes |
| 4 | Raw-row occupancy figures include sentinels/artifacts | **ACCEPT** | Yes |
| 5 | Acquisition action at t=24h is temporally undefined | **ACCEPT** | Yes — formal spec written |
| 6 | Policy must never see the support oracle | **ACCEPT** | Yes — specified, tests planned |
| 7 | "set-c never touched" is inaccurate | **PARTIALLY ACCEPT** | Yes — wording corrected |
| 8 | T2 (LOS>3d) arbitrary; T3 underspecified | **PARTIALLY ACCEPT** | Yes |
| 9 | Panel names overstate what P12 records | **ACCEPT** | Yes — renamed |
| 10 | Cost model should avoid fake money | **PARTIALLY ACCEPT** | Yes |
| 11 | Mask-only baseline is mandatory | **ACCEPT** | Yes — **already executed, E-002** |
| 12 | Non-overlapping CIs are the wrong test | **ACCEPT** | Yes |
| 13 | Brier/NLL are not pure calibration metrics | **ACCEPT** | Yes |
| 14 | CP-MDA misapplied to classification | **ACCEPT** | Yes |
| 15 | "clinician-inspired" is unearned wording | **ACCEPT** | Yes |

---

## 1. Yu et al. (ICLR 2023) invalidates the panel-level novelty claim — **ACCEPT**

**Verification.** I retrieved arXiv:2302.10261 — *Deep Reinforcement Learning for Cost-Effective
Medical Diagnosis*, Yu, Li, Kim, Huang, Luo, Wang, ICLR 2023. Confirmed from primary sources:

- The method (SM-DDPO) "uses reinforcement learning to find a dynamic policy that **selects lab
  test panels sequentially** based on previous observations."
- "The framework handles a complex action space related to **panels of medical tests (groups of
  tests acquired together)**."
- "The research compared fixed test selection strategies, including testing with the two most
  relevant panels: **CBC and CMP**. **Each group of tests is assigned a time-cost by observing
  time-stamps in the MIMIC-IV database.**"
- Clinical endpoints: ferritin abnormality, **sepsis mortality**, acute kidney injury.
- Cost-vs-performance: Pareto-front solutions; AKI testing cost reduced from **$591 to $90**;
  up to 85% cost reduction.

This is sequential, panel-level, shared-cost acquisition on clinical outcomes with cost-performance
curves. **the reviewer is factually correct. Our claimed primary contribution does not exist.**

**Action taken.**
- D-001 is **superseded by D-008**.
- Deleted from all documents: "panel-structured acquisition — strongest remaining contribution",
  "every AFA method surveyed acquires individual features at individual cost", and any use of
  *first*, *novel panel acquisition*, or *clinically costed panels*.
- Yu et al. added as **closest prior art**.
- **Should SM-DDPO be a mandatory baseline?** *Partially.* A faithful reproduction is out of scope:
  it is an RL method tuned on MIMIC-IV, and reproducing it on P12 within this timeframe would
  produce a weak strawman that would be worse than not running it. Instead we adopt its
  **evaluation setup** — sequential panel actions, shared group cost, cost-vs-performance
  curves — as the standard our benchmark must match, and we run a **greedy panel-EIG policy and a
  learned discriminative panel policy** as the strong method-side comparators. This is recorded as
  a stated limitation, not glossed over.

---

## 2. Weight leakage — **ACCEPT** (confirmed, severe)

**Verification.** Counted directly in `set-a`:

| Metric | Value |
|---|---|
| Total `Weight` rows | 129,165 |
| Rows at hour 0 | 5,347 (**4.1%**) |
| Rows after hour 0 | 123,818 (**95.9%**) |
| Rows at hour ≥ 24 | 67,338 (**52.1%**) |
| Patients with a Weight at hour ≥ 24 | **2,726 of 4,000 (68%)** |

The parser routed **all** of these into the static vector, because it tested
`if param in config.statics` *before* considering the timestamp. A model with a 24h cutoff
therefore received weights measured as late as hour 47. **Confirmed and severe.**

**Broader audit (the reviewer asked for one).** I audited every static descriptor. `Age`, `Gender`,
`Height`, `ICUType` and `RecordID` each appear **exactly 4,000 times in set-a, all at hour 0**.
**Weight was the only offender.**

**Fix.**
1. `Weight` is now a **time-series variable** (37th).
2. A new static `AdmissionWeight` is sourced from `Weight` **at hour 0 only**, declared in the
   schema as `source_parameter: Weight, at_hour: 0`.
3. The parser now resolves **time before routing**: a static field can only be populated at its
   pinned hour; the same parameter at any later hour becomes a time-series observation.
4. A second bug surfaced while testing the fix: `AdmissionWeight` disagreed with the hour-0 grid
   cell for **65/3,701 patients (1.76%)**, because statics took the last hour-0 row while binning
   takes the within-hour mean. Statics sourced from a time-series parameter are now **derived from
   the binned grid cell**, so the two views agree by construction.
5. `tests/test_leakage.py` added — written as a **general** post-cutoff property, not a
   Weight-specific check, so any future variable of the same shape is caught.

---

## 3. 37 variables, not 36 — **ACCEPT**

Direct consequence of finding 2. Our 36 excluded `Weight`. Both the parser *and*
`scripts/verify_physionet2012.py` shared the error — the verification script could not have
caught it because it encoded the same wrong assumption. Both corrected.

---

## 4. Raw-row occupancy is not the binned missingness statistic — **ACCEPT**

the reviewer is right, and the cleanest demonstration is the degenerate-record discrepancy.

Records `140501`, `140936`, `141264` contain **only** `Weight,-1` at hour 0. The raw row-count
script now sees a row and calls them non-empty; the production parser drops `-1` as an
implausible/sentinel value and correctly reports them as having **zero** valid observations.

Corrected statistics (production parser, set-a):

| Statistic | Old (wrong) | Corrected |
|---|---|---|
| Time-series variables | 36 | **37** |
| Raw row-count occupancy **upper bound** | 23.28% | **24.46%** |
| **Binned grid occupancy (the real figure)** | 19.35% | **20.25%** |
| **Binned missingness (the real figure)** | 80.65% | **79.75%** |
| Degenerate records | 3 | 3 (unchanged, now for a documented reason) |

**Wording rule adopted:** the raw figure is described only as *a loose upper bound that counts
sentinel rows and within-hour collisions*; the reported missingness statistic is always
`Cohort.describe()['grid_occupancy']` from the production parser.

---

## 5. Temporal semantics of the acquisition action — **ACCEPT** (P0)

the reviewer is right that the action was undefined. "At t=24h, buy a BMP" cannot mean *reveal a past
observation as if newly ordered* (incoherent), nor *reveal a future observation* (leakage).

**Resolution: Option A, stated honestly.** The benchmark is **sequential selective disclosure
(replay) of historically recorded panel-like events**, not prospective test ordering. Formal
specification in `docs/BENCHMARK_SPEC.md`. Summary of the estimand:

- Timeline is divided into **decision epochs**. The information boundary at epoch *k* is
  `t_k`; targets lie strictly beyond the final boundary.
- At epoch *k* a policy may request a panel. Disclosure reveals **only** values with
  `timestamp ≤ t_k` that are in the hidden set. Nothing after the boundary is ever disclosed.
- Therefore the action is *"disclose what was recorded for this panel up to now"*, which is
  well-defined, leakage-free, and honest about being replay.

**We will not call this "temporal test ordering."** It is disclosure replay under a budget. The
limitation — that it cannot tell us what would have happened had a test genuinely been ordered
that was not — is stated in `LIMITATIONS.md` and follows directly from the AFAPE result.

---

## 6. Support oracle must be invisible to the policy — **ACCEPT** (P0)

Correct, and this is the single most important implementation invariant, because the whole point
of E-002 (below) is that availability is informative. If the policy can see which panels have
hidden values waiting, it reads the clinician's decisions directly and every number is void.

**Specification** (`docs/BENCHMARK_SPEC.md` §4). The policy observes only: disclosed values, the
disclosed mask, elapsed budget, remaining budget, epoch index, and the static panel catalogue.
The policy **never** observes `S_hidden`, the historical support, counts of hidden values, whether
a gap is natural or synthetic, or future availability.

**Unavailable requests.** A policy may request any panel at any epoch. If nothing is available,
**the cost is charged in full and nothing is disclosed.** This is what makes synthetic-hidden and
naturally-missing indistinguishable from the policy's side. Charging nothing would itself leak
availability.

Enforced structurally by handing the policy a `PolicyView` object that has no reference to the
evaluator state, plus tests asserting non-inferability.

---

## 7. "set-c never touched" — **PARTIALLY ACCEPT**

**Accept:** the wording was inaccurate. I read aggregate set-c outcome statistics (n=4,000,
585 deaths, 14.62%) during dataset assessment, and reported them.

**Reject the implication of contamination:** aggregate prevalence was read for feasibility
assessment; no set-c record has entered any fit, any hyperparameter choice, or any model
selection, and `final_holdout()` requires an explicit unlock token.

**Corrected wording, adopted everywhere:** *"set-c is quarantined from model fitting and model
selection following an aggregate cohort audit."*

---

## 8. Task audit — **PARTIALLY ACCEPT**

- **T1 (in-hospital mortality) — KEEP as primary.** Agreed.
- **T2 (LOS > 3 days) — ACCEPT, DROPPED.** The 3-day cut was arbitrary, and length-of-stay is
  contaminated by discharge/death processes (a patient who dies early has a short stay). It added
  a metric, not a question. Removed rather than demoted.
- **T3 — ACCEPT, now precisely specified.** Reduced to **one** leakage-safe scalar:
  *the mean creatinine over hours (24, 48], for patients with at least one creatinine
  measurement in that window*, predicted from data up to hour 24. Rationale: creatinine has 98.4%
  cohort coverage, ~3.5 draws per stay, is clinically interpretable, and belongs to a panel that
  is itself acquirable — so the acquisition and forecasting problems interact meaningfully.
  Restricting to patients with an observed target is stated as a selection-bias limitation.

---

## 9. Panel naming overstates the data — **ACCEPT**

P12 records analytes, not laboratory orders. Our clusters are co-measurement groupings, not
recovered order records. Renamed throughout:

`BMP → BMP-like`, `CBC → CBC-like`, `ABG → ABG-like`, `LFT → hepatic-like`.

There was no coagulation panel to remove — P12 contains no PT/PTT/INR, and we never defined one.
The empirical derivation (E-001) stands and is unaffected by the renaming; what changes is the
claim attached to it. We now say: *these analytes are co-measured in this dataset at Jaccard
0.78–0.97, consistent with panel-like ordering*, not *these are laboratory panels*.

---

## 10. Cost model — **PARTIALLY ACCEPT**

**Accept:** no fake monetary values. The `$` framing is removed; costs are dimensionless relative
units. Adopted the three regimes the reviewer asked for: **uniform per-event**, **shared event cost +
marginal analyte cost**, and **ordinal routine/targeted/specialised tiers** — replacing our
previous ad-hoc numbers.

**Partially:** I keep a `per_analyte` regime as well, because it is precisely the implicit
assumption of feature-level AFA and is the comparison that tests whether grouping matters.

**Agreed and elevated:** policy rankings will be tested for sensitivity across all regimes, and
**if rankings change materially under plausible regimes, that is a headline result, not an
appendix**.

---

## 11. Mask-only baseline — **ACCEPT** (executed, and it changed the project)

the reviewer called this "extremely important." It was. **Already run — see `EXPERIMENTS.md` E-002.**

In-hospital mortality at a 24h cutoff, 5-fold CV, n=8,000 (sets a+b), prevalence 14.03%:

| Feature view | Model | AUROC | 95% CI |
|---|---|---|---|
| **availability only** (no measured value at all) | logreg | **0.7224** | [0.707, 0.738] |
| **availability only** | gbdt | 0.7187 | [0.703, 0.733] |
| values only | gbdt | 0.8184 | [0.807, 0.831] |
| availability + statics | gbdt | 0.7510 | [0.736, 0.765] |
| all | gbdt | **0.8363** | [0.825, 0.847] |

**A model that cannot see a single laboratory value reaches AUROC 0.72 — recovering 65–70% of the
full model's discrimination above chance, purely from which tests were ordered, how often, and how
recently.**

This is the empirical foundation of the revised thesis (D-008) and it directly justifies the
support-blind protocol.

---

## 12. Statistics — **ACCEPT**

Non-overlapping standalone CIs is the wrong test and would have inflated our confidence.
Adopted: **paired patient-level inference**. The headline comparison is a **paired bootstrap CI on
ΔAUBC** with identical patients, masks, seeds and epochs across policies. The primary comparison
is **pre-registered in `docs/BENCHMARK_SPEC.md` before final results are computed.**

---

## 13. Brier and NLL are proper scoring rules, not pure calibration — **ACCEPT**

Correct; our wording conflated them. They decompose into calibration *and* refinement, so a better
Brier does not by itself demonstrate better calibration. Corrected wording, and added the metrics
the reviewer asked for: **reliability curves, calibration slope, and calibration intercept** (the latter
two being the direct calibration readouts).

---

## 14. Conformal methodology — **ACCEPT**

Applying CP-MDA to classification because it handles missingness in regression was unjustified.
Corrected scope:

- **T1 (classification):** calibrated probabilities, reliability curve, calibration slope and
  intercept, plus **risk–coverage / selective prediction**. No CP-MDA.
- **T3 (scalar regression):** **split conformal** intervals for the single defined target, with
  coverage reported **stratified by missingness pattern and budget** — which is the honest way to
  surface the Zaffran et al. conditional-coverage failure without overclaiming a method we did not
  implement.

Stronger conformal claims only if mathematically justified, which currently they are not.

---

## 15. "Clinician-inspired" heuristic — **ACCEPT**

No clinician designed or validated our ordering. Renamed to **"fixed domain-motivated ordering"**
and documented as *authored by the engineering team from routine-ordering frequency in the data*.
Any oracle policy is labelled an **unattainable diagnostic upper bound**.

---

## Recommendation: **MODIFY**

Not GO (the original contribution is void), not PIVOT (the dataset, pipeline, and now the E-002
result all remain valid and load-bearing). The scientific question changes; the engineering does
not need to be rebuilt.

See `docs/NOVELTY_REASSESSMENT.md` for the revised contribution, the claims we must not make, and
the Best Overall positioning.
