# Cliniverse — Architecture Decisions

Each entry records the decision, the alternatives rejected, and why. Decisions are amendable;
supersede rather than silently edit.

---

## D-001 — Reframe the project's claimed contribution
**Status:** **SUPERSEDED BY D-008** (2026-08-09, after independent review #0)
**Date:** 2026-08-09

> **Why superseded.** D-001 claimed panel-level, shared-cost acquisition as our contribution.
> Yu et al., ICLR 2023 (arXiv:2302.10261) already does exactly this — sequential panel selection
> (CBC, CMP) with shared group costs on MIMIC-IV, clinical endpoints, and Pareto cost-performance
> curves. The claim is void. See `docs/REVIEW_RESPONSE_0.md` §1 and D-008 below.

**Context.** The brief names "active information acquisition applied to a longitudinal patient
world model" as the defining innovation.

**Finding.** This is established, published work. Deep Sensing (ICLR 2018) and ASAC (MLHC 2019)
do active sensing on clinical time series; EDDI (ICML 2019) does EIG-based acquisition motivated
by diagnostic test ordering; Clairvoyance (ICLR 2021) ships a medical time-series pipeline with
an explicit information-acquisition pathway; NOCTA (2025) does longitudinal cost-aware
acquisition; AFABench (2025) already benchmarks AFA methods *including on PhysioNet 2012*;
ETHOS (npj Digital Medicine 2024) generates multiple plausible future patient timelines.

**Decision.** Reframe to: *a reproducible benchmark and evaluation harness for cost-aware,
**panel-level**, time-aware observation acquisition, with calibration as a first-class metric.*
Contribution rests on (a) panel-structured acquisition with shared cost — absent from surveyed
AFA work, which acquires individual features at individual cost; (b) a temporal clinical AFA
benchmark on credentialing-free data; (c) calibration-vs-budget rather than accuracy-vs-budget;
(d) budget-coupled abstention.

**Alternatives rejected.**
- *Keep the original framing.* Rejected: trivially falsified by a literature search, and the
  brief forbids indefensible claims.
- *Pivot to a genuinely novel method (e.g. non-myopic RL acquisition).* Rejected: A2MT reports
  SOTA agents failed to learn adaptive acquisition; AFABench finds non-myopic gains often do not
  justify their cost. Too high-variance for the timeframe.

---

## D-002 — PhysioNet/CinC Challenge 2012 as the primary dataset
**Status:** ACCEPTED
**Date:** 2026-08-09

**Decision.** Use P12 (all 12,000 labeled patients) as primary. Synthea as a secondary
controlled generator for structured-missingness and OOD cases only. MIMIC-IV credentialing
started in parallel but kept off the critical path.

**Why.** Verified open access (ODC-BY, no credentialing, ~20 MB); natively irregular, with
**79.75% missingness on the binned hourly grid** (production parser, set-a — see D-007 for the
corrected figures); and its coverage structure separates a near-continuous monitoring tier from a
discretely measured laboratory tier (~3.5 measurements per 48h stay), which is what makes an
acquisition problem well-posed. Established literature baselines exist for the mortality task.

**Alternatives rejected.**
- *MIMIC-IV full.* Credentialing is multi-day human review; cannot gate the project on it.
- *Synthea as primary.* Its data comes from hand-authored clinician-designed state machines;
  information-gain results would be artifacts of the generator's rules.
- *MIMIC-IV demo.* 100 patients — unusable for training or evaluation.

---

## D-003 — Masking-on-observed-support evaluation protocol
**Status:** **AMENDED BY D-009** (2026-08-09)
**Date:** 2026-08-09

> **Amendment.** The rule "naturally-missing cells are permanently unavailable and never
> acquirable" is superseded. Under D-009 a policy may *request* any panel; an unavailable request
> costs full price and discloses nothing. Forbidding the request would itself tell the policy which
> panels hold hidden values — the exact support leak that invalidates the benchmark. The rest of
> D-003 (synthetic, seeded, disclosed mechanism; ground truth by construction) stands.

**Context.** AFAPE (von Kleist et al., JMLR 26) shows estimating deployed AFA performance from
retrospective data requires No-Direct-Effect and No-Unobserved-Confounding. **NUC is false in
ICU data**: a lactate exists in the record because the patient already looked septic, using
information absent from the dataset. Missingness is NMAR and informative.

**Decision.** Do not estimate deployment value. Evaluate policies under a **synthetic masking
mechanism we specify and control**: hide a seeded subset of *observed* cells; policies buy them
back. Ground truth exists by construction, the mechanism is known by construction, so
paired cases isolate implemented random variation with respect to that mechanism. This does not
establish statistical or causal unbiasedness. Naturally-missing cells are
permanently unavailable and never acquirable.

**Cost.** Claims are limited to relative policy performance under a disclosed mechanism. This
limitation ships in the README, the API payloads, and `LIMITATIONS.md`.

**Alternatives rejected.**
- *Naive retrospective evaluation.* Produces impressive, biased, meaningless numbers.
- *Full AFAPE semi-offline RL estimators (DM/IPW/DRL).* Correct but requires assumptions we
  cannot defend on this data, plus substantial machinery. Revisit only if time permits.

---

## D-004 — Baselines gate sophistication
**Status:** ACCEPTED
**Date:** 2026-08-09

**Decision.** Strict tiering: prevalence → SAPS-I/SOFA → logistic regression → GBDT → GRU-D →
transformer. Each tier must beat the one below on validation, with bootstrap CIs, before the
next is built.

**Why.** SAPS-I and SOFA ship in the outcomes files and are real clinical severity scores; a
model that cannot beat SOFA is not interesting. GBDT on summary features is expected to be the
strongest baseline, and a deep model losing to it is the likely outcome, not a bug.

**Guard.** SAPS-I/SOFA are legitimate *baselines* but would be *leakage* as features — they are
computed from the same 48h window. Enforced by a unit test.

---

## D-005 — Cut multimodality and MedGemma
**Status:** ACCEPTED
**Date:** 2026-08-09

**Decision.** Ship one modality: structured irregular clinical time series. Remove notes,
imaging, wearables, diagnoses/medications, MedGemma, BigQuery, Healthcare API, Firestore.

**Why.** P12 contains none of those modalities — implementing them would be architecture
theater. MedGemma is a medical *vision-language* model; our data is numeric time series, so it
cannot contribute to the tasks. Separately, Vertex AI endpoints bill from provision to undeploy
including idle (A100 ≈ $2.93/GPU-hr ≈ $2,642/month; L4 ≈ $0.70/GPU-hr), which disqualifies a
persistent endpoint against $300 of credit.

**Alternatives rejected.** *Include MedGemma for narrative explanation.* Rejected: the brief
forbids explanations generated purely by an LLM, and it would be logo-driven architecture.

---

## D-006 — Pin to Python 3.12
**Status:** ACCEPTED
**Date:** 2026-08-09

**Decision.** Pin the project interpreter to Python 3.12 via `uv`, despite the machine's system
Python being 3.14.4.

**Why.** Verified on PyPI: torch has no Windows cp314 wheel yet. xgboost 3.4.0 and lightgbm
4.7.0 ship `py3-none-win_amd64` and would be fine either way, so torch is the binding
constraint. 3.12 has universal wheel coverage for the whole stack.

---

## D-007 — Weight is a time-series variable; statics are pinned to hour 0
**Status:** ACCEPTED
**Date:** 2026-08-09

**Context.** independent review #0 flagged that `Weight` was treated as a static descriptor despite
being recorded longitudinally. Verified in set-a: of 129,165 `Weight` rows, only 4.1% are at
hour 0; 95.9% come later and 52.1% are at hour >= 24, affecting 2,726 of 4,000 patients. The
parser tested `param in statics` before looking at the timestamp, so a model with a 24h cutoff
received weights measured as late as hour 47.

**Broader audit.** `Age`, `Gender`, `Height`, `ICUType` and `RecordID` each appear exactly 4,000
times in set-a, all at hour 0. Weight was the only offender.

**Decision.**
1. `Weight` becomes a time-series variable, giving **37**, not 36.
2. A new static `AdmissionWeight` is declared with `source_parameter: Weight, at_hour: 0`.
3. The parser resolves **time before routing**: a static may be populated only at its pinned hour.
4. Statics sourced from a parameter that is also a time series are **derived from the binned grid
   cell**, because parsing them separately made the two views disagree for 1.76% of patients
   (statics took the last hour-0 row; binning takes the within-hour mean).
5. `tests/test_leakage.py` asserts the general property "no model input may depend on post-cutoff
   data", so any future variable of this shape is caught automatically.

**Consequences.** Corrected statistics: 37 variables; binned occupancy 20.25% (missingness
79.75%); raw row-count upper bound 24.46%. The raw bound counts sentinel rows and within-hour
collisions and is never reported as the missingness statistic.

**Alternatives rejected.** *Keep Weight static but take its first value.* Rejected: it would still
require trusting parse order, and it discards a genuinely longitudinal signal.

---

## D-008 — Revised contribution: measuring availability-driven bias in acquisition evaluation
**Status:** ACCEPTED (supersedes D-001)
**Date:** 2026-08-09

**Context.** D-001's panel-level novelty claim is void (Yu et al., ICLR 2023). A second novelty
search confirmed that active sensing, EIG acquisition, longitudinal AFA, AFA benchmarking on
PhysioNet 2012, informative missingness, and calibration under missing modalities are all
established. See `docs/NOVELTY_REASSESSMENT.md`.

**Decision.** The project's question becomes:

> When an acquisition policy is evaluated by replaying a historical ICU record, how much of its
> measured benefit survives when the historical measurement-policy shortcut is disrupted — i.e.
> when measurement presence is no longer a free signal?

Method: evaluate identical policies under **support-aware** and **support-blind** disclosure on
identical patients, masks and seeds; report paired ΔAUBC; test whether policy rankings survive
four cost regimes.

**Empirical basis (already executed, E-002).** At a 24h cutoff on 8,000 patients, a model using
only measurement-presence patterns — no measured value at all — reaches AUROC **0.7224
[0.707, 0.738]**, versus 0.8184 values-only and 0.8363 all-blocks. No ratio between these is
claimed until M2 provides a directly comparable tuned full-value baseline on the identical
cohort, split and protocol.

**What we may claim.** That we *quantify* the gap on open data with an assumption-light protocol,
and test ranking stability. **What we may not claim:** that we discovered informative missingness
(Agniel 2018; JAMA Netw Open 2019), that we identified availability-shift evaluation bias
(von Kleist et al., JMLR 26), or anything involving the words *first*, *novel panel acquisition*,
or *clinically costed panels*.

**Honest characterisation.** This is a measurement contribution, not a new method.

**Alternatives rejected.** Uncertainty-gated abstention (L2M 2025 overlaps; method-heavy); full
AFAPE DM/IPW/DRL estimators (require the NUC assumption that fails here); pattern-conditional
calibration as the thesis (MOSAIC 2026 overlaps) — retained as secondary analysis.

---

## D-009 — Benchmark is disclosure replay, not prospective ordering
**Status:** ACCEPTED
**Date:** 2026-08-09

**Context.** the reviewer correctly observed that "at t=24h, buy a panel" was undefined: it can neither
re-order a past observation nor reveal future data without leakage.

**Decision.** The benchmark is **sequential retrospective selective disclosure of historically
recorded values in panel-like feature groups**. At epoch k with boundary t_k, a purchase discloses
all hidden recorded values for that group with timestamp <= t_k. It does not identify or replay
orders, specimens, or events. Targets lie strictly beyond the final boundary. Formal specification in
`docs/BENCHMARK_SPEC.md`.

**Binding wording rule.** We do not describe this as prospective or temporal test ordering.

**Unavailable requests cost full price and disclose nothing.** This is what makes synthetically
hidden and naturally missing indistinguishable to the policy; free failed probes would let a
policy reconstruct the historical ordering pattern, which is precisely the signal E-002 shows is
worth AUROC 0.72.

**Alternatives rejected.** *Genuine prospective epochs.* Not achievable on retrospective data
without the counterfactual "what would this untaken test have shown", which is unidentifiable here.
