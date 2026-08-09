# Novelty Reassessment — Round Two

**Date:** 2026-08-09
**Trigger:** independent review #0 disproved our first remaining novelty claim (panel-level AFA).
**Verdict:** **MODIFY** — a defensible and competitive contribution remains, but it is a different
one, and it is narrower than what we claimed before.

---

## 1. Known prior art

Searched across active feature acquisition, value of information, informative missingness,
missing-modality calibration, benchmark-stability meta-science, and clinical test-ordering RL.

| Prior work | What they already do |
|---|---|
| **Yu et al., ICLR 2023** (arXiv:2302.10261) — *DRL for Cost-Effective Medical Diagnosis* | **Closest prior art.** Sequential **panel-level** test acquisition (CBC, CMP), groups acquired together, **shared group cost** derived from MIMIC-IV timestamps, clinical endpoints (sepsis mortality, AKI, ferritin), Pareto cost-vs-performance curves, up to 85% cost reduction. |
| **von Kleist et al., JMLR 26** + static companion (arXiv:2312.03619) | Formalize **active feature acquisition performance evaluation (AFAPE)** under **feature availability distribution shift**. Identify NDE/NUC/positivity assumptions; propose DM, IPW and Double-RL estimators to **correct** the bias. Explicitly warn that evaluation bias "might lead to false promises about the agent's performance." |
| **Yoon et al.** — Deep Sensing (ICLR 2018), ASAC (MLHC 2019) | Active sensing on clinical time series: what to measure and when, under cost. |
| **Ma et al., EDDI (ICML 2019)** | Partial-VAE + expected information gain, motivated by ordering diagnostic tests. |
| **Jarrett et al., Clairvoyance (ICLR 2021)** | Medical time-series pipeline with an integrated information-acquisition pathway. |
| **NOCTA (2025)**, **L2M (2025)**, **AFABench (2025)** | Longitudinal non-greedy cost-aware AFA; in-context AFA with uncertainty; a standardized AFA benchmark **that already includes PhysioNet 2012**. L2M explicitly notes methods that ignore missingness structure "inherit acquisition bias and produce poorly calibrated uncertainty estimates." |
| **Agniel et al. 2018**; **JAMA Netw Open 2019**; **JMIR Med Inform 2021** | **Informative missingness is established.** Missingness indicators *alone* predict mortality — the 2019 study reports AUROC ≈ 0.684 for 30-day mortality from indicators only. |
| **MOSAIC / pattern-calibrated multimodal prediction (2026)**; evidential fusion under missing modalities | Calibration conditional on the **observed-modality pattern** under blockwise missingness. |
| **GRN benchmarking ranking-instability (2026), StabilityBench (2026)** | Ranking instability across evaluation-protocol axes as an object of study — 16–32% reversal rates — but in other domains, not AFA. |

---

## 2. What they already do — claims we can no longer make

Every one of these was in our previous documents and is now **retracted**:

| Retracted claim | Refuted by |
|---|---|
| Panel-level / shared-cost acquisition is our contribution | Yu et al., ICLR 2023 |
| "Prior AFA only buys individual features at individual cost" | Yu et al., ICLR 2023 |
| Active acquisition on longitudinal patient models is novel | Deep Sensing 2018, ASAC 2019, EDDI 2019, Clairvoyance 2021, NOCTA 2025 |
| A temporal clinical AFA benchmark is an open gap | AFABench 2025 (includes PhysioNet 2012) |
| Identifying availability-driven evaluation bias is our insight | von Kleist et al., JMLR 26 |
| Showing missingness is predictive is a finding | Agniel 2018; JAMA Netw Open 2019 |
| Calibration under acquisition is unexamined | L2M 2025; MOSAIC 2026 |

**Forbidden vocabulary, enforced in review:** *first*, *novel panel acquisition*, *clinically
costed panels*, *prior AFA only buys individual features*, *we discovered informative
missingness*.

---

## 3. What we still do differently

Three things survive. They are narrow, and I state them at the size they actually are.

**(a) The bias is formalized and corrected in the literature, but not *quantified* on open data.**
von Kleist et al. prove naive AFAPE is biased and propose estimators. They do not report *how
large the inflation is* for concrete acquisition policies. Nobody has published: *"policy P
reports X% cost savings under standard support-aware replay and Y% when availability is no longer
a free signal."* Our E-002 result shows availability alone gives AUROC 0.7224 on this dataset, which makes the
measurement worth making.

**(b) Support-blind replay as an executable protocol, not an estimator.**
The AFAPE response is *statistical correction* (IPW/DM/DRL) requiring assumptions that are false
in ICU data. Our response is *protocol design*: charge full cost for unavailable requests so that
synthetic-hidden and naturally-missing are indistinguishable to the policy. This is cruder than
their estimators and does **not** recover deployment value — but it is assumption-light, it runs on
open data, and it directly answers "how much of this policy's advantage was availability?"

**(c) Ranking stability of acquisition policies as the object of study.**
Ranking instability is a recognised genre in other fields (GRN benchmarking, LLM stability) and
AFABench compares methods, but under one protocol. Whether the *ordering* of acquisition policies
survives changes in cost regime and disclosure protocol is, as far as I can find, unmeasured. If
rankings flip, that is a result about the field's evaluation practice.

**Honest caveat:** (a) and (b) are contributions to *measurement*, not to method. We are not
proposing a better acquisition algorithm. A reviewer who values novel methods over careful
measurement will rate this lower, and that is a real risk we are accepting deliberately.

---

## 4. Defensible contribution — the revised thesis

> **Question.** When an acquisition policy is evaluated by replaying a historical ICU record, how
> much of its measured benefit survives when the historical measurement-policy shortcut is
> disrupted — i.e. when measurement presence is no longer a free signal?
>
> **Method.** Evaluate identical policies under two disclosure protocols on identical patients,
> masks and seeds — **support-aware** (standard practice: only historically recorded panels can be
> requested, so availability is a free signal) and **support-blind** (any panel may be requested;
> unavailable requests cost full price and return nothing). Report paired ΔAUBC, and test whether
> policy *rankings* survive four cost regimes.
>
> **Foundation (already measured, E-002).** At a 24h decision point on 8,000 patients, a model
> using **only** which tests were ordered — no measured value whatsoever — reaches
> **AUROC 0.7224 [0.707, 0.738]**, versus 0.8184 for values-only and 0.8363 for the all-blocks
> model. No ratio between these is claimed until M2 provides a directly comparable, tuned
> full-value baseline on the identical cohort, split and protocol.

This is falsifiable in both directions. If support-blind and support-aware give the same rankings
and similar gains, we report that acquisition benefits are robust — a useful negative result. If
they diverge, we have quantified an evaluation bias the field currently corrects only in theory.

---

## 5. Hackathon differentiator — the Best Overall gate

The five required answers, honestly:

**1. One healthcare ML failure understandable in <20 seconds.**
> "A model that never sees a single lab value reaches AUROC 0.7224 using only
> measurement-presence patterns. Clinical AI can learn the care process itself — not just
> patient physiology."

**2. One technical mechanism.**
Support-blind disclosure replay: a benchmark protocol where requesting an unavailable panel costs
full price and returns nothing, making clinician-driven availability unusable as a signal. Paired
with a panel co-measurement structure derived from the data (E-001) and four cost regimes.

**3. One stress test judges can SEE.**
Side-by-side: the same acquisition policy run under support-aware and support-blind replay, with
its budget–performance curve collapsing (or not) in real time, and a live counter showing spend on
panels that returned nothing.

**4. One measurable result we can produce.**
Paired ΔAUBC between protocols with bootstrap CIs, plus a rank-reversal count across cost regimes.
The foundational number (availability-only AUROC 0.7224, 95% CI [0.707, 0.738]) is **already
executed**, not projected.

**5. One reason this differs from the closest prior work.**
Yu et al. optimise a policy *within* support-aware replay and report the resulting cost savings.
von Kleist et al. prove that number is biased and propose estimators under assumptions that fail in
ICU data. We do neither: we **measure the size of the gap directly** with an assumption-light
protocol on credentialing-free data, and test whether method rankings survive it.

All five are answerable. **The gate passes** — but on measurement quality and framing, not on
methodological novelty. That is the honest characterisation.

---

## 6. Verdict

**CURRENT THESIS IS COMPETITIVE — AS MODIFIED.** Not as originally stated.

Reasons to proceed rather than pivot:
- The core empirical hook is **already measured** on real data, not hoped for.
- All M0 engineering (loader, 37-variable schema, leakage guards, splits, panel derivation,
  feature blocks) is reused unchanged.
- The failure mode is visceral and legible to non-specialist judges in one sentence.
- Both possible outcomes are publishable; we are not staking the project on a hoped-for direction.

Reasons for caution, stated plainly:
- It is a measurement contribution, not a new method.
- The underlying phenomenon (informative missingness) is well established; **only its consequence
  for acquisition-policy evaluation is ours**, and we must say so every time.
- If the support-blind gap turns out to be small, the headline weakens to "acquisition gains are
  more robust than feared" — still honest and worth reporting, but a quieter result.

### Alternatives considered and not taken

Recorded so the choice is auditable, not to pad the document.

1. **Uncertainty-gated abstention under budget** ("know when to stop testing"). Attractive and
   clinically meaningful, but L2M (2025) already does uncertainty-guided acquisition, and this
   would be a method contribution needing more runway than we have.
2. **Full AFAPE estimator implementation** (DM/IPW/DRL on P12). Most rigorous option, but it
   requires NUC — which is precisely what fails here — so it would inherit the assumption we are
   trying to avoid depending on, and it is heavy machinery for the timeframe.
3. **Pattern-conditional calibration under panel-shaped missingness.** Genuinely interesting, but
   MOSAIC (2026) occupies much of it. Retained as a **secondary** analysis inside the main
   experiment rather than as the thesis.
