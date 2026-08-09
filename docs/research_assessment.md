# Cliniverse — Research & Architecture Assessment

**Status:** v2.0 — **CURRENT SOURCE OF TRUTH**
**Date:** 2026-08-09
**Recommendation:** **MODIFY** (see §14)

---

## 0. History

| Version | Date | Fate |
|---|---|---|
| v1.0 | 2026-08-09 | **Superseded.** Archived verbatim at [`archive/research_assessment_v1_superseded.md`](archive/research_assessment_v1_superseded.md) |
| **v2.0** | 2026-08-09 | **Current.** Rewritten after independent independent review #0 |

v1.0 is preserved unedited as a record of what we believed and why it was wrong. It should be
read only as history. Its central novelty claim was disproved, and it contained a confirmed data
leak. **Do not cite v1.0 for any factual claim.** What changed:

- Its claim that panel-level, shared-cost acquisition was our contribution — **false**
  (Yu et al., ICLR 2023; §3).
- Its dataset description — **wrong**: 36 variables, not 37; a longitudinal variable was
  misclassified as a static, causing post-cutoff leakage (§5.2).
- Its evaluation plan — statistically unsound in places (standalone CI comparisons), and it
  applied a regression conformal method to classification.

Companion documents, all current:
[`REVIEW_RESPONSE_0.md`](REVIEW_RESPONSE_0.md) ·
[`NOVELTY_REASSESSMENT.md`](NOVELTY_REASSESSMENT.md) ·
[`BENCHMARK_SPEC.md`](BENCHMARK_SPEC.md) ·
[`DECISIONS.md`](DECISIONS.md) ·
[`LIMITATIONS.md`](LIMITATIONS.md) ·
[`EXPERIMENTS.md`](EXPERIMENTS.md)

---

## 1. Project hypothesis

Clinical prediction models trained on ICU records learn from two different things: the patient's
physiology, and **the care process that generated the record** — which tests a clinician chose to
order, how often, and how recently. The second is not noise. It is a strong signal, and it is a
property of historical practice rather than of the patient.

This matters most for **information-acquisition methods**, which are evaluated by replaying a
historical record: the set of observations a policy is allowed to acquire is itself the product of
those same historical ordering decisions.

> **Question.** When an acquisition policy is evaluated by replaying a historical ICU record, how
> much of its measured benefit survives when the historical measurement-policy shortcut is
> disrupted?

**Falsifiable form.**

- **H1.** Acquisition policies' budget–performance curves differ materially between
  **support-aware** replay (only historically recorded groups are requestable — standard practice)
  and **support-blind** replay (any group may be requested; unavailable requests cost full price
  and return nothing). Measured as paired ΔAUBC.
- **H2.** The *ranking* of acquisition policies by AUBC is not stable across the four cost regimes
  of [`BENCHMARK_SPEC.md`](BENCHMARK_SPEC.md) §6.
- **H3.** Under support-blind replay, calibration (slope, intercept, reliability) degrades faster
  than discrimination as budget shrinks — i.e. models become confidently wrong before they become
  visibly inaccurate.

All three are falsifiable in both directions. Null results are reportable and will be reported.

---

## 2. Motivating result (executed)

**E-002**, in-hospital mortality, 24h decision point, 5-fold CV, n = 8,000 (sets a+b),
prevalence 14.03%:

| Feature view | Model | AUROC | 95% CI |
|---|---|---|---|
| **availability only** (measurement-presence patterns; **no measured value at all**) | logreg | **0.7224** | [0.707, 0.738] |
| availability only | gbdt | 0.7187 | [0.703, 0.733] |
| values only | gbdt | 0.8184 | [0.807, 0.831] |
| all blocks | gbdt | 0.8363 | [0.825, 0.847] |

A model that never sees a single laboratory or vital value — only counts, ever-measured flags and
recency derived from the observation mask — reaches **AUROC 0.7224**.

**Reporting discipline.** These are executed E-002 numbers on the development cohort. We do
**not** state what fraction of a full model's skill this represents: that ratio requires a
directly comparable, properly tuned full-value baseline on the identical cohort, split and
protocol, which is M2 work. Until M2 lands, the availability figure is reported on its own.

**This phenomenon is not our discovery.** Informative missingness in EHR is established
(Agniel et al. 2018; JAMA Netw Open 2019 reports AUROC ≈ 0.684 from missingness indicators alone
for 30-day mortality; JMIR Med Inform 2021). Our figure is directionally consistent. What we take
from it is narrower and specific to this project: **the acquirable support in a retrospective
acquisition benchmark is itself strongly predictive**, which is why the support-blind protocol is
necessary rather than merely tidy.

---

## 3. Prior art, and what we retract

### 3.1 Closest prior art

**Yu, Li, Kim, Huang, Luo, Wang — *Deep Reinforcement Learning for Cost-Effective Medical
Diagnosis*, ICLR 2023 (arXiv:2302.10261).** SM-DDPO learns a policy that **selects lab test panels
sequentially**; the action space is over **panels of medical tests (groups acquired together)**;
each group is assigned a **shared cost** from MIMIC-IV timestamps; evaluated on ferritin
abnormality, **sepsis mortality** and AKI; reports **Pareto cost-vs-performance curves** (AKI
testing cost $591 → $90; up to 85% cost reduction).

### 3.2 Other established work

| Work | What it already does |
|---|---|
| von Kleist et al., **JMLR 26** + arXiv:2312.03619 | Formalise AFA performance evaluation under **feature availability distribution shift**; NDE/NUC/positivity assumptions; DM, IPW, Double-RL estimators to **correct** the bias |
| Yoon et al. — Deep Sensing (ICLR 2018), **ASAC** (MLHC 2019) | Active sensing on clinical time series: what to measure and when, under cost |
| Ma et al. — **EDDI** (ICML 2019) | Partial-VAE + expected information gain, motivated by ordering diagnostic tests |
| Jarrett et al. — **Clairvoyance** (ICLR 2021) | Medical time-series pipeline with an integrated information-acquisition pathway |
| **NOCTA** (2025), **L2M** (2025), **AFABench** (2025) | Longitudinal non-greedy cost-aware AFA; in-context AFA with uncertainty; a standardized AFA benchmark **already including PhysioNet 2012** |
| Agniel et al. 2018; JAMA Netw Open 2019; JMIR Med Inform 2021 | **Informative missingness is established**; indicators alone predict mortality |
| MOSAIC / pattern-calibrated multimodal prediction (2026) | Calibration conditional on the observed-modality pattern under blockwise missingness |
| ETHOS (npj Digit Med 2024), EHRSHOT (NeurIPS 2023) | Generative patient timelines; longitudinal EHR benchmark with released foundation model |

### 3.3 Retracted claims

Every claim below appeared in v1.0 and is **withdrawn**:

| Retracted | Refuted by |
|---|---|
| Panel-level / shared-cost acquisition is our contribution | Yu et al., ICLR 2023 |
| "Prior AFA only buys individual features at individual cost" | Yu et al., ICLR 2023 |
| Active acquisition on longitudinal patient models is novel | Deep Sensing 2018; ASAC 2019; EDDI 2019; Clairvoyance 2021; NOCTA 2025 |
| A temporal clinical AFA benchmark is an open gap | AFABench 2025 (includes PhysioNet 2012) |
| Identifying availability-driven evaluation bias is our insight | von Kleist et al., JMLR 26 |
| Showing missingness is predictive is a finding of ours | Agniel 2018; JAMA Netw Open 2019 |
| Calibration under acquisition is unexamined | L2M 2025; MOSAIC 2026 |

**Forbidden vocabulary, enforced at review:** *first*, *novel panel acquisition*, *clinically
costed panels*, *prior AFA only buys individual features*, *we discovered informative
missingness*, *clinician-inspired* (unless a clinician actually designed it).

### 3.4 Candidate contribution — what remains

Narrow, and stated at its true size:

1. **Quantifying the gap, not just correcting it.** von Kleist et al. prove naive evaluation is
   biased and propose estimators; they do not report *how large* the inflation is for concrete
   policies. We measure it directly.
2. **Support-blind replay as an executable protocol** rather than a statistical correction.
   Assumption-light, runs on credentialing-free data, and answers "how much of this advantage was
   availability?" It does **not** recover deployment value, and we do not claim it does.
3. **Ranking stability of acquisition policies** across cost regimes and disclosure protocols as
   the object of study — a recognised genre elsewhere, not yet applied to AFA.

**Honest characterisation: this is a measurement contribution, not a new method.** We are not
proposing a better acquisition algorithm. A reviewer who values novel methods over careful
measurement will rate it lower; we accept that deliberately.

---

## 4. Technical risks

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| RISK-1 | **Retrospective AFA evaluation is confounded.** NUC is false in ICU data — a lactate exists because someone was worried, using information absent from the record. | Critical | We do not estimate deployment value. Support-blind vs support-aware comparison under a disclosed synthetic mechanism; scope limit stated in every document, payload and figure (§9) |
| RISK-2 | **Support leakage into the policy.** Given §2, any leak of the hidden support turns the benchmark into a test of reading historical practice. | Critical | `PolicyView` holds no evaluator state; unavailable requests cost full price and disclose nothing; non-inferability tests |
| RISK-3 | **Effective sample size.** 12,000 patients, 1,707 deaths. Many policies × many budget points invites false positives. | High | Paired patient-level inference; pre-registered primary comparison; bootstrap CIs; set-c quarantined |
| RISK-4 | **Baseline strength.** GBDT on summary features is strong; a deep model losing to it is the expected outcome. | High | Baselines first; publish the number and beat it or concede |
| RISK-5 | **Cost model is an assumption.** P12 has no prices. | Medium | Four dimensionless regimes; sensitivity is a headline, not an appendix |
| RISK-6 | **Further cutoff leaks.** One was found and fixed (§5.2); others may exist. | Medium | `tests/test_leakage.py` asserts the general post-cutoff property, not per-variable checks |
| RISK-7 | **Null result.** If the support-blind gap is small, the headline weakens. | Medium | Accepted. "Acquisition gains are more robust than feared" is honest and reportable |

---

## 5. Dataset

### 5.1 PhysioNet/CinC Challenge 2012 — verified

Access verified by unauthenticated HTTP, 2026-08-09. **Open Data Commons Attribution License
v1.0. No credentialing, no CITI training, no DUA.** ~20 MB total.

Outcomes are published for **all three sets**, giving **12,000 labeled patients** — not the 4,000
commonly cited from the original challenge, where B and C were withheld.

| Set | n | In-hospital deaths | Rate |
|---|---|---|---|
| set-a | 4,000 | 554 | 13.85% |
| set-b | 4,000 | 568 | 14.20% |
| set-c | 4,000 | 585 | 14.62% |

Outcome columns: `RecordID, SAPS-I, SOFA, Length_of_stay, Survival, In-hospital_death`.
SAPS-I and SOFA are legitimate **baselines** but would be **leakage as features** — they are
computed from the same 48h window. A test enforces this.

### 5.2 Corrected structure — **37 time-series variables**

v1.0 reported 36. That was wrong, and the error was not cosmetic.

`Weight` is recorded **throughout** the stay, not once at admission. Verified in set-a: of
**129,165** Weight rows, only **4.1%** are at hour 0; **95.9%** come later and **52.1%** are at
hour ≥ 24, affecting **2,726 of 4,000 patients (68%)**. The parser tested membership in the static
schema *before* looking at the timestamp, so a model with a 24h cutoff **received weights measured
as late as hour 47**.

**Fix (D-007):** `Weight` is a time-series variable (the 37th); a new static `AdmissionWeight` is
declared `source_parameter: Weight, at_hour: 0`; the parser resolves **time before routing**.
A second bug surfaced during the fix — `AdmissionWeight` disagreed with the hour-0 grid cell for
65/3,701 patients (1.76%) because statics took the last hour-0 row while binning takes the
within-hour mean — so statics sourced from a time-series parameter are now **derived from the
binned cell**, making both views agree by construction.

**Audit:** `Age`, `Gender`, `Height`, `ICUType`, `RecordID` each appear exactly 4,000 times in
set-a, all at hour 0. **Weight was the only offender.**

### 5.3 Corrected statistics (production parser, set-a)

| Statistic | v1.0 (wrong) | **Current** |
|---|---|---|
| Time-series variables | 36 | **37** |
| **Binned grid occupancy** | 19.35% | **20.25%** |
| **Binned missingness** | 80.65% | **79.75%** |
| Raw row-count occupancy bound | 23.28% | 24.46% |
| Degenerate records | 3 | 3 |

The raw row-count figure is **a loose upper bound only** — it counts `-1` sentinel rows and
within-hour collisions. **It is never the reported missingness statistic**; that is always
`Cohort.describe()['grid_occupancy']`.

The three degenerate records (`140501`, `140936`, `141264`) contain **only** `Weight,-1`, the
missing sentinel, which the plausibility filter correctly drops. They are retained and flagged,
never silently removed.

### 5.4 Acquisition structure

Coverage splits into a near-continuous monitoring tier (HR, Temp, GCS, Urine, ABP variants;
15–58 observations per record) and a discretely measured laboratory tier (Creatinine, BUN, HCT,
Platelets, WBC, Na, HCO3, K, Mg, Glucose; **~3.3–3.6 measurements per 48h stay**), plus a
selective tier (Lactate 0.55 down to TroponinI 0.051). The laboratory tier's discreteness is what
makes an acquisition problem well-posed here.

### 5.5 Secondary and rejected datasets

| Dataset | Verdict |
|---|---|
| **Synthea** | **SECONDARY — controlled generator only.** Hand-authored clinician-designed state machines; information-gain results would be artifacts of the generator's rules. Used only where a known ground-truth mechanism is the point (structured missingness, OOD). Never primary evidence. |
| **MIMIC-IV demo** (100 patients, ODbL) | Demo only; far too small |
| **MIMIC-IV, eICU, HiRID, AmsterdamUMCdb** | Credentialed. Application may proceed in parallel but the project must be complete without them |
| **EHRSHOT** | Gated; stretch only |

---

## 6. Benchmark estimand

Full specification in [`BENCHMARK_SPEC.md`](BENCHMARK_SPEC.md). Summary:

TwinBench is **sequential selective disclosure (replay) of historically recorded co-measured
events under a budget**. It is **not** prospective test ordering, and is never described as such:
a policy cannot cause a test to be performed that was never performed.

- Decision epochs at `t = 12, 18, 24` hours; prediction time `T = 24`.
- **BR-1:** no value with timestamp `> t_k` may enter the policy view, model input, or any
  derived feature. Enforced by `Cohort.truncate`.
- **BR-2:** every target has support strictly `> T`.
- **Headline:** `Δ AUBC(π) = AUBC(π; support_blind) − AUBC(π; support_aware)` on identical
  patients, masks, seeds, epochs and predictive model, as a **paired patient-level bootstrap CI**.
- **Unavailable requests cost full price and disclose nothing** — otherwise a policy could probe
  availability for free and reconstruct exactly the signal §2 shows is worth AUROC 0.72.

---

## 7. Tasks

| Task | Type | Definition | Status |
|---|---|---|---|
| **T1** | Binary | In-hospital mortality from data ≤ 24h | **Primary** |
| **T3** | Scalar regression | Mean creatinine over hours (24, 48], for patients with ≥1 creatinine in that window | **Secondary** |
| ~~T2~~ | — | LOS > 3 days | **Dropped** — arbitrary threshold, confounded by the discharge/death process |

T3 is restricted to patients with an observed target — a **selection bias**, reported as a
limitation rather than absorbed silently.

---

## 8. Models, uncertainty, acquisition

**Model tiers (each must beat the one below before the next is built).** Prevalence → SAPS-I /
SOFA → logistic regression → **GBDT on summary features** (expected strongest) → GRU-D →
temporal transformer (only if it wins). Feature blocks are kept disjoint — `AVAILABILITY`,
`VALUES`, `STATICS` — so any of them can be trained on alone.

**Uncertainty.** Deep ensembles (5 seeds). For **T1**: calibrated probabilities, reliability
curves, **calibration slope and intercept**, risk–coverage / AURC. Brier and log-loss are reported
as **proper scoring rules** — they combine calibration and refinement and are not calibration
metrics on their own. ECE is secondary only (biased, binning-sensitive estimator: Nixon 2019;
Kumar 2019; Roelofs 2022). For **T3**: **split conformal** intervals, coverage stratified by
missingness pattern and budget. **CP-MDA is not applied to classification** — it is a regression
method for missing covariates.

**Acquisition policies.** No acquisition; full-observation ceiling (labelled an **unattainable
diagnostic upper bound**); random; most-missing-first; static importance; cost-normalised
importance; **fixed domain-motivated ordering** (authored by the engineering team from observed
frequency — *not* clinician-designed); ensemble-uncertainty heuristic; feature-level EIG (EDDI-style);
group-level EIG; learned greedy discriminative policy. A faithful SM-DDPO reproduction is **out of
scope** — reproducing an RL method tuned on MIMIC-IV would yield a strawman; we instead adopt its
*evaluation setting* and record this as a stated limitation.

**Statistics.** Paired patient-level inference throughout; standalone non-overlapping CIs are not
used for policy comparison. Primary comparison pre-registered before final results.

---

## 9. Scope limits, cloud, and what not to build

**Scope limit, binding and repeated in payloads and figures:** TwinBench measures *relative policy
performance under a disclosed synthetic mechanism*. It does **not** estimate clinical utility or
deployment value.

**GCP.** Cloud Run (scales to zero) + Cloud Storage + Artifact Registry only; projected spend
**< $15**. **No persistent Vertex AI endpoint** — they bill from provision to undeploy including
idle (A100 ≈ $2.93/GPU-hr; L4 ≈ $0.70/GPU-hr), which alone would exhaust $300.

**Not building:** MedGemma (a medical *vision-language* model, useless for numeric time series,
and cost-prohibitive); BigQuery (our data is 20 MB); Healthcare API / FHIR store; Firestore;
clinical notes, imaging, wearables, medication or diagnosis modalities (**absent from the
dataset**); RL acquisition; diffusion trajectory models; **LLM-generated explanations**; any
causal or treatment-effect claim; production UI (deferred).

---

## 10. Limitations

See [`LIMITATIONS.md`](LIMITATIONS.md) in full. The load-bearing ones:

1. **We cannot estimate real-world clinical value of any acquisition policy.** NUC is false here.
2. Panel-like groups are **co-measurement clusters, not recorded laboratory orders**; named
   `*_like` throughout.
3. Costs are **dimensionless relative units**, not prices, and never economic findings.
4. Population is adult ICU stays ≥48h from a limited set of units, reflecting historical practice.
5. T3 conditions on an observed target — selection bias.
6. Availability-only performance is reported as a raw executed figure; **no ratio to full-model
   skill is claimed until M2 provides a directly comparable baseline**.

---

## 11. Implementation roadmap

| M | Deliverable | Exit criterion |
|---|---|---|
| **M0** | Repo, tooling, CI, P12 loader, splits, leakage guards | **Done** — ruff/mypy clean, 95 tests |
| **M1** | TwinBench: case schema, epochs, masking mechanisms, `PolicyView`, disclosure engine, support-blind/aware protocols, seeded manifests | Regeneration reproduces identical content hashes; non-inferability tests pass |
| **M2** | Baselines + full metric suite; **directly comparable full-value baseline** | Real AUROC/AUPRC/Brier with paired CIs; GBDT ≥ SOFA |
| **M3** | Ensembles, calibration, conformal, abstention | Coverage reported stratified by pattern, including failures |
| **M4** | **Acquisition — the core**: all policies × 2 protocols × 4 cost regimes | Paired ΔAUBC with CIs; H1/H2/H3 answered including nulls |
| **M5** | Ablations, robustness sweep, OOD, cost sensitivity | Each component justified or removed |
| **M6** | Minimal API + inspection UI | Lowest priority; may be cut for experiments |
| **M7** | set-c evaluated **once**; independent review #1 response | Every claim traceable to an executed run |

---

## 12. Success criteria

1. TwinBench regenerates bit-identically from seeds (CI-asserted).
2. Policy cannot infer evaluator-only support (test-asserted).
3. GBDT baseline beats SOFA on T1 with paired CIs.
4. Budget–performance curves for ≥6 policies × ≥8 budget points × 2 protocols × 4 cost regimes.
5. **H1 answered** with a paired ΔAUBC CI — in either direction.
6. **H2 answered** with a rank-reversal count.
7. **H3 answered** with calibration-vs-budget curves.
8. Every number traceable to run ID + git SHA + config hash + seed.
9. GCP spend < $50.

---

## 13. Positioning

> A model that never sees a single lab value reaches **AUROC 0.7224** using only
> measurement-presence patterns. Clinical AI can learn the care process itself — not just patient
> physiology. Cliniverse stress-tests whether information-acquisition methods still look reliable
> when that historical measurement-policy shortcut is disrupted.

0.7224 is an **executed E-002 result** on the development cohort (n = 8,000, 5-fold CV,
95% CI [0.707, 0.738]). It is an associational finding about what a model can predict from
measurement-presence patterns. It is **not** a causal claim, and it does not assert that any model
infers clinician intent.

---

## 14. Recommendation: **MODIFY**

Not GO — the original contribution is void. Not PIVOT — the dataset, pipeline and E-002 result
remain valid and load-bearing, and all M0 engineering is reused unchanged.

**Proceed** with the revised thesis of §1, the estimand of §6, and the claim discipline of §3.3.

---

## 15. Primary sources

**Closest prior art** — Yu, Li, Kim, Huang, Luo, Wang, *Deep Reinforcement Learning for
Cost-Effective Medical Diagnosis*, ICLR 2023 — https://arxiv.org/abs/2302.10261

**Acquisition** — EDDI, ICML 2019 (https://proceedings.mlr.press/v97/ma19c.html) ·
ASAC, MLHC 2019 (https://arxiv.org/abs/1906.06796) · Deep Sensing, ICLR 2018 ·
Clairvoyance, ICLR 2021 (https://github.com/vanderschaarlab/clairvoyance) ·
NOCTA (arXiv:2507.12412) · L2M (arXiv:2510.12624) · AFABench (arXiv:2508.14734) ·
A2MT (arXiv:2211.05039) · RL with Efficient AFA (arXiv:2011.00825)

**Evaluation methodology** — von Kleist et al., JMLR 26
(https://www.jmlr.org/papers/volume26/23-1635/23-1635.pdf) · static companion (arXiv:2312.03619)

**Uncertainty & calibration** — Zaffran et al., ICML 2023
(https://proceedings.mlr.press/v202/zaffran23a.html) · Lakshminarayanan et al., NeurIPS 2017 ·
Nixon et al. 2019; Kumar et al. 2019; Roelofs et al. 2022

**Informative missingness** — Agniel et al. 2018 · JAMA Netw Open 2019 · JMIR Med Inform 2021

**Longitudinal modeling** — ETHOS, npj Digit Med 2024
(https://www.nature.com/articles/s41746-024-01235-0) · EHRSHOT, NeurIPS 2023
(https://arxiv.org/abs/2307.02028)

**Datasets** — PhysioNet/CinC Challenge 2012, ODC-BY v1.0
(https://physionet.org/content/challenge-2012/1.0.0/) · MIMIC-IV v2.2 (credentialed) ·
Synthea, JAMIA 25(3):230 (https://academic.oup.com/jamia/article/25/3/230/4098271)

**Cloud** — Vertex AI pricing (https://cloud.google.com/vertex-ai/pricing) · MedGemma model card
(https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)
