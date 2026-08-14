# Cliniverse — Research & Architecture Assessment

**Status:** v1.0 — **PARTIALLY SUPERSEDED, 2026-08-09, after independent review #0**
**Date:** 2026-08-09
**Author:** Research/ML engineering pass over the proposed project thesis

> ## ⚠ Corrections — read before citing anything below
>
> This document is retained as a record of the initial assessment. Several of its
> load-bearing claims have since been **disproved** and must not be reused.
>
> | Section | Claim | Status |
> |---|---|---|
> | §3.2(a) | Panel-structured acquisition is our contribution; "every AFA method surveyed acquires individual features at individual cost" | **FALSE.** Yu et al., ICLR 2023 (arXiv:2302.10261) already performs sequential panel-level acquisition with shared group costs on MIMIC-IV. Retracted. |
> | §3.2(b) | A temporal clinical AFA benchmark is an open gap | **Weakened.** AFABench (2025) already benchmarks AFA including on PhysioNet 2012. |
> | §5.1, §13 | 36 time-series variables | **WRONG — it is 37.** `Weight` is longitudinal and was misclassified as a static, which also caused a confirmed post-cutoff leak. See D-007. |
> | §5.1 | "≥76.7% missing"; occupancy 23.28% | **Superseded.** That is a loose raw row-count bound including sentinel rows. Corrected raw bound 24.46%; the real production figure is **binned occupancy 20.25%, missingness 79.75%**. |
> | §6.2 | Task T2 (LOS > 3 days) | **DROPPED** — arbitrary threshold, confounded by the discharge/death process. |
> | §8 | CP-MDA for classification | **WRONG.** CP-MDA is a regression method for missing covariates; scope corrected. |
> | §8 | Brier/NLL described as calibration metrics | **Imprecise.** They are proper scoring rules combining calibration and refinement. |
> | §9 | "clinician-inspired heuristic" | **Renamed** to "fixed domain-motivated ordering" — no clinician designed or validated it. |
> | §10, §15 | "set-c is locked / never touched" | **Reworded** to "quarantined from model fitting and model selection following an aggregate cohort audit". |
> | §15 | Comparisons via non-overlapping standalone CIs | **Replaced** by paired patient-level inference on ΔAUBC. |
>
> **Current authoritative documents:**
> [`REVIEW_RESPONSE_0.md`](../REVIEW_RESPONSE_0.md) (findings and verification) ·
> [`NOVELTY_REASSESSMENT.md`](../NOVELTY_REASSESSMENT.md) (what we may claim) ·
> [`BENCHMARK_SPEC.md`](../BENCHMARK_SPEC.md) (estimand and information boundary) ·
> [`DECISIONS.md`](../DECISIONS.md) D-007/D-008/D-009.
>
> §1–2 (literature survey), §4.1 (the AFAPE risk), §5 (dataset access), §11–12 (GCP and
> components not to build) remain valid.

> **Reading note.** This document is deliberately adversarial toward our own proposal.
> Its purpose is to establish what is defensible *before* code is written. Section 3 concludes
> that the project's stated "defining innovation" is **not novel**, and Section 13 proposes a
> reframing. Everything numeric in this document that describes our data was produced by
> executing code against the real dataset (see §5.1); nothing is estimated or recalled.

---

## 1. Project hypothesis

### 1.1 As originally stated

> Given incomplete longitudinal patient information, can an AI system (1) construct a latent
> patient state, (2) model multiple plausible futures, (3) estimate uncertainty, (4) determine
> which missing observation would most reduce uncertainty, (5) update beliefs on new evidence,
> (6) support counterfactual simulation, and (7) explain state changes?

### 1.2 Assessment of the hypothesis as a research question

The seven capabilities are **each individually solved problems** with published methods and, in
most cases, public code. As a research question, "can this be done?" has a known answer: yes.
Restating it as a hypothesis produces a project that cannot fail and therefore cannot be
evidence of anything.

The hypothesis must be narrowed to something falsifiable. The version I propose we actually test:

> **H1 (primary).** Under a fixed, disclosed observation budget, an acquisition policy that
> selects *clinically-costed lab panels* using expected information gain outperforms
> (a) random acquisition, (b) most-missing-first, (c) static global feature importance, and
> (d) a clinician-inspired fixed panel ordering — measured by area under the
> budget-vs-performance curve on held-out patients.
>
> **H2.** Acquisition chosen to reduce *predictive uncertainty* improves **calibration**
> (Brier, NLL, conformal interval width at fixed coverage), not merely discrimination (AUROC).
> This is under-reported in the AFA literature, which overwhelmingly reports accuracy/F1 vs cost.
>
> **H3 (falsifiable, and we should expect it may fail).** The advantage of a learned/EIG policy
> over the *clinician-inspired panel heuristic* is small. Prior work on this dataset suggests
> gradient-boosted trees on summary features are extremely strong and that simple ordering
> heuristics are hard to beat. **If H3's null holds, we report that.**

H1–H3 are answerable in a hackathon timeframe with the compute we have. The original
seven-part formulation is not.

---

## 2. Similar existing research

Grouped by what part of our proposal each piece of prior work already covers.

### 2.1 Active feature acquisition / value of information — the core idea

| Work | Venue | What it already does |
|---|---|---|
| **Deep Sensing** — Yoon, Zame, van der Schaar | ICLR 2018 | Active sensing on clinical time series; decides what to measure and when, under cost. |
| **ASAC: Active Sensing using Actor-Critic** — Yoon, Jordon, van der Schaar | MLHC 2019 | Selector/predictor networks; explicitly "what and when to observe when observations are costly", on real medical datasets. |
| **EDDI: Efficient Dynamic Discovery of High-Value Information** — Ma et al. | ICML 2019 | Partial-VAE + expected-information-gain acquisition; **explicitly motivated by ordering diagnostic tests**; evaluated on two healthcare applications. |
| **Clairvoyance** — Jarrett, Yoon, Bica, Qian, Ercole, van der Schaar | ICLR 2021 | A *pipeline toolkit* for medical time series with an integrated **information-acquisition / active-sensing** pathway alongside prediction and treatment-effect estimation. |
| **RL with Efficient Active Feature Acquisition** — Yin et al. | arXiv 2020 | Sequential VAE + policy deciding when to acquire expensive information; evaluated on a **sepsis medical simulator**. |
| **NOCTA** — Non-Greedy Objective Cost-Tradeoff Acquisition | arXiv 2025 | **Longitudinal** AFA, non-greedy, joint predictive-loss + acquisition-cost objective, on real medical datasets. |
| **L2M: Learning-To-Measure** | arXiv 2025 | In-context AFA; uncertainty quantification under arbitrary missingness; greedy CMI-maximizing agent. |
| **A2MT** — Active Acquisition for Multimodal Temporal Data | arXiv 2022 | **Temporal** modality-level acquisition under cost (Perceiver IO). Non-clinical (Kinetics-700, AudioSet). Notably reports agents **failed to learn adaptive strategies** — a useful negative result. |

### 2.2 Benchmarking and evaluation of AFA

| Work | What it covers |
|---|---|
| **AFABench** (arXiv 2508.14734) | Standardized AFA benchmark. Implements static (PT-S, CAE-S), myopic (EDDI-GM, GDFS-DM, DIME-DM), and RL (JAFA, OL, ODIN-MF/MB) methods plus AACO. **Includes PhysioNet Challenge 2012** as a clinical dataset. Soft- and hard-budget protocols. **Explicitly static — no temporal/longitudinal acquisition.** |
| **AFAPE** — Evaluation of AFA Methods for *Time-varying* Feature Settings (von Kleist et al., JMLR vol. 26) | The methodological core. Formalizes why estimating real-world AFA performance from retrospective data is biased, and what assumptions make it identifiable. See §4.1 — this is the single most important paper for our benchmark's defensibility. |
| **AFA for Static Feature Settings** (arXiv 2312.03619) | Companion static-setting treatment of the same evaluation problem. |

### 2.3 Longitudinal patient modeling / "patient world models"

| Work | What it covers |
|---|---|
| **ETHOS** — Zero-shot health trajectory prediction using transformers (npj Digital Medicine, 2024) | Tokenizes Patient Health Timelines, GPT-2-style decoder, **generates multiple plausible future timelines**, zero-shot across tasks. This is essentially our "world model with multiple futures". |
| **EHRSHOT** (NeurIPS 2023 D&B) | Longitudinal (non-ICU-restricted) EHR benchmark, 6,739 Stanford patients, 15 tasks, releases CLMBR-T-base (141M) foundation model weights. |
| **CLMBR / MOTOR / Med-BERT / CEHR-BERT** | Structured-EHR foundation models — the pretraining direction. |
| **GRU-D** and successors | The canonical irregular-sampling architecture using masking + time-decay. Still competitive; recent work ("Still Competitive: Revisiting Recurrent Models for Irregular Time Series", arXiv 2510.16161) argues RNNs remain hard to beat here. |

### 2.4 Uncertainty under missingness

| Work | What it covers |
|---|---|
| **Conformal Prediction with Missing Values** — Zaffran, Dieuleveut, Josse, Romano, ICML 2023 | Directly load-bearing for us. Marginal coverage holds on imputed data for essentially any missingness distribution and imputation function, **but average coverage varies by missingness pattern — intervals undercover conditional on some patterns.** Proposes missing-data-augmentation (CP-MDA) for pattern-conditional validity. Code public. |
| **Deep Ensembles** — Lakshminarayanan et al., NeurIPS 2017 | The strong, simple uncertainty baseline. |
| Calibration measurement critiques — Nixon et al. 2019; Kumar et al. 2019; Roelofs et al. 2022 | **ECE is a biased and inconsistent estimator**, sensitive to binning hyperparameters. Recommendation is equal-mass binning with bias correction, and reporting proper scoring rules alongside. Constrains how we may report calibration (§8). |

### 2.5 Informative missingness in EHR

Agniel et al. (2018) and successors establish that in real EHR, **the fact that a lab was ordered
is itself strongly predictive** — missingness encodes clinician concern, disease severity, and
care pathway. Missingness indicators improve models; naive imputation that discards them
destroys signal *and* the data are Not-Missing-At-Random. This has a direct and severe
consequence for our benchmark design (§4.1).

---

## 3. What is actually novel vs. what already exists

### 3.1 The blunt verdict

**The project's stated defining innovation — "active information acquisition applied to a
longitudinal patient world model" — is not novel. It is a described, published, and
benchmarked research area.**

Specifically:

- Active sensing on clinical time series: **Deep Sensing (2018), ASAC (2019)** — 7–8 years old.
- EIG-based acquisition motivated by ordering medical tests: **EDDI (2019)**.
- A *pipeline* combining prediction + uncertainty + information acquisition for medical time
  series: **Clairvoyance (ICLR 2021)** — this is close to our proposed system architecture.
- Longitudinal, cost-aware, non-greedy acquisition: **NOCTA (2025)**.
- A standardized AFA benchmark that already includes PhysioNet Challenge 2012: **AFABench (2025)**.
- Generating multiple plausible future patient timelines: **ETHOS (2024)**.

If we present Cliniverse as inventing this, a competent reviewer — and certainly a literature
search — will find the above within minutes. **We should not make that claim.**

### 3.2 What genuinely remains open

Four gaps survive the literature review. These are narrow, real, and achievable.

**(a) Panel-structured acquisition. — strongest remaining contribution.**
Every AFA method surveyed acquires **individual features at individual cost**. Clinical
laboratory medicine does not work that way: clinicians order **panels**. A basic metabolic
panel returns Na, K, HCO3, BUN, Creatinine, Glucose as *one billable, one-blood-draw event*.
A CBC returns WBC, HCT, Platelets. An ABG returns pH, PaO2, PaCO2.

This changes the optimization problem structurally — it becomes **set-valued acquisition with
shared cost and strong within-set correlation**, not per-feature greedy selection. Per-feature
EIG is the wrong objective under panel pricing, and per-feature benchmarks systematically
overstate achievable cost-efficiency because they let a policy buy exactly one analyte for
1/7th the price of the real-world minimum purchase. I could not find an AFA benchmark that
models this. **This is our primary claim to a contribution.**

**(b) Temporal AFA benchmark on open clinical data.**
AFABench includes PhysioNet 2012 but is explicitly static. A2MT is temporal but
non-clinical. NOCTA is longitudinal but is a method paper, not a public benchmark with a
released protocol. The intersection — *reproducible temporal + clinical + cost-aware AFA
benchmark on a dataset requiring no credentialing* — is open. That is TwinBench.

**(c) Calibration as a first-class acquisition objective and metric.**
The AFA literature reports accuracy/F1 vs cost. Whether acquisition improves *calibration and
conformal interval width* is largely unreported. H2 tests this.

**(d) Budget-coupled abstention.**
"Acquire more, or abstain?" as a joint decision, evaluated on risk-coverage curves, is a
natural but under-explored coupling of the selective-prediction and AFA literatures.

### 3.3 Honest contribution classification

| Category | Content |
|---|---|
| **Established technique (we implement, we do not claim)** | GRU-D, gradient boosting, deep ensembles, split/Mondrian conformal prediction, EDDI-style EIG, partial-VAE imputation, greedy discriminative acquisition, risk-coverage analysis. |
| **Our engineering contribution** | TwinBench: a reproducible, seeded, machine-readable temporal-AFA benchmark on fully open data, with an AFAPE-honest masking protocol; a panel-cost model; a unified evaluation harness. |
| **Our experimental contribution** | Empirical comparison of panel-level vs feature-level acquisition; H1/H2/H3 results including negative results; calibration-vs-budget curves; ablations. |
| **Speculative / experimental** | Learned non-myopic panel policy; abstention coupled to budget; anything involving MedGemma. |

---

## 4. Technical risks

### 4.1 RISK-1 (critical, methodological): retrospective AFA evaluation is biased

This is the risk most likely to invalidate the whole project, and the one the independent review should be
expected to find.

The AFAPE work (von Kleist et al., JMLR 26) formalizes it. Two assumptions are required to
identify real-world AFA performance from retrospective data:

- **NDE (No Direct Effect):** acquisitions do not change the underlying feature values.
- **NUC (No Unobserved Confounding):** historical acquisition decisions depended only on
  *observed* features.

**NUC is false in ICU data.** A clinician ordered a lactate because the patient looked septic —
using information (gestalt, exam findings, nursing concern) that is not in the dataset. Because
missingness is NMAR and informative (§2.5), a policy evaluated naively on retrospective data
gets credit for "discovering" that ordering a lactate is informative, when in truth *the
lactate exists in the record because the patient was already known to be sick.*

**Naive evaluation would produce impressive, meaningless numbers.** We will not do that.

**Mitigation — the masking-on-observed-support protocol (see §6.3 and TwinBench §13.3).**
We do not estimate what an AFA agent would achieve if deployed. We evaluate acquisition
policies under a **synthetic masking mechanism that we specify and control**. We take cells
that *were* observed, hide a seeded subset, and let policies buy them back. Ground truth for
every acquirable cell exists by construction, the mechanism is known by construction, and the
comparison between policies is therefore unbiased *with respect to that mechanism*.

The price of this honesty is a strictly limited claim, which we will state in the paper, the
README, and the UI:

> TwinBench measures **relative policy performance under a disclosed synthetic masking
> mechanism.** It does **not** estimate the clinical utility or real-world deployment value of
> any acquisition policy. Doing so would require the NUC/NDE assumptions of AFAPE, which do not
> hold in ICU data and which we do not claim.

Naturally-missing cells (never measured for that patient) are treated as **permanently
unavailable and never acquirable** — we cannot reveal a value we do not have.

### 4.2 Other risks

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| RISK-2 | **Effective sample size.** 12,000 patients, ~14% mortality → ~1,707 positives total. Budget-curve comparisons across ~8 policies × ~10 budget points invite false positives from multiple comparisons. | High | Pre-register metrics; bootstrap CIs on all curve points; paired tests on identical patients/seeds; report effect sizes not just p-values; hold out set-c until the end. |
| RISK-3 | **Baseline strength.** GBDT on summary features is very strong on P12. A deep model that loses to it is the expected outcome, not a bug. | High | Baselines first. Publish the GBDT number and beat it or concede. |
| RISK-4 | **Panel cost model is an assumption, not data.** P12 contains no prices. | Medium | Treat cost as a *declared, configurable, cited-where-possible* parameter. Run sensitivity analysis across cost schedules. Never present costs as real prices. |
| RISK-5 | **Scope collapse.** The brief lists ~9 modalities, 8 evaluation categories, a counterfactual sandbox, explainability, cloud deployment and a web app. | High | Milestone gating (§14). Modalities beyond structured time series are cut (§12). |
| RISK-6 | **Label leakage via SAPS-I/SOFA.** These are provided in the outcomes files and are computed from the same 48h window. Using them as *features* would leak; using them as *baselines* is legitimate. | Medium | Hard separation, enforced by a unit test. |
| RISK-7 | **Train/test contamination via imputation and scaling.** Fitting an imputer or scaler on all data before splitting is the classic error. | Medium | All preprocessing inside a fold-aware pipeline; test asserting imputer never sees test rows. |
| RISK-8 | **Python 3.14 toolchain.** Local interpreter is 3.14.4; torch does not yet ship Windows cp314 wheels. | Low | Pin project to **Python 3.12** via `uv`. (Verified: xgboost 3.4.0 and lightgbm 4.7.0 ship `py3-none-win_amd64`, so they are fine either way; torch is the binding constraint.) |
| RISK-9 | **GCP cost overrun.** A persistent Vertex AI GPU endpoint would exhaust $300 in days. | Medium | §11 — no persistent GPU endpoint. Budget alerts. |

---

## 5. Dataset options

### 5.1 Recommended primary: PhysioNet/CinC Challenge 2012 — **verified, executed**

I downloaded and analyzed this dataset rather than citing it. All numbers below were produced
by running code against the actual files.

**Access — verified by HTTP HEAD, 2026-08-09, no authentication:**

| File | Bytes |
|---|---|
| `Outcomes-a.txt` / `-b` / `-c` | 79,219 / 79,149 / 79,191 |
| `set-a.tar.gz` / `-b` / `-c` | 6,632,372 / 6,652,690 / 6,600,293 |

License: **Open Data Commons Attribution License v1.0**. Open to all. **No credentialing, no
CITI training, no DUA.** Total footprint ≈ 20 MB.

**Important finding:** outcomes are now published for **all three sets**, not just set-a as
during the original challenge. This gives us **12,000 labeled patients**, tripling the usable
data versus the commonly-cited 4,000.

**Verified label statistics:**

| Set | n | In-hospital deaths | Rate |
|---|---|---|---|
| set-a | 4,000 | 554 | 13.85% |
| set-b | 4,000 | 568 | 14.20% |
| set-c | 4,000 | 585 | 14.62% |

Outcome columns: `RecordID, SAPS-I, SOFA, Length_of_stay, Survival, In-hospital_death`.

**Verified structure (set-a, n=4,000):** long-format `Time,Parameter,Value` records; 36
time-series parameters plus 6 static descriptors (Age, Gender, Height, ICUType, Weight,
RecordID); `-1` sentinel for missing.

- Observation rows per record: median **392**, mean 402, min **0**, max 1,318
- Distinct timestamps per record: median **71**, mean 73.8, max 202
- On a naive (record × 48h × 36var) grid: 1,608,815 observation rows into 6,912,000 cells →
  occupancy **≤ 23.3%**, i.e. **≥ 76.7% missing** (upper bound on occupancy, since multiple
  rows can fall in one hour-cell).

**Verified data-quality edge cases:** **3 records have zero time-series observations**
(`140501`, `140936`, `141264`); 4 more have fewer than 20. These are genuine and must be
handled explicitly — they also make excellent robustness test fixtures.

**Verified acquisition structure — why this dataset fits the thesis.** Per-variable coverage
(fraction of records where the variable is *ever* measured) splits cleanly into a
continuously-monitored tier and a discretely-ordered tier:

| Tier | Variables | Coverage | Obs per covered record |
|---|---|---|---|
| Continuous monitoring (cheap, dense) | HR, Temp, GCS, Urine, NIxxxABP, xxxABP, MAP | 0.70–0.98 | 15–58 |
| Routine labs (BMP/CBC — discretely ordered) | Creatinine, BUN, HCT, Platelets, WBC, Na, HCO3, K, Mg, Glucose | 0.97–0.98 | **~3.3–3.6** |
| Blood gas (ABG) | pH, PaO2, PaCO2 | 0.76 | ~7.7–8.0 |
| Selective / expensive | Lactate 0.55, SaO2 0.45, LFTs (AST/ALT/Bilirubin/ALP) 0.42–0.43, Albumin 0.40, TroponinT 0.22, Cholesterol **0.076**, TroponinI **0.051** | 0.05–0.55 | 1.0–4.6 |

The routine labs being measured **~3.5 times across a 48-hour stay** is exactly the discrete,
countable, panel-shaped acquisition event our thesis needs. The rare tier (TroponinI at 5.1%,
Cholesterol at 7.6%) provides natural high-cost/low-frequency options. **This structure is why
P12 is the right dataset and not merely a convenient one.**

### 5.2 Secondary and rejected options

| Dataset | Access | Verdict |
|---|---|---|
| **Synthea** | Open, Apache-2.0 | **SECONDARY — controlled generator only.** Its generative process is hand-authored clinician-designed state machines. A model trained on it partly recovers the generator's rules, and information-gain results would be artifacts of those rules. Documented limitations include limited treatment variation and absence of real-world data-quality noise. Use it *only* where we want a known ground-truth generative mechanism: structured missingness, OOD shift, contradictory-observation cases. **Never as the primary evidence for H1–H3.** |
| **MIMIC-IV Clinical Database Demo** | Open (ODbL), 100 patients | **DEMO ONLY.** Far too small to train or evaluate. Useful solely if we want a FHIR/interop demonstration surface. |
| **MIMIC-IV (full)** | PhysioNet Credentialed license 1.5.0 + CITI "Data or Specimens Only Research" + signed DUA | **DO NOT DEPEND ON.** Credentialing is a multi-day-to-week human review process. Start the application in parallel as an option for later external validation; the project must be complete and defensible without it. |
| **eICU-CRD, HiRID, AmsterdamUMCdb** | Credentialed | Same conclusion as MIMIC-IV. Candidate external-validation sets if credentialing lands. |
| **EHRSHOT** | Research DUA required | Attractive (longitudinal, non-ICU, 15 tasks, released foundation model) but gated. Track as stretch. |

**Recommendation:** P12 primary (all 12,000 patients), Synthea secondary for controlled/OOD
cases, MIMIC-IV credentialing started in parallel but off the critical path.

---

## 6. Recommended prediction task

### 6.1 Rejected framings

- **"Predict the full future patient state."** Unfalsifiable and unevaluable at our scale.
- **Mortality alone at 48h.** A single binary at the end of the window gives no trajectory
  content and makes "world model" language indefensible.

### 6.2 Recommended task set

A **decision point at t = 24h** into the ICU stay. The model sees data from [0, 24h] (subject
to TwinBench masking), may acquire additional observations, then predicts:

| Task | Type | Target | Metrics |
|---|---|---|---|
| **T1 (primary)** In-hospital mortality | Binary | `In-hospital_death` | AUROC, AUPRC, Brier, NLL, ECE(equal-mass, debiased) |
| **T2** Prolonged stay (LOS > 3 days) | Binary | from `Length_of_stay` | same |
| **T3** Short-horizon trajectory forecast | Regression | value of selected vitals/labs in (24h, 48h] | MAE, RMSE, conformal coverage @90%, mean interval width |

T1 anchors us to a well-known literature baseline. T2 gives a second, less-imbalanced task so
conclusions do not rest on one label. T3 is what makes "trajectory"/"multiple futures" language
*earned* rather than decorative, and it is where uncertainty is most naturally evaluated.

The **24h decision point is deliberate**: it leaves a genuine 24h forward horizon for T3, and
it is the point at which "should I order another panel?" is a real question.

### 6.3 The acquisition problem, stated precisely

At t = 24h, given the masked view, a policy holds budget `B` (in cost units) and repeatedly
selects a **panel** from the catalogue (§13.3) to unmask within the observed support, until `B`
is exhausted or it elects to stop. Performance is the task metric as a function of spend.

---

## 7. Recommended model architectures

Strict ordering — **nothing sophisticated is built until the tier below it is beaten.**

| Tier | Model | Rationale | Classification |
|---|---|---|---|
| 0 | Prevalence / majority | Sanity floor | KEEP |
| 0 | **SAPS-I and SOFA** as single-feature predictors | Free in the outcomes files; genuine clinical severity-score baselines. Any model that cannot beat SOFA is not interesting. | KEEP |
| 1 | Logistic regression on last-value-carried-forward + **missingness indicators** + statics | Interpretable, fast, and a real contender | KEEP |
| 1 | **XGBoost / LightGBM** on per-variable summary features (count, mean, min, max, first, last, slope, time-since-last, measured-flag) | Expected strongest baseline. This is the number to beat. | KEEP |
| 2 | **GRU-D** (masking + learnable time decay) | Canonical irregular-time-series architecture; consumes missingness natively rather than imputing it away | KEEP |
| 3 | Temporal transformer with time encoding | Only if it beats tier 2. Recent evidence suggests RNNs remain competitive on irregular series. | EXPERIMENTAL |
| — | Diffusion trajectory models | No justification at this scale; large cost | **REMOVE** |
| — | Pretrained EHR foundation model (CLMBR/MOTOR) | Gated data, wrong vocabulary for P12 | **REMOVE** |

**Patient state Z(t).** Defined concretely as the hidden state of the tier-2 sequence encoder
at the decision point, conditioned on the observed mask. It is not a metaphysical claim — it is
a named tensor with a documented shape, and it is only meaningful to the extent it supports T1–T3.

**"Multiple plausible futures."** Implemented for T3 as a **calibrated predictive distribution**
via ensemble + conformal intervals, and optionally sampled trajectory rollouts. We will not
claim these are exhaustive or causally valid futures.

---

## 8. Recommended uncertainty methodology

| Component | Choice | Classification |
|---|---|---|
| Epistemic | **Deep ensemble, 5 seeds** — strongest simple baseline, trivially parallel | KEEP |
| Cheap comparator | MC dropout | EXPERIMENTAL (report only if it changes conclusions) |
| Distribution-free intervals | **Split conformal**, with **Mondrian/pattern-conditional variants** | KEEP |
| Missingness-aware conformal | **CP-MDA** (Zaffran et al., ICML 2023) | KEEP — directly required, see below |
| Bayesian NN / SWAG | — | **REMOVE** (cost ≫ value here) |

**Why CP-MDA is not optional.** Zaffran et al. show marginal conformal coverage survives
imputation, **but coverage conditional on the missingness pattern does not** — intervals
undercover for some patterns. Our entire benchmark deliberately varies the missingness pattern.
Reporting only marginal coverage would therefore hide exactly the failure mode our benchmark
is designed to expose. **We must report coverage stratified by missingness pattern and budget.**

**Calibration reporting rules (binding).** ECE is a biased, inconsistent, binning-sensitive
estimator (Nixon 2019; Kumar 2019; Roelofs 2022). Therefore:

1. **Brier score and NLL are primary** (proper scoring rules).
2. ECE is reported as *secondary*, always with equal-mass binning, bias correction, bootstrap CI,
   and an explicit note that it is biased.
3. **No claim rests on ECE alone.** Reliability diagrams accompany every ECE number.

**Abstention.** Selective prediction with a confidence threshold; evaluated by
**risk-coverage curves and AURC**, not by a single cherry-picked operating point.

---

## 9. Recommended active acquisition strategy

Baselines are mandatory and must be run before the proposed method.

| Policy | Type | Classification |
|---|---|---|
| Random panel | Baseline | KEEP |
| Most-missing-first | Baseline | KEEP |
| Static global importance (permutation importance on the GBDT) | Baseline | KEEP |
| Cost-normalized static importance (importance ÷ panel cost) | Baseline | KEEP |
| **Clinician-inspired fixed ordering** (BMP → CBC → ABG → targeted) | Baseline — **the one to genuinely fear** | KEEP |
| Max predictive-entropy / variance reduction (ensemble disagreement) | Heuristic | KEEP |
| **EDDI-style expected information gain over panels**, with a partial-VAE/conditional imputer | Proposed | KEEP |
| Learned greedy discriminative policy (GDFS/DIME-style, panel-adapted) | Proposed | EXPERIMENTAL |
| RL (non-myopic) | — | **DEFER.** A2MT reports SOTA agents failed to learn adaptive acquisition; AFABench finds non-myopic gains do not always justify cost. Not a hackathon-scale bet. |

**Evaluation.** Task metric at **each** budget point B ∈ {0, 1, 2, 3, 4, 5, 8, 10} panel-cost
units; **area under the budget-performance curve (AUBC)** as the headline scalar;
cost-normalized utility (Δmetric per cost unit); and — per H2 — **Brier/NLL/interval-width vs
budget**, not only AUROC vs budget.

**We are explicitly prepared to report that our proposed policy does not beat the clinical
heuristic.** That is a legitimate and publishable outcome, and pre-committing to it is what
keeps the experiment honest.

---

## 10. Recommended evaluation framework

```
evaluation/
  trajectory/     # T3 forecast error, interval coverage/width
  acquisition/    # budget curves, AUBC, cost-normalized utility, per-policy traces
  uncertainty/    # ensemble spread, entropy, risk-coverage, AURC
  calibration/    # Brier, NLL, reliability diagrams, debiased ECE, pattern-stratified coverage
  missingness/    # performance at 10/20/30/40/50/70% additional masking
  robustness/     # degenerate records, malformed input, corrupted/contradictory values
  ablations/      # see §14
  ood/            # Synthea-generated shift; ICU-type holdout; set-c as final holdout
```

**Statistical discipline (non-negotiable):**
- **set-c is locked** until final reporting. No tuning against it, ever.
- Bootstrap CIs (≥1,000 resamples) on every reported metric.
- Paired comparisons on identical patients and seeds.
- Every number traceable to a run ID, git SHA, config hash, and seed.
- `EXPERIMENTS.md` records executed runs only. **No planned or expected numbers, ever.**

---

## 11. GCP services actually needed

Verified pricing check: a Vertex AI endpoint bills **per node-hour from provision until
undeploy, including idle**. Reported rates: **A100 ≈ $2.93/GPU-hr (≈ $2,642/month if left up)**;
**L4 ≈ $0.70/GPU-hr (≈ $500/month)**; even a CPU `e2-standard-2` endpoint bills ≈ $0.077/hr
continuously. Against **$300 total credit**, a persistent GPU endpoint is disqualified outright.

| Service | Verdict | Reasoning |
|---|---|---|
| **Cloud Run** | **KEEP** | Scales to zero; the demo API costs ~$0 idle. The one service that genuinely fits. |
| **Cloud Storage** | **KEEP (minimal)** | Artifacts/results bucket. Cents at our data volume (~20 MB raw). |
| **Artifact Registry** | KEEP (minimal) | Required to deploy a container to Cloud Run. |
| **Vertex AI training** | **EXPERIMENTAL** | Only if local training becomes the bottleneck. Our full training set is 12,000 patients × 48h × 36 vars ≈ 20 MB — **this trains on a laptop.** Prefer local; use Vertex only for a parallel ensemble/hyperparameter sweep, with a hard cap. |
| **Vertex AI endpoints (persistent)** | **REMOVE** | Cost disqualifies it. See above. |
| **MedGemma deployment** | **REMOVE from core; EXPERIMENTAL offline at most** | Two independent reasons, either sufficient. **(1) Technical:** MedGemma is a medical *vision-language* model (4B/27B, Gemma-3 based, trained on X-ray, histopathology, dermatology, fundus, CT/MR, documents). Our data is irregular structured numeric time series. It contributes nothing to T1–T3. **(2) Cost:** see GPU rates above. Deploying it would be logo-driven architecture — precisely what the brief forbids. If we ever want natural-language rendering of a model explanation, run a small model locally and label it clearly as presentation-layer only. |
| **BigQuery** | **REMOVE** | Our dataset is 20 MB. This is a pandas problem, not a warehouse problem. |
| **Healthcare API / FHIR store** | **REMOVE** | Real cost and complexity for zero effect on H1–H3. P12 is not FHIR. |
| **Firestore** | **REMOVE** | No multi-user persistent state requirement. |

**Projected spend: < $15 for the entire project**, dominated by Cloud Run and egress.
Local execution is the default and every cloud path has a documented local fallback.
Billing alerts to be set at $25/$50/$100.

---

## 12. Components we should NOT build

| Component | Verdict | Reason |
|---|---|---|
| Clinical notes / NLP modality | **REMOVE** | P12 has no notes. Adding a modality with no data is fabrication. |
| Medical imaging modality | **REMOVE** | Same. |
| Wearables modality | **REMOVE** | Same. |
| Diagnoses/medications modality | **REMOVE** | Not in P12. Available only in credentialed datasets. |
| MedGemma in the core pipeline | **REMOVE** | §11. |
| BigQuery / Healthcare API / Firestore | **REMOVE** | §11. |
| RL acquisition policy | **DEFER** | §9. |
| Diffusion trajectory model | **REMOVE** | §7. |
| **LLM-generated explanations** | **REMOVE** | The brief explicitly forbids fake explanation. An LLM narrating a model it cannot introspect is exactly that. Use real attributions (§13.6). |
| Causal treatment-effect claims | **REMOVE** | P12 supports no causal identification. Counterfactuals are *model* counterfactuals only. |
| Symptom checker / diagnosis / treatment recommendation | **REMOVE** | Out of scope by the project charter and unsafe. |
| Production UI | **DEFER** | Per the brief — final design supplied later. |

**Net effect:** the proposed 9-modality multimodal system collapses to **one modality
(structured irregular clinical time series) that we can actually evaluate.** Per the brief's own
guidance, a smaller properly-evaluated system beats a fake universal one.

---

## 13. Revised architecture

### 13.1 Reframed project statement

> Cliniverse is a **reproducible benchmark and evaluation harness** for **cost-aware, panel-level,
> time-aware observation acquisition** under incomplete longitudinal patient data, together with
> calibrated-uncertainty baseline models. It measures whether choosing *what to measure next*
> can be done better than clinical and statistical heuristics, and whether doing so improves
> **calibration**, not just discrimination.

This is defensible, novel in its narrow way, and achievable. It drops the unfalsifiable
"world model" framing while keeping every technically real component.

### 13.2 Repository structure

```
cliniverse/
  cliniverse/
    data/            # P12 download, parse, hourly binning, splits, feature builders
    encoders/        # summary-feature builder; GRU-D; (later) transformer
    world_model/     # predictive heads T1/T2/T3, ensembles, trajectory rollout
    acquisition/     # panel catalogue, cost model, policies, budget loop
    uncertainty/     # ensembling, conformal (split/Mondrian/CP-MDA), abstention
    counterfactual/  # model-counterfactual perturbation API
    explainability/  # attributions, belief-change traces
    safety/          # claim guards, disclaimer surfaces, input validation
  twinbench/
    schemas/         # pydantic case schema + JSON Schema export
    generation/      # case construction from P12; Synthea path for OOD
    masking/         # MCAR / MAR / structured-block / panel-level mechanisms (seeded)
    corruption/      # noise, contradictions, malformed inputs
    datasets/        # generated case manifests (hashes, not blobs)
  experiments/       # baselines/, ablations/, robustness/
  evaluation/        # metric implementations + report generation
  apps/api/          # FastAPI service
  apps/web/          # minimal inspection UI (throwaway styling)
  tests/             # unit, integration, evaluation, leakage tests
  docs/              # this file, STATUS, DECISIONS, LIMITATIONS, EXPERIMENTS
  scripts/           # reproducible entrypoints
```

Modules are created **only when they contain working code** — no empty architecture theater.

### 13.3 TwinBench design

**Unit.** A *case* = one patient timeline T0…Tn + a masking specification + a budget + a
machine-readable ground-truth block.

**Construction pipeline (fully seeded, deterministic):**

1. **Parse** raw P12 records → long format → hourly-binned matrix `X[patient, hour, variable]`
   with an explicit observation mask `M`. Degenerate records (§5.1) are retained and flagged,
   not silently dropped.
2. **Split** by patient: set-a + set-b → train/val (with fixed CV folds); **set-c held out**.
   Splitting precedes every fit of every imputer, scaler and model.
3. **Observed support** `S` = cells where `M = 1`. Only `S` is ever acquirable.
4. **Apply masking mechanism** `m` with seed `s` to hide `S_hidden ⊂ S` — the "not yet ordered"
   set. Everything in `S_hidden` has known ground truth.
5. **Emit case** with: patient id, decision point, visible view, `S_hidden` index, budget,
   cost schedule, mechanism id, seed, and ground-truth targets.

**Case categories (per the brief, each with a defined mechanism):**

| Category | Mechanism |
|---|---|
| Complete records | No additional masking (baseline ceiling) |
| Randomly missing | MCAR at rates 10/20/30/40/50/70% over `S` |
| Structured missingness | Panel-level and contiguous-time-block masking (realistic: whole panels unordered, monitoring gaps) |
| Noisy observations | Gaussian/multiplicative noise at declared SNR on revealed values |
| Contradictory observations | Physiologically inconsistent pairs injected at declared rates (Synthea path preferred, where the generator's ground truth is known) |
| Stable / deteriorating / recovering / ambiguous trajectories | Stratified by a **declared, published rule** over observed trajectory slope + outcome — the rule ships in the repo and is not post-hoc |
| Distribution shift / OOD | ICU-type holdout (ICUType is a native field); Synthea-generated shift |

**Reproducibility contract.** Every case carries `(mechanism_id, seed, git_sha, config_hash)`.
The repo ships **manifests with content hashes**, not regenerated blobs, so any run is
bit-reproducible from the open source data.

**Panel catalogue and cost model.** A declared, configurable YAML schedule grouping the 36
variables into ordering units (BMP, CBC, LFT, ABG, coagulation, cardiac markers, individual
sends) with relative costs. **Costs are explicitly a modeling assumption, not real prices**
(RISK-4), and every headline result is accompanied by a sensitivity analysis over alternative
schedules.

### 13.4 Data flow

```
P12 raw  ->  parse/bin  ->  split (patient-level, set-c locked)
                                |
                         TwinBench case gen (mechanism, seed)
                                |
                    visible view + hidden support + budget
                                |
              encoder -> Z(t) -> ensemble heads (T1/T2/T3)
                                |
                    uncertainty (ensemble + conformal)
                                |
              acquisition policy: pick panel, spend, unmask, re-encode
                                |  (loop until budget exhausted / stop / abstain)
                                v
                    evaluation: budget curves + calibration + coverage
```

### 13.5 Counterfactual sandbox — narrowed

Supported and defensible: **model counterfactuals** only — remove an observation, alter a
revealed value, hold a variable constant, simulate sensor absence, shift observation timing;
then report how `Z(t)`, the predictive distribution, and uncertainty change.

Every such output is labeled **MODEL COUNTERFACTUAL** in the API response payload, not merely
in UI copy, so the label cannot be lost downstream. **No causal or treatment-effect claim.**

### 13.6 Explainability — real methods only

| Question | Method |
|---|---|
| What new evidence arrived? | Explicit diff of the revealed set (exact, not inferred) |
| Which features contributed? | SHAP on the GBDT; integrated gradients / attention on the neural encoder |
| How did uncertainty change? | Logged Δentropy, Δensemble-variance, Δinterval-width per acquisition step |
| Which trajectory probabilities changed? | Logged predictive distribution before/after each step |
| Why did the policy pick that panel? | The policy's own scores over the panel catalogue — for EIG and greedy policies this is a directly loggable quantity, not a post-hoc story |

The last row is the important one: because our policies compute an explicit score per candidate
panel, the explanation *is* the decision variable. Nothing is narrated by an LLM.

---

## 14. Implementation roadmap

Milestone gates: tests run, experiments executed, docs updated, committed, short report written.

| M | Deliverable | Exit criterion (measurable) |
|---|---|---|
| **M0** | Repo skeleton, `uv` env pinned to Py3.12, lint/format/type/test CI, P12 download+parse+bin, patient-level splits, `docs/{STATUS,DECISIONS,LIMITATIONS,EXPERIMENTS}.md` | `pytest` green; parser reproduces the verified §5.1 statistics exactly; **leakage test passes**; set-c untouched |
| **M1** | TwinBench v0: schemas, MCAR + panel-structured masking, case generation, manifests | Regenerating from a seed reproduces identical content hashes; JSON Schema published |
| **M2** | Baselines: prevalence, SAPS-I, SOFA, LR, XGBoost, GRU-D + full metric suite | Real AUROC/AUPRC/Brier with bootstrap CIs on val; GBDT ≥ SOFA; results in `EXPERIMENTS.md` with run IDs |
| **M3** | Uncertainty: 5-seed ensembles, split/Mondrian conformal, CP-MDA, abstention | Coverage within tolerance of nominal, **reported stratified by missingness pattern**; risk-coverage curves produced |
| **M4** | **Acquisition (core contribution)**: panel catalogue + cost model + all §9 policies + budget loop | Budget-performance curves with CIs for every policy; AUBC table; **H1/H2/H3 answered with real numbers, including negative results** |
| **M5** | Ablations, robustness sweep (10–70%), OOD, cost sensitivity | Each ablation either justifies a component or removes it |
| **M6** | Minimal API + inspection UI + Cloud Run deploy | Endpoints tested; disclaimers in payloads; spend < $15 |
| **M7** | Final: LIMITATIONS complete, set-c evaluated **once**, independent review responses | Every claim traceable to an executed run |

**Ablations planned (M5):** full model − longitudinal context (last value only); − active
acquisition (random at equal spend); − uncertainty component (point estimate); − missingness
indicators; − panel structure (per-feature acquisition at equivalent cost — **this ablation
directly tests contribution 3.2(a)**); − ensemble (single model).

---

## 15. Measurable success criteria

The project succeeds if these are **answered with executed numbers**, regardless of direction:

1. GBDT baseline beats SOFA on T1 AUROC on validation, with non-overlapping bootstrap CIs.
2. TwinBench regenerates bit-identically from seeds (hash equality test in CI).
3. Conformal coverage is within ±2% of nominal marginally, **and** pattern-stratified coverage
   is reported — including where it fails.
4. Budget-performance curves exist for ≥6 policies across ≥8 budget points with CIs.
5. **H1 answered:** EIG-over-panels either beats all four baselines on AUBC with non-overlapping
   CIs, or it does not — and we say which.
6. **H2 answered:** calibration-vs-budget curves reported alongside discrimination.
7. **H3 answered:** explicit comparison against the clinician-inspired heuristic, reported even
   if unfavourable.
8. Every number in `EXPERIMENTS.md` traceable to run ID + git SHA + config hash + seed.
9. Total GCP spend < $50.

---

## 16. Safety and claims posture

We will **not** claim clinical validation, diagnostic accuracy, hospital deployment, clinician
endorsement, patient-outcome improvement, or regulatory compliance. We have none of these.

We will **not** invent dataset sizes, metrics, patient counts, partnerships, or clinician testing.

All data is open-licensed (ODC-BY for P12; Apache-2.0 for Synthea) and used within license terms.

Every model output — API payload and UI — is labeled a **research simulation, not medical
advice**, and the AFAPE limitation of §4.1 is stated in the README, the paper, and
`docs/LIMITATIONS.md`.

---

## Appendix A — Component classification summary

| Component | Verdict |
|---|---|
| Longitudinal patient state Z(t) | **MODIFY** — concrete encoder hidden state, not a metaphysical claim |
| Multiple plausible futures | **MODIFY** — calibrated predictive distribution + ensemble rollouts |
| Uncertainty as first-class | **KEEP** |
| Active information acquisition | **KEEP** — but **REFRAME**: not novel (§3.1); our angle is panel-level + cost-aware + calibration-evaluated |
| Belief updating on new evidence | **KEEP** — this is just re-encoding after unmasking; do not oversell it |
| Counterfactual sandbox | **MODIFY** — model counterfactuals only, labeled in payload |
| Explainability | **MODIFY** — real attributions + logged policy scores; **no LLM narration** |
| TwinBench | **KEEP** — with the AFAPE-honest protocol of §4.1 |
| Multimodality (notes/images/wearables/meds) | **REMOVE** — no data |
| MedGemma / Vertex endpoints / BigQuery / Healthcare API / Firestore | **REMOVE** — §11 |
| RL acquisition | **DEFER** |
| Production UI | **DEFER** |

---

## Appendix B — Primary sources

**Active feature acquisition / active sensing**
- Ma et al., *EDDI: Efficient Dynamic Discovery of High-Value Information with Partial VAE*, ICML 2019 — https://proceedings.mlr.press/v97/ma19c.html
- Yoon, Jordon, van der Schaar, *ASAC: Active Sensing using Actor-Critic models*, MLHC 2019 — https://arxiv.org/abs/1906.06796
- Yoon, Zame, van der Schaar, *Deep Sensing*, ICLR 2018
- Jarrett et al., *Clairvoyance: A Pipeline Toolkit for Medical Time Series*, ICLR 2021 — https://github.com/vanderschaarlab/clairvoyance
- *Reinforcement Learning with Efficient Active Feature Acquisition*, arXiv 2011.00825
- *Active Acquisition for Multimodal Temporal Data (A2MT)*, arXiv 2211.05039
- *NOCTA: Non-Greedy Objective Cost-Tradeoff Acquisition for Longitudinal Data*, arXiv 2507.12412
- *Learning-To-Measure: In-Context Active Feature Acquisition*, arXiv 2510.12624
- *AFABench: A Generic Framework for Benchmarking Active Feature Acquisition*, arXiv 2508.14734

**Evaluation methodology (critical)**
- von Kleist et al., *Evaluation of Active Feature Acquisition Methods for Time-varying Feature Settings*, JMLR 26 — https://www.jmlr.org/papers/volume26/23-1635/23-1635.pdf
- *Evaluation of Active Feature Acquisition Methods for Static Feature Settings*, arXiv 2312.03619

**Uncertainty and calibration**
- Zaffran, Dieuleveut, Josse, Romano, *Conformal Prediction with Missing Values*, ICML 2023 — https://proceedings.mlr.press/v202/zaffran23a.html
- Lakshminarayanan et al., *Deep Ensembles*, NeurIPS 2017
- Roelofs et al., *Mitigating Bias in Calibration Error Estimation*; Nixon et al., *Measuring Calibration in Deep Learning*, 2019

**Longitudinal patient modeling**
- *Zero shot health trajectory prediction using transformer (ETHOS)*, npj Digital Medicine 2024 — https://www.nature.com/articles/s41746-024-01235-0
- Wornow et al., *EHRSHOT*, NeurIPS 2023 D&B — https://arxiv.org/abs/2307.02028

**Datasets**
- PhysioNet/CinC Challenge 2012 — https://physionet.org/content/challenge-2012/1.0.0/ (ODC-BY v1.0)
- MIMIC-IV v2.2 — https://physionet.org/content/mimiciv/2.2/ (credentialed)
- MIMIC-IV Clinical Database Demo v2.2 — https://physionet.org/content/mimic-iv-demo/2.2/ (ODbL)
- Walonoski et al., *Synthea*, JAMIA 25(3):230 — https://academic.oup.com/jamia/article/25/3/230/4098271

**Google Cloud**
- Vertex AI pricing — https://cloud.google.com/vertex-ai/pricing
- MedGemma model card — https://developers.google.com/health-ai-developer-foundations/medgemma/model-card
