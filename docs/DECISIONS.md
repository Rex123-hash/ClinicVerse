# Cliniverse — Architecture Decisions

Each entry records the decision, the alternatives rejected, and why. Decisions are amendable;
supersede rather than silently edit.

---

## D-001 — Reframe the project's claimed contribution
**Status:** PROPOSED — awaiting project-owner sign-off
**Date:** 2026-08-09

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

**Why.** Verified open access (ODC-BY, no credentialing, ~20 MB); natively irregular and
≥76.7% missing; and — decisively — its variable-coverage structure is genuinely panel-shaped:
routine labs are measured ~3.5× per 48h stay while vitals are near-continuous, giving real
discrete acquisition events. Established literature baselines exist for the mortality task.

**Alternatives rejected.**
- *MIMIC-IV full.* Credentialing is multi-day human review; cannot gate the project on it.
- *Synthea as primary.* Its data comes from hand-authored clinician-designed state machines;
  information-gain results would be artifacts of the generator's rules.
- *MIMIC-IV demo.* 100 patients — unusable for training or evaluation.

---

## D-003 — Masking-on-observed-support evaluation protocol
**Status:** ACCEPTED
**Date:** 2026-08-09

**Context.** AFAPE (von Kleist et al., JMLR 26) shows estimating deployed AFA performance from
retrospective data requires No-Direct-Effect and No-Unobserved-Confounding. **NUC is false in
ICU data**: a lactate exists in the record because the patient already looked septic, using
information absent from the dataset. Missingness is NMAR and informative.

**Decision.** Do not estimate deployment value. Evaluate policies under a **synthetic masking
mechanism we specify and control**: hide a seeded subset of *observed* cells; policies buy them
back. Ground truth exists by construction, the mechanism is known by construction, so
policy comparison is unbiased with respect to that mechanism. Naturally-missing cells are
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
