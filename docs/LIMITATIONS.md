# Cliniverse — Limitations

This file is written *before* results exist, so that limitations constrain the work rather than
being retrofitted to excuse it. It is updated as milestones land.

---

## 1. This is research software, not a medical device

Cliniverse produces **research/model simulations**. It is not medical advice, not a diagnostic
tool, and not a treatment recommender. We claim **no** clinical validation, **no** diagnostic
accuracy in real healthcare settings, **no** hospital deployment, **no** clinician endorsement,
**no** patient-outcome improvement, and **no** regulatory compliance of any kind.

## 2. The central evaluation limitation (most important)

**We do not — and cannot — estimate the real-world clinical value of any acquisition policy.**

Estimating deployed active-feature-acquisition performance from retrospective data requires the
No-Direct-Effect and No-Unobserved-Confounding assumptions formalized in the AFAPE literature
(von Kleist et al., JMLR 26). **NUC is false in ICU data**: a lab exists in the record because a
clinician was already concerned, using information (exam findings, gestalt, nursing report) that
is not in the dataset. Missingness is Not-Missing-At-Random and is itself strongly predictive.

TwinBench therefore measures **relative policy performance under a synthetic masking mechanism
that we specify, seed, and disclose**. Comparisons between policies are unbiased *with respect
to that mechanism*. They do not transfer to a claim about clinical utility.

A policy that wins on TwinBench has not been shown to be useful in a hospital.

## 3. Dataset limitations

- **Population.** PhysioNet/CinC 2012 is adult ICU patients with stays ≥48h, from a limited set
  of units. Results do not generalize to non-ICU, paediatric, or outpatient populations.
- **Vintage.** The data reflects historical ICU practice, not current standards of care.
- **Scale.** 12,000 patients, ~1,707 in-hospital deaths total. Confidence intervals on
  subgroup and per-budget-point metrics will be wide, and we report them rather than hiding them.
- **Multiple comparisons.** Comparing ~8 policies across ~10 budget points invites false
  positives. Mitigated by pre-registered metrics, paired tests on identical patients/seeds, and
  bootstrap CIs — but not eliminated.
- **Degenerate records.** 3 records in set-a have zero time-series observations; several more
  have fewer than 20. These are retained and flagged, not silently dropped.
- **Naturally-missing cells are unrecoverable.** If a variable was never measured for a patient,
  no policy can acquire it. Our acquirable set is strictly the observed support, which is itself
  a biased sample of what could have been measured.

## 4. Cost model limitations

PhysioNet 2012 contains no prices. The panel cost schedule is a **declared modeling assumption**,
not real billing data. Costs are configurable and every headline result is accompanied by
sensitivity analysis across alternative schedules. **Cost figures must never be presented as
real prices or as economic findings.**

## 5. Panel definitions are a simplification

Our panel catalogue (BMP, CBC, LFT, ABG, …) approximates real ordering practice. Real ordering
varies by institution, and some analytes appear in multiple panels. The catalogue is documented
and configurable, but it is a model of practice, not practice itself.

## 6. Uncertainty limitations

- Conformal prediction gives **marginal** coverage guarantees. Coverage *conditional on
  missingness pattern* is known to degrade (Zaffran et al., ICML 2023) — precisely the regime
  this benchmark varies. We report pattern-stratified coverage, including where it fails.
- **ECE is a biased, inconsistent, binning-sensitive estimator** (Nixon 2019; Kumar 2019;
  Roelofs 2022). It is reported as secondary only, with equal-mass binning, bias correction and
  bootstrap CIs. No conclusion rests on ECE alone; Brier and NLL are primary.
- Deep ensembles approximate epistemic uncertainty; they are not Bayesian posteriors.

## 7. Counterfactual limitations

Cliniverse supports **model counterfactuals only**: how the model's output changes when inputs
are perturbed. These are **not** causal effects, **not** treatment effects, and carry no
identification guarantees. This dataset supports no causal identification strategy. Every such
output is labeled `MODEL COUNTERFACTUAL` in the API payload itself, not only in UI copy.

## 8. Explainability limitations

SHAP and gradient-based attributions describe **model behaviour**, not clinical causation or
ground truth. Attribution methods are known to be unstable under correlated features — and
clinical variables are heavily correlated. Acquisition-policy explanations are the policy's own
logged scores, which is honest for our policies but does not generalize to methods whose
decision variable is not directly inspectable.

## 9. Synthea limitations

Synthea data is produced by hand-authored, clinician-designed state machines. Models trained on
it partly recover the generator's rules, and it lacks the transcription errors, local coding
conventions and data-quality noise of real EHR. It is used **only** for controlled
structured-missingness and OOD cases where a known ground-truth mechanism is the point — never
as primary evidence.

## 10. Scope limitations

Single modality (structured irregular time series). No clinical notes, imaging, wearables,
medications, or diagnoses — those modalities are absent from the dataset. No RL acquisition
policy in the current scope. No production UI.
