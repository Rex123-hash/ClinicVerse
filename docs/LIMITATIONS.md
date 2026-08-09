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
that we specify, seed, and disclose**. Paired cases control the implemented randomness, but that
does not make policy comparisons statistically or causally unbiased: estimates remain sensitive
to policy selection, model fitting, finite samples, and the chosen mechanism. They do not
transfer to a claim about clinical utility.

A policy that wins on TwinBench has not been shown to be useful in a hospital.

## 3. Dataset limitations

- **Population.** PhysioNet/CinC 2012 is adult ICU patients with stays ≥48h, from a limited set
  of units. Results do not generalize to non-ICU, paediatric, or outpatient populations.
- **Vintage.** The data reflects historical ICU practice, not current standards of care.
- **Scale.** 12,000 patients, 1,707 in-hospital deaths total. Confidence intervals on
  subgroup and per-budget-point metrics will be wide, and we report them rather than hiding them.
- **Multiple comparisons.** Comparing ~8 policies across ~10 budget points invites false
  positives. Mitigated by pre-registered metrics, paired tests on identical patients/seeds, and
  bootstrap CIs — but not eliminated.
- **Degenerate records.** 3 records in set-a have zero valid time-series observations (they contain only `Weight,-1`, the missing sentinel); several more
  have fewer than 20. These are retained and flagged, not silently dropped.
- **Naturally-missing cells are unrecoverable.** If a variable was never measured for a patient,
  no disclosure can produce it. Under the support-blind protocol a policy may still *request*
  it and is charged in full for nothing — which is the point — but the value cannot appear.
  The historical support is itself a biased sample of what could have been measured.

## 4. Cost model limitations

PhysioNet 2012 contains no prices. The panel cost schedule is a **declared modeling assumption**,
not real billing data. Costs are configurable and every headline result is accompanied by
sensitivity analysis across alternative schedules. **Cost figures must never be presented as
real prices or as economic findings.**

## 5. Panel definitions are a simplification

PhysioNet 2012 records analytes with timestamps; it does **not** record what was ordered.
Our groups (`BMP_like`, `CBC_like`, `hepatic_like`, `ABG_like`) are hourly-bin co-presence
clusters that resemble familiar panels, not recovered laboratory orders, specimens, or events.
An action retrospectively discloses all hidden recorded values for that feature group through the
current boundary. It does not simulate a prospective order. The groups are named `*-like`
throughout and must be described that way.

The `support_aware` protocol and `random_support_oracle` baseline receive patient-specific
availability and are diagnostic oracles, not deployable comparators. `random_uniform_all` is an
honest support-blind floor but can be weak for rare groups; `random_train_frequency` therefore
provides a stronger support-blind baseline fitted on training support only.

## 6. Uncertainty limitations

- Conformal prediction gives **marginal** coverage guarantees. Coverage *conditional on
  missingness pattern* is known to degrade (Zaffran et al., ICML 2023) — precisely the regime
  this benchmark varies. We report pattern-stratified coverage, including where it fails.
- **ECE is a biased, inconsistent, binning-sensitive estimator** (Nixon 2019; Kumar 2019;
  Roelofs 2022). It is reported as secondary only, with equal-mass binning, bias correction and
  bootstrap CIs. No conclusion rests on ECE alone.
- **Brier and log-loss are proper scoring rules, not pure calibration metrics.** They combine
  calibration and refinement, so a better Brier does not by itself demonstrate better
  calibration. Calibration slope and intercept, plus reliability curves, are the direct readouts.
- **Conformal scope.** Split conformal is applied to the T3 scalar regression only. CP-MDA is a
  regression method for missing covariates and is **not** applied to T1 classification.
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
