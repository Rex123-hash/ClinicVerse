# Cliniverse — Project Handoff

**Written for a fresh assistant with zero conversation history.**
**Date:** 2026-08-11 · **HEAD at handoff:** `848c9030aa3553d7ec977ec2f7cdde3bc7955b39`
**Status:** M0–M4 complete. **M5 NOT STARTED and must be reconsidered before implementation.**

> **Update 2026-08-11 (post adversarial repair #4).** This document was first written at `ebb814a`, when M4
> was classified M4-B. Repair #4 then found a result-invalidating bug in the batched
> `fixed_domain_order` policy and reran M4. **M4 is now M4-C — largely stable / null.** Every M4
> number below is the repaired one. If you find an M4-B claim anywhere in this repository, it is
> stale: `docs/ADVERSARIAL_REVIEW_4.md` and `experiments/acquisition/results/m4/results.json` are the
> live truth.

---

## 0. Source-of-truth order — read in this order, always

1. **Repository code** — the definitive statement of what actually happens.
2. **Machine-readable artifacts** (`results.json`, `predictions.npz`) — the definitive numbers.
3. **Milestone / the reviewer reports** in `docs/`.
4. **`docs/STATUS.md`.**
5. **This handoff.**
6. Old conversation transcripts — **least authoritative**, may contain superseded claims.

If any two disagree, the earlier item wins. Several numbers in old transcripts were later
corrected; do not quote a transcript over an artifact.

---

## 1. What Cliniverse is, in plain English

ICU patients generate very incomplete records: a lab is measured only when someone decides to
measure it. Cliniverse asks what happens to a clinical prediction model when that information is
missing, degraded, or has to be bought back under a budget.

Concretely it does three things:

1. **Measures** how much of a mortality prediction comes from the *values* of tests versus the mere
   *pattern of which tests were recorded*.
2. **Stress-tests** what happens to the model's probabilities — not just its ranking — when
   coherent chunks of information disappear.
3. **Benchmarks** strategies for deciding what information to request next under a budget, and asks
   whether the "best" strategy stays best when the benchmark's assumptions change.

It is a **research/measurement project**, not a clinical product. It makes no diagnostic,
treatment, deployment or causal claims.

---

## 2. Technical architecture

```
cliniverse/
  config.py              typed frozen YAML schema loading (pydantic)
  log.py                 structlog setup (named log.py, NOT logging.py)
  exceptions.py          ConfigError, DataError, LeakageError, BudgetError, ...
  data/
    physionet2012.py     download / parse / hourly-bin PhysioNet 2012
    cohort.py            Cohort: x (n,T,V), m mask, statics, labels; invariant-checked
    splits.py            patient-level folds; set-c lock
  encoders/summary.py    availability / values / statics feature blocks (disjoint)
  evaluation/
    representations.py   mask_only / values_only / values_mask; FittedImputer
    metrics.py           AUROC, AP, Brier, NLL, calibration slope+intercept, paired bootstrap
    calibration.py       identity / Platt / isotonic, fitted on isolated calibration data
    selective.py         risk-coverage, AURC, predictive entropy
    information_loss.py  structured group / count-random / variable-matched loss
    artifacts.py         provenance (git SHA, dirty flag, cohort/split/config hashes)
  acquisition/
    catalogue.py         co-measurement groups + cost regimes
    policies.py          batched acquisition policies
    simulation.py        feature-space counterfactual completion for surrogate EIG
    evaluator.py         drives DisclosureEngine in lockstep; never reimplements it
twinbench/
  disclosure.py          DisclosureEngine, PolicyView, Protocol
  masking/mechanisms.py  group_hours, mcar_cells, time_blocks
  cases.py               CaseSpec / CaseManifest v2 with content hashing
  episode.py             per-patient episode runner + reference policies
experiments/{baselines,robustness,acquisition}/
tests/                   324 tests
docs/                    milestone reports, adversarial repairs, designs
```

**Key invariant:** the acquisition evaluator only *chooses actions and gathers visible state*. All
disclosure, costing, boundary enforcement and budget accounting happen inside the tested
`DisclosureEngine`.

---

## 3. Dataset, splits, cutoff

- **PhysioNet/CinC Challenge 2012.** Open (ODC-BY v1.0), **no credentialing**, ~20 MB.
- **12,000 labelled patients** across sets a/b/c (outcomes published for all three, not just set-a).
- Mortality: set-a 13.85%, set-b 14.20%, set-c 14.62%.
- **37 time-series variables.** `Weight` is longitudinal, not a static — misclassifying it caused a
  confirmed post-cutoff leak that is now fixed and regression-tested.
- Binned hourly-grid occupancy **20.25%** → **79.75% missing**. The raw row-count bound of 24.46%
  is a loose upper bound only (counts `-1` sentinels and within-hour collisions) and must never be
  quoted as the missingness statistic.
- **Cutoff: 24 hours.** Decision epochs at 12 / 18 / 24 h.
- **Development cohort = sets a+b, n = 8,000, prevalence 14.03%.**
- Task **T1 = in-hospital mortality.** T2 (LOS>3d) was dropped as arbitrary. T3 deferred.

### SET-C HOLDOUT LOCK — still locked

`load_cohort()` defaults to sets a+b. Loading set-c requires `allow_final_holdout=True`, and
`final_holdout()` additionally requires an explicit unlock token. **Set-c has never been used for
fitting or model selection.** Its aggregate outcome counts were read once during dataset
assessment; the accurate wording is *"quarantined from model fitting and model selection following
an aggregate cohort audit"*, **not** "never touched".

---

## 4. Support-blind replay semantics

Cliniverse performs **sequential selective disclosure / retrospective replay**. A policy requests
disclosure of eligible *already-recorded* historical information.

**This is NOT prospective clinical test ordering.** It cannot cause a test to happen that never
happened. Never use language implying test recommendation, ordering, intervention, or generating
future results.

| aspect | rule |
|---|---|
| action | requests one catalogue group, at timestamps ≤ current epoch boundary |
| overlaps | none — the catalogue partitions eligible variables |
| repeated action | permitted, charged in full, discloses nothing new |
| **unavailable request** | **charged in full, discloses nothing — no free probing** |
| beyond boundary | never disclosed until `advance_epoch()` |

**`support_blind`** = fair protocol; any action requestable.
**`support_aware`** = **DIAGNOSTIC AVAILABILITY ORACLE ONLY. Never deployable, never a headline.**

Before acting, a support-blind policy must never see: hidden support, `S_hidden`, hidden values,
future availability, hidden-value counts, availability-filtered candidate lists, the
natural-vs-synthetic distinction, or the label. Enforced by
`tests/test_policies.py::TestLeakageBoundary::test_action_unchanged_when_hidden_values_change`.

---

## 5. Calibration and leakage rules

**Three-way isolation per outer fold** (5-fold, patient-level, stratified):

| partition | n | fits imputer/scaler | fits model | fits calibrator | evaluated |
|---|---|---|---|---|---|
| model-train | 4,800 | yes | yes | no | no |
| calibration | 1,600 | no | no | yes | no |
| outer test | 1,600 | no | no | no | yes |

- Imputer fitted **once per fold on clean data** and **never refitted under stress or acquisition**.
- Calibrator fitted on the **clean** calibration partition; never recalibrated on test conditions.
- No outer-test information enters any fitted object.
- Every learned policy quantity (action frequencies, EIG quantiles) comes from **training folds only**.
- Model comparison uses **paired patient-level bootstrap**, never overlapping standalone CIs.

**Aggregate calibration slope/intercept computed on the same labels they are reported against are
DESCRIPTIVE ONLY** (this was an M2 finding). M3/M4 use the isolated design above.

---

## 6. Milestone history M0–M4

| milestone | outcome |
|---|---|
| **M0** | Repo, tooling, CI, P12 loader, splits, leakage guards. Found and fixed the `Weight` static-vs-longitudinal leak. |
| **M1** | TwinBench: disclosure engine, seeded masking, case manifests, episode runner. Hardened by adversarial repair #1. |
| **M2** | Representation ablation (mask vs values vs both). Corrected by adversarial repair #2 after a real nested-preprocessing defect. |
| **M3** | Calibration robustness under structured information loss. Reclassified **M3-B** by adversarial repair #3. |
| **M4** | Acquisition-policy ranking stability. Repaired by adversarial repair #4. Classified **M4-C**. |

### Verified commit SHAs

| SHA | meaning |
|---|---|
| `610f614` | research assessment, dataset verification, charter |
| `5dd0588` | M0 data layer, splits, tooling |
| `2d1a800` | independent review #0 response — novelty retraction + P0 leak fix |
| `29bdc75` / `6455696` | M1 disclosure engine / episode runner |
| `4bfed43` | adversarial repair #1 (TwinBench hardening) |
| `be9549c` | M2 representation ablation |
| `9628ff5` | **fix: correct M2 nested evaluation protocol** |
| `6bb9215` | adversarial repair #2 (M2 hardening) |
| `bff052a` | M3 calibration robustness |
| `a359ede` / `af07c2c` | adversarial repair #3 (M3 structured-loss audit + report) |
| `1f1ddc7` | **M4 design predeclared BEFORE any result** |
| `9357303` | M4 policy layer + leakage test |
| `e2c905c` | M4 evaluator (2 real bugs fixed) |
| `3528223` | degenerate-booster guard |
| `ebb814a` | M4 complete as M4-B — **superseded** |
| `abaeb44` | this handoff, first written at the M4-B state |
| **`848c903`** | **adversarial repair #4: fixed-order bug fixed, M4 rerun, M4-C — HEAD at handoff** |

---

## 7. Final corrected results

### M2 — representation ablation (n = 8,000, 24h, 5-fold)

| model / view | AUROC | AP | Brier | NLL |
|---|---|---|---|---|
| prevalence | 0.5000 | 0.1403 | 0.1206 | 0.4054 |
| LR mask-only | 0.7278 | 0.2783 | 0.1114 | 0.3657 |
| XGBoost mask-only | 0.7319 | 0.2812 | 0.1111 | 0.3634 |
| LR values-only | 0.8095 | 0.4273 | 0.0997 | 0.3276 |
| XGBoost values-only | 0.8279 | 0.4471 | 0.0970 | 0.3151 |
| LR values+mask | 0.8240 | 0.4511 | 0.0969 | 0.3182 |
| XGBoost values+mask | 0.8295 | 0.4502 | 0.0968 | 0.3141 |

**Corrected paired comparisons (XGBoost):**
- VALUES+MASK − VALUES-ONLY = **+0.0016 [−0.0028, +0.0059]** → **not distinguishable**
- VALUES-ONLY − MASK-ONLY = **+0.0960 [+0.0819, +0.1104]** → physiology dominates

**Verdict:** measurement-presence patterns are predictive on their own (~0.73 AUROC with no values),
but explicit mask features do not materially improve the strongest model.

### M3 — calibration under structured information loss (**M3-B, feature-identity effect**)

Highest severity, XGBoost values+mask, Platt:

| condition | AUROC | AP | Brier | NLL | intercept | slope | mean predicted risk |
|---|---|---|---|---|---|---|---|
| count-matched random cell | 0.8016 | 0.4018 | 0.1022 | 0.3345 | +0.115 | 0.926 | 0.1184 |
| variable-matched scattered | 0.8002 | 0.3990 | 0.1049 | 0.3445 | +0.573 | 1.023 | 0.0944 |
| structured group | 0.8002 | 0.3990 | 0.1049 | 0.3445 | +0.573 | 1.023 | 0.0944 |

Structured **minus** variable-matched: ΔNLL **0.0000 [0.0000, 0.0000]**, ΔBrier **0.0000**.

Because whole-window removal deletes *every* occurrence of a selected analyte, exact per-analyte
matching has zero degrees of freedom and the masks are bit-identical. The excess over count-random
is **analyte identity**, not a separable coherence effect.

**Supported M3 result:** discrimination degrades modestly while predicted risk drifts **downward**
(0.1397 → 0.0944 against 14.03% prevalence) — systematic **risk underestimation**. Platt improves
proper scores and slope but does **not** remove intercept drift. Isotonic was worse **under this
protocol only**. Demo record **142380**.

### M4 — acquisition-policy ranking stability (**M4-C**, repaired)

Primary condition (n = 8,000, support_blind / shared_plus_marginal / mask 0.6), **AUNLLC lower better**:

| rank | policy | AUNLLC |
|---|---|---|
| 1 | fixed_domain_order | **0.319414** |
| 2 | random_train_frequency | 0.320947 |
| 3 | random_uniform_all | 0.323660 |
| 4 | greedy_eig | 0.326586 |
| 5 | greedy_eig_per_cost | 0.326713 |
| 6 | no_acquisition | 0.326957 |

Paired `fixed_domain_order − random_train_frequency` = **−0.001533**, patient-level 1,000-replicate
percentile 95% CI **[−0.002814, −0.000251]**. The top two are statistically resolved.

Ranking stability (16 conditions, n = 2,000 fixed paired subsample):

| pair set | mean Kendall τ-b | min | supported reversals |
|---|---|---|---|
| **within support_blind (fair)** | **+0.776** | **+0.600** | **0 of 0 winner changes** |
| all 120 condition pairs (incl. oracle) | +0.557 | −0.200 | 0 |

`fixed_domain_order` wins **8 of 8** fair support-blind conditions. There are **zero** descriptive
fair-protocol winner changes across the 28 fair pairs, therefore zero supported fair reversals.
Repair #4 also tightened the label: `SUPPORTED REVERSAL` now requires **both** condition-specific
paired intervals to exclude zero, not just one. The predeclared one-condition flag is retained in the
artifact as `predeclared_one_condition_evidence` (25 across all pairs) for transparency only.

---

## 8. M4-C classification, and why M4-A and M4-B are NOT supported

**M4-C — LARGELY STABLE / NULL.**

Repair #4 found that `FixedOrderBatch.score_batch()` ignored step and history and returned the same
static priority at every step. Under support-blind replay `fixed_domain_order` therefore repeatedly
re-bought its first affordable group instead of advancing through the predeclared sequence,
contradicting the tested `twinbench.episode.FixedOrder` and producing non-monotone budget curves.
The repair added per-patient cursors. **No policy, order, mask rate, cost regime, budget, endpoint,
seed, sample, model or bootstrap choice changed** — but because the defect changed the primary
policy and the grid rankings, the entire run was necessarily repeated.

M4-A (strong ranking instability) and M4-B (moderate assumption sensitivity) are both **not
supported** after the repair:

1. Within the fair `support_blind` protocol, ordering is **stable**: mean τ-b **+0.776**, min
   **+0.600**, `fixed_domain_order` wins **8 of 8** conditions.
2. There are **zero** fair-protocol winner changes across all 28 fair condition pairs, so there is
   nothing left for M4-B to rest on. The single marginal reversal that carried M4-B was eliminated
   twice over: by the bug fix, and by the stricter both-intervals reversal definition.
3. Every remaining descriptive winner change involves the **`support_aware` availability oracle**,
   which is explicitly never deployable. Instability that only appears when a policy is handed an
   oracle is not evidence that benchmark design changes real scientific conclusions.
4. Across all 120 condition pairs, `n_supported_reversals` is **0**.

### Exact M4 headline (use this wording)

> Under support-blind retrospective disclosure replay on PhysioNet 2012, acquisition-policy ordering
> is stable to cost-regime and disclosure-rate assumptions: a fixed domain-motivated sequence ranks
> first in all eight predeclared fair conditions, mean Kendall τ-b +0.776, with no fair winner
> changes and no supported reversals. In the primary condition a surrogate expected-entropy-reduction
> heuristic has a much higher zero-new-cell request rate (94.59%) than a training-frequency random
> baseline (75.99%), disclosing 2.428 versus 11.556 new cells per patient.

That second sentence is **benchmark-specific and descriptive**. It does not establish that
availability is generally more valuable than information utility, and must never be written that way.

---

## 9. The EIG availability-failure result (most interesting finding)

At β = 1.0 in the primary condition:

| policy | mean cells disclosed | failed-request rate | mean requests | spend |
|---|---|---|---|---|
| **greedy_eig (surrogate)** | **2.428** | **94.59%** | 9.482 | 11.899 |
| greedy_eig_per_cost | 2.338 | 94.66% | 9.530 | 11.900 |
| **random_train_frequency** | **11.556** | **75.99%** | 8.348 | 11.898 |
| random_uniform_all | 8.783 | 80.21% | 9.662 | 11.839 |
| fixed_domain_order | 13.946 | 69.87% | 10.000 | 12.300 |

The `fixed_domain_order` row is the **repaired** one. Before repair #4 that policy re-bought its
first affordable group and appeared to disclose only ~5.1 cells at a 0.91 failure rate; corrected, it
discloses the most information of any policy, which is consistent with it also winning the ranking.

A failure is one paid action returning exactly **zero newly disclosed cells**. The denominator is
requests, not patients. A partially successful group counts as successful.

The surrogate EIG scores a group by how much its *simulated* completion would move the prediction,
which systematically favours rarely-measured, high-leverage analytes — exactly the ones least
likely to have hidden data available. Under support-blind replay it therefore burns most of its
budget on empty requests.

**In this benchmark condition, knowing what would be informative bought less disclosure than knowing
what is usually available.** State it as a benchmark observation, never as a general law.

Note: failure rates are 70–95% for *every* policy. Acquisition headroom is genuinely small (at mask
0.6 the `group_hours` mechanism leaves only ~12% of cells hidden by the final boundary). This was
retained as a result and **deliberately not engineered away**.

---

## 10. Rejected / superseded hypotheses — do not resurrect

| claim | status |
|---|---|
| Panel-level / shared-cost acquisition is our novel contribution | **RETRACTED.** Yu et al., ICLR 2023 (arXiv:2302.10261) already does sequential panel-level acquisition with shared group costs on MIMIC-IV. |
| Active acquisition on longitudinal patient models is novel | **RETRACTED.** Deep Sensing 2018, ASAC 2019, EDDI 2019, Clairvoyance 2021, NOCTA 2025. |
| A temporal clinical AFA benchmark is an open gap | **RETRACTED.** AFABench (2025) includes PhysioNet 2012. |
| Identifying availability-driven evaluation bias is our insight | **RETRACTED.** von Kleist et al., JMLR 26 (AFAPE). |
| Showing missingness is predictive is our finding | **RETRACTED.** Agniel 2018; JAMA Netw Open 2019 (AUROC ≈0.684 from indicators alone). |
| Calibration under acquisition is unexamined | **RETRACTED.** L2M 2025; MOSAIC 2026. |
| "Measurement shortcut drives prediction" (strong form) | **SUPERSEDED by M2.** Values dominate; explicit masks add +0.0016, not distinguishable. |
| XGBoost gains +0.0090 AUROC from explicit masks | **SUPERSEDED.** Corrected to +0.0016 [−0.0028, +0.0059]. |
| Median-vs-stochastic imputation gap quantifies missingness contribution (+0.0145) | **SUPERSEDED.** The empirical-marginal control is structurally incoherent (74.1% violation rate). |
| M3-A: "structure of loss causes excess degradation beyond amount" | **SUPERSEDED by M3-B.** Variable-matched control is mask-identical; ΔNLL exactly 0.0000. |
| "Calibration is perfect" (mask-only slope 1.000) | **SUPERSEDED.** Descriptive same-label regression only; fold slopes range 0.884–1.168. |
| M4-A strong ranking instability | **NOT SUPPORTED.** See §8. |
| M4-B moderate assumption sensitivity | **SUPERSEDED by M4-C** after repair #4. Zero fair winner changes remain. |
| Pre-repair M4 figures: fixed-order AUNLLC 0.32071, fair τ-b +0.743, 7/8 wins, 1 of 7 supported reversals | **SUPERSEDED.** Repaired: 0.319414, +0.776, 8/8, 0. Produced by the `FixedOrderBatch` static-priority bug. |
| Pre-repair `fixed_domain_order` spend figures (5.1 cells, 0.91 failure) | **SUPERSEDED.** Repaired: 13.946 cells, 69.87% failure. |
| E-003's "10.89 vs 0.62 / ~17× random collapse" | **SUPERSEDED.** Correct repaired figures: support-oracle random 11.33; support-blind uniform-all 4.58; support-blind training-frequency 7.32. |

---

## 11. Claim bans — binding

**Never say:**
- "first", "novel panel acquisition", "clinically costed panels", "prior AFA only buys individual features"
- "we discovered informative missingness"
- "structure itself causes the excess degradation" / "coherent grouping adds damage beyond analyte identity" / any M3-A phrasing
- "clinically realistic missingness" → use **"structured group-level information loss"**
- "the model doesn't know what it doesn't know" (entropy evidence is confounded: at 14% prevalence an entropy drop follows mechanically from predicted risks falling)
- "AI remains confidently wrong" without demonstrating *both* predictive deterioration *and* insufficient confidence response
- prospective test ordering / clinical recommendation / intervention / generating future results
- `support_aware` described as deployable
- `fixed_domain_order` described as clinician-derived → it is **domain-motivated**, engineering-authored
- `greedy_eig_per_cost` described as a reproduction of Yu et al. → it is a **closest justified analogue**
- "exact EIG" → it is a **greedy surrogate expected-information-gain** policy (3-point quadrature at training quantiles, completion simulated in feature space)
- any causal, deployment-utility, clinical-validation or clinician-intent claim
- that automated failure search exists (it does not — see §14)

Also: the `*-like` groups (`BMP_like`, `CBC_like`, `ABG_like`, `hepatic_like`) are **reconstructed
co-measurement feature groups**, **not verified clinical orders**.

---

## 12. Artifact locations

```
experiments/baselines/results/
  panel_derivation*.json                     E-001 co-measurement clustering
  availability_ablation.json / .oof.npz      E-002 mask-only
  m2/results.json, predictions.npz, figures/ E-004 representation ablation
experiments/robustness/results/m3/
  results.json, predictions.npz              E-005 calibration under loss
  m3_demo_patient.json                       record 142380
  figures/
experiments/acquisition/results/m4/
  results.json                               all 16 conditions + stability
  primary_predictions.npz                    n=8,000 primary condition
  grid_predictions.npz                       n=2,000 × 16 conditions
  figures/m4_primary_nll_vs_budget.png, m4_rank_flow.png
experiments/acquisition/m4_run.log
```

Every artifact carries git SHA, dirty flag, cohort fingerprint, split hash and config hash. M4
AUNLLC, rankings and Kendall τ-b were **independently recomputed from raw predictions with
0.000e+00 error**.

Reproduce:
```bash
python scripts/verify_physionet2012.py
python experiments/baselines/m2_representation_ablation.py --folds 5 --n-boot 2000
python experiments/robustness/m3_calibration_under_loss.py --n-boot 2000
python experiments/acquisition/m4_ranking_stability.py --scope both --folds 5 --n-boot 1000
```

---

## 13. Quality gate at handoff

| check | result |
|---|---|
| pytest | **324 collected — 307 passed, 17 slow deselected in fast mode** |
| ruff check | clean |
| ruff format --check | clean |
| mypy strict (`cliniverse twinbench`) | clean, 28 source files |
| `git diff --check` | clean |
| working tree | clean |

---

## 14. Current Best Overall assessment

**Strong scientific rigor, but no killer technical result yet.**

Strengths: predeclared designs committed before results; three independent adversarial adversarial audits
survived; leakage guards and support-oracle invariants enforced by tests; every headline
independently recomputable from raw predictions; superseded claims retracted in writing rather than
quietly dropped.

Weakness: the three completed results are a **dominance result** (M2: values dominate), a
**calibration-drift result** (M3-B), and a **null stability result** (M4-C). None is a single
striking failure a judge grasps in 20 seconds. The most novel finding is the EIG availability
failure (§9), but it is a single-dataset negative result.

**BEST OVERALL KILLER RESULT: NO.**

---

## 15. M5 — direction selected. See `docs/M5_DESIGN.md`.

> **Update 2026-08-11.** The literature review and feasibility assessment called for below were
> carried out, and **CANDIDATE DIRECTION A was selected and approved**. It is predeclared in
> `docs/M5_DESIGN.md` as **M5-A — Discrimination-Silent Reliability Failure Search**. Directions B
> and C were considered and **not** selected: B's motivating problem (the 94.59% failed-request rate)
> is an artifact of retrospective replay rather than of deployment, and its winning fix is already
> implemented as the `random_train_frequency` baseline; C is well covered by existing selective-
> prediction and conformal-under-missingness work, and M4 shows the recovery arm has little headroom.
> Useful parts of C survive inside M5-A's recovery arm. The candidate text below is retained
> unedited as the historical statement of the options.

**None of the following was implemented at the time of writing. All required literature review and
feasibility assessment before any commitment. Do not treat these as planned features.**

### CANDIDATE DIRECTION A — Automated adversarial reliability/failure search
Search over information-loss configurations for settings that **maximise probability/calibration
failure while ordinary discrimination stays deceptively strong**. Motivated by M3: AUROC fell only
0.827 → 0.800 while the calibration intercept moved −0.010 → +0.573. Requires: a defensible search
space, a failure objective that cannot be gamed, and a multiplicity/overfitting story.
**Not implemented. No automated failure search exists in the repository today.**

### CANDIDATE DIRECTION B — Availability-aware acquisition
Learn, support-blind and from training folds only, the **probability that a request succeeds**, and
combine expected information utility with expected availability and cost. Directly motivated by §9:
surrogate EIG wastes 95% of requests. Requires: a leakage-safe availability estimator (it must not
become a back-door support oracle), and prior-art review — this is close to existing cost-sensitive
AFA work and is unlikely to be novel.

### CANDIDATE DIRECTION C — Reliability-aware abstention / recovery
Detect when information loss has made a probability untrustworthy and **flag or abstain**, then test
whether targeted acquisition restores reliability. Builds on M3's risk-underestimation result and
the existing risk-coverage/AURC machinery. Requires: an abstention rule fitted only on calibration
data, and honest accounting of the coverage/accuracy trade-off.

---

## 16. Potential final product framing

**CLINIVERSE = a pre-deployment red-team / crash-test lab for healthcare AI.**

```
MODEL UNDER TEST
  → STRESS INFORMATION ENVIRONMENT
  → DISCOVER FAILURE
  → DIAGNOSE SENSITIVE INFORMATION
  → TEST RECOVERY
  → VERIFY RELIABILITY
  → RELIABILITY REPORT
```

Honest mapping to what exists **today**:

| stage | status |
|---|---|
| Model under test | **exists** (frozen M2/M3 XGBoost pipeline) |
| Stress information environment | **exists** (`information_loss.py`, `masking/`) |
| Discover failure | **manual only** — severities and conditions are predeclared by hand. **Automated failure search does NOT exist.** |
| Diagnose sensitive information | **partial** (analyte-identity analysis from M3 repair #3) |
| Test recovery | **partial** (M4 acquisition raises information but recovery is not framed or measured as recovery) |
| Verify reliability | **exists** (calibration slope/intercept, reliability curves, risk-coverage/AURC) |
| Reliability report | **not built** |

Do not present this framing as an implemented product.

---

## 17. Practical notes for the next assistant

- Python **3.12** via `uv` (torch has no Windows cp314 wheel). Run `uv sync --extra models --extra api --extra figures`.
- Windows + PowerShell. Multi-line strings to native exes break easily — write commit messages to a
  file and use `git commit -F`. Avoid `Get-Content -Raw`/`Out-File` round-trips on UTF-8 docs (they
  mojibake em-dashes).
- Commits are authored by **Amaan Khan**. **Do not add a AI-assistant co-author trailer.**
- **A degenerate-booster guard exists** (`FoldModel.n_features_used`). If a fold trains on too few
  rows, `min_child_weight=10` forbids all splits, the model becomes a constant, and every policy
  scores identically — which looks exactly like a propagation bug. The guard raises instead.
  **Do not lower the frozen hyperparameters to work around it; enlarge the cohort.**
- Long runs: the full M4 grid takes ~90 minutes, dominated by 71 winner-change pairs × 2 paired
  bootstraps × 1,000 replicates.
- **Do not work on:** frontend, GCP/deployment, Gemini, MedGemma, FHIR, auth, product screens or
  presentation assets. Science first.
