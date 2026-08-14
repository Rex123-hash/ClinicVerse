# Cliniverse the independent review Review + Repair #1

**Date:** 2026-08-09  
**Scope:** M1 audit, repair, and M2 safety gate. No substantive M2 model training was run.

# Executive Verdict

**PASS M1 WITH REMAINING NONBLOCKING RISKS.** M1's core support-blind disclosure design is
sound after repair. The review found no outcome-value path into policy state and no new
post-cutoff value leak. It did find one evaluator-object alias, ambiguous random baselines,
overstated event semantics, incomplete manifests, undefined runner termination, non-finite cost
edges, and a permissive set-c loading path. All were repaired with regression tests.

# Findings

## CR1-01

- **SEVERITY:** P1
- **PROBLEM:** `PolicyView.catalogue` referenced the evaluator's mutable Pydantic catalogue.
  A malicious policy could mutate its nested dictionaries. Policy arrays were copies but writable.
- **REPRODUCTION:** Object tracing showed `view.catalogue is engine._catalogue`; frozen Pydantic
  models do not freeze nested dictionaries.
- **FIX:** Added a separate immutable `ActionCatalogue`/`ActionSpec` policy type and made every
  policy-visible NumPy array read-only.
- **TEST:** Adversarial mutation tests assert detached types, frozen fields, read-only arrays, and
  unchanged evaluator disclosure state.
- **STATUS:** FIXED.

## CR1-02

- **SEVERITY:** P1
- **PROBLEM:** One policy named `random` changed meaning by protocol: it sampled an
  availability-filtered oracle list under `support_aware` and all groups under `support_blind`.
  Uniform-all was also a weak floor for rare groups. E-003's 10.89-versus-0.62 interpretation was
  therefore not a fair same-policy contrast.
- **REPRODUCTION:** Traced `RandomPolicy.select()` to `view.requestable`; that field is
  patient-support-filtered only in the aware protocol.
- **FIX:** Replaced it with `random_uniform_all`, `random_train_frequency`, and
  `random_support_oracle`. The frequency baseline counts active group-hours from an explicitly
  supplied training mask with smoothing; it receives no patient-specific support.
- **TEST:** Determinism, protocol constraints, naming, and held-out-support invariance are tested.
- **STATUS:** FIXED; E-003 rerun below.

## CR1-03

- **SEVERITY:** P1
- **PROBLEM:** `panel_events` was described as reconstructed co-measured clinical events, but it
  operated on hourly feature bins. Masks were drawn over the full cohort horizon even when the
  case cutoff was earlier. `time_blocks` skipped a partial final block.
- **REPRODUCTION:** The implementation defined an event as any hourly bin with one group member;
  `hidden_mask_for` passed the full mask; floor division omitted trailing hours.
- **FIX:** Renamed the mechanism `group_hours`, documented exact retrospective disclosure, limited
  mask generation to the case cutoff, and used ceiling block coverage. `uniform_event` became
  `uniform_group`.
- **TEST:** Tests cover partial groups, partial final blocks, rate endpoints, determinism,
  observed-only hiding, and absence of post-cutoff hidden cells.
- **STATUS:** FIXED.

## CR1-04

- **SEVERITY:** P1
- **PROBLEM:** Manifest v1 omitted `config_hash` from its content hash, left it empty, and did not
  capture catalogue content/version, explicit mask rate/seed, dataset content, or Git revision.
  Patient indices and declared sets were not verified during reconstruction.
- **REPRODUCTION:** Changing `config_hash` did not affect `content_hash`; a case could be rebuilt
  against a changed catalogue or mismatched record index.
- **FIX:** Manifest v2 hashes schema/config, cutoff-safe dataset fingerprint, Git SHA, catalogue
  version/content, explicit mechanism seed/rate, and all case fields. Set, cutoff/epoch, patient
  reference, schedule, and catalogue mismatches now fail closed.
- **TEST:** Round-trip/tamper, seed variation, provenance presence, set/cutoff mismatch,
  record-reference mismatch, catalogue mismatch, and no-patient-values invariants are tested.
- **STATUS:** FIXED.

## CR1-05

- **SEVERITY:** P1
- **PROBLEM:** The episode runner swallowed invalid actions, advanced epochs, and emitted no
  termination reason. A malformed unhashable action could crash; a repeating policy was retried
  each epoch.
- **REPRODUCTION:** Adversarial policies returned an unknown string, a list, or the same action
  forever.
- **FIX:** Added explicit termination reasons for policy stop, exhausted budget, request guard,
  malformed/unknown/unavailable/unaffordable actions. Invalid actions terminate once; the guard
  terminates the episode.
- **TEST:** Covers immediate stop/`None`, unknown and malformed actions, repetition, zero and huge
  budgets, unavailable and unaffordable choices, and bounded execution.
- **STATUS:** FIXED.

## CR1-06

- **SEVERITY:** P1
- **PROBLEM:** `load_cohort()` defaulted to all sets, including set-c, and
  `development_cohort()` silently dropped set-c after it had already been materialised.
- **REPRODUCTION:** Calling `load_cohort()` without `sets` selected `('a','b','c')`.
- **FIX:** Normal loading now defaults to a+b. Set-c requires explicit
  `allow_final_holdout=True`; development validation rejects rather than drops locked records.
  `final_holdout()` still requires its separate unlock token.
- **TEST:** Default-signature, explicit-load flag, fold rejection, and final token tests pass.
- **STATUS:** FIXED.

## CR1-07

- **SEVERITY:** P2
- **PROBLEM:** NaN budgets and masking rates passed comparison-based validation; infinite budgets
  and alternative-schedule costs were accepted. These could break budget monotonicity or random
  draws.
- **REPRODUCTION:** `budget=np.nan`, `rate=np.nan`, and an infinite alternative cost constructed
  successfully.
- **FIX:** Budgets/rates must be finite; action costs are finite and strictly positive. Zero
  budget and very large finite budgets remain valid. Repeated actions pay full cost and groups
  partition variables, so overlap semantics are explicit.
- **TEST:** Boundary table covers zero, negative, below/exact cost, huge finite, NaN, and infinity.
- **STATUS:** FIXED.

## CR1-08

- **SEVERITY:** P2
- **PROBLEM:** Live documents called hourly bins ordering events, described comparisons as
  unbiased, retained an ambiguous random result, and README still reported 36 variables.
- **REPRODUCTION:** Claim search across all requested live documents found each phrase.
- **FIX:** Narrowed wording to hourly feature-group co-presence and controlled paired comparisons;
  corrected README to 37 variables; documented support-aware as a diagnostic oracle and replaced
  stale E-003 semantics. Explicitly superseded historical material remains historical.
- **TEST:** Repository claim search plus production-parser statistic verification.
- **STATUS:** FIXED.

## CR1-09

- **SEVERITY:** P3
- **PROBLEM:** Support-blind post-purchase absence semantics were implemented but not stated
  precisely enough.
- **REPRODUCTION:** An empty request charged full cost and returned `n_disclosed=0`, which reveals
  acquired absence after payment.
- **FIX:** The specification now states that an empty paid result reveals no hidden value existed
  for that group within the current boundary; this acquired absence is part of the estimand.
- **TEST:** Indistinguishability before purchase and equal successful/empty costs remain enforced.
- **STATUS:** VERIFIED AND DOCUMENTED.

# Scientific Semantics Changed

- An action is retrospective selective disclosure of **all hidden recorded values for one
  panel-like feature group through the current hourly boundary**. It is not a clinical order or
  event.
- `support_aware` is an availability oracle used diagnostically.
- Random baselines have distinct information contracts; no result may label all three `random`.
- `group_hours` preserves binned co-presence only. It does not reconstruct specimens or orders.
- Paired cases control implemented randomness but are not described as statistically or causally
  unbiased.

# E-002 Verdict

**VALID.** The committed artifact was generated in the same repair-0 commit as the Weight fix and
contains 113 availability features (`37 × 3 + 2`). Feature construction runs after 24-hour
truncation; counts, ever flags, and recency use only that truncated mask. Five-fold patient splits
are disjoint, preprocessing is fitted inside each training fold, sets are a+b only, all 8,000
record IDs are unique, and the retained OOF predictions reproduce AUROC **0.7223745892**. The
result remains associational and is not interpreted causally. No rerun was required.

# E-003 Verdict

**ORIGINAL RANDOM ROWS SUPERSEDED; MECHANICS RERUN.** On the same 300 evaluation patients, budget
5, `group_hours@0.5#20260809`, and `shared_plus_marginal` costs:

| Policy | Protocol | Spent | Requests | Values disclosed | Empty | Wasted |
|---|---|---:|---:|---:|---:|---:|
| no_acquisition | support_aware | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| random_support_oracle | support_aware | 3.59 | 2.69 | 11.33 | 0.00 | 0.00 |
| fixed_order | support_aware | 2.84 | 2.02 | 10.82 | 0.00 | 0.00 |
| no_acquisition | support_blind | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| random_uniform_all | support_blind | 4.59 | 3.78 | 4.58 | 2.72 | 3.19 |
| random_train_frequency | support_blind | 4.46 | 3.45 | 7.32 | 1.96 | 2.44 |
| fixed_order | support_blind | 4.30 | 3.00 | 10.08 | 1.27 | 1.81 |

Training-frequency weights used 7,700 development patients and excluded all 300 evaluation
patients. This remains a mechanics check with no predictive model or clinical claim.

# Remaining Risks

- Every masking mechanism is synthetic; external validity to deployed acquisition is unresolved.
- The `*-like` groups depend on one dataset and one hourly binning/clustering choice.
- Support-aware results are oracle diagnostics and are easy to misuse if labels are removed.
- The fixed ordering is engineering-authored and not clinician-validated.
- No predictive acquisition comparison, uncertainty analysis, or external validation exists yet.

# M2 Contract

Predictive baselines must compare prevalence; mask-only LR; mask-only GBDT/XGBoost; values-only;
values+mask; and a strong XGBoost/LightGBM model. A sequential model is optional only if it beats
the strong tabular baseline. The three binding representations are **MASK ONLY**, **VALUES ONLY**,
and **VALUES + MASK**.

Acquisition evaluation must include no acquisition, `random_uniform_all`,
`random_train_frequency`, diagnostic `random_support_oracle`, a declared fixed ordering,
full-information ceiling, feature-level EIG, group-level EIG, a strong discriminative AFA
baseline, and a justified Yu et al./SM-DDPO reproduction or clearly documented closest analogue.
All train-derived quantities are fold-local; patients, masks, seeds, predictor, and budgets are
paired across protocol comparisons. Set-c remains locked until final evaluation.

# Quality Gate

- Full pytest: **214 passed in 19.50s** (Python 3.12.13)
- Ruff: **All checks passed**
- Ruff format: **43 files formatted / check clean**
- Mypy strict: **Success: no issues found in 32 source files**
- Reproducibility/adversarial checks: included in the full suite
- Production statistics: **37 variables**; set-a occupancy **0.2025306869** and missingness
  **0.7974693131** (20.25% / 79.75% rounded)
- E-002: artifact and OOF predictions independently rechecked
- E-003: mechanics rerun with repaired baselines

# Final Recommendation

**PASS M1 WITH REMAINING NONBLOCKING RISKS.** M2 may proceed under the contract above. Do not
promote the mechanics numbers to predictive or clinical results.
