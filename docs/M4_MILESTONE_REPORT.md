# M4 Milestone Report — Acquisition-Policy Ranking Stability

**Date:** 2026-08-11
**Design:** predeclared in [`M4_DESIGN.md`](M4_DESIGN.md), committed at `1f1ddc7` before any result.
**Classification:** **M4-B — MODERATE ASSUMPTION SENSITIVITY**
**Recommendation:** **MODIFY** for M5 (§20)

Every number is read from `experiments/acquisition/results/m4/results.json` and was
**independently recomputed from the raw prediction arrays** (§2). Nothing is typed by hand.

---

## 1. Run status

The background run launched at the end of the previous session **completed successfully** in
~90 minutes. No restart was needed and no duplicate run was launched. Primary condition finished
first, then all 16 grid conditions, then the ranking-stability bootstrap (71 winner-change pairs ×
2 paired bootstraps × 1,000 replicates — this dominated the runtime).

## 2. Artifact integrity — all checks pass

| check | result |
|---|---|
| AUNLLC recomputed from raw predictions | **max error 0.000e+00** |
| Rankings recomputed from AUNLLC | **identical** |
| Kendall tau-b recomputed independently | **mean 0.5689, min −0.0667 — matches artifact exactly** |
| β=0 matches no-acquisition in every condition | **yes** |
| no-acquisition budget-invariant in every condition | **yes** |
| Policies / protocols / cost regimes / mask rates / budget grid | 6 / 2 / 4 / 2 / 8 — all present |
| Conditions × policies × budgets in NPZ | 16 × 6 × 8 = 768 arrays (+labels, +record ids) |
| Non-finite predictions | **0** |
| Sets used | `[a, b]` — **set-c never loaded** |

**Provenance caveat, recorded honestly:** the artifact carries `git_dirty=true`. The *source* was
the committed revision `3528223`; the working tree additionally held the untracked run log and the
results directory. No tracked source file differed from `3528223` at run time.

## 3. Semantics, policies, scope

**Evaluator.** Every state transition goes through the tested `DisclosureEngine`. Sequential
selective disclosure / retrospective replay — **not** prospective test ordering. Unavailable
requests are charged in full and disclose nothing.

**Policies (ranked):** `no_acquisition`, `random_uniform_all`, `random_train_frequency`,
`fixed_domain_order`, `greedy_eig`, `greedy_eig_per_cost`.

**Adaptive-policy approximation (exact wording).** `greedy_eig` is a **greedy surrogate
expected-information-gain** policy. For each action it integrates over a *predicted* distribution
for the unknown value using a fixed 3-point quadrature at the **training-fold** 25th/50th/75th
percentiles, with the completion simulated in feature space (one extra observation at the current
boundary; `n_obs`, `ever`, `recency`, `last`, `mean`, `min`, `max` updated consistently). It is an
approximation, **not exact EIG**. No test label and no hidden value enters the choice — enforced by
`test_action_unchanged_when_hidden_values_change`. `greedy_eig_per_cost` divides by declared cost
and is a **closest justified analogue** to a cost-sensitive greedy comparator — **not** a
reproduction of Yu et al.

**Scope:** primary condition **n = 8,000** (all development patients); stability grid
**n = 2,000** (fixed seeded paired subsample, identical across all 16 conditions). 5 folds.

**Protocols:** `support_blind` (fair, primary) and `support_aware` (**diagnostic oracle only —
never deployable**). **Cost regimes:** `shared_plus_marginal`, `uniform_group`, `ordinal_tier`,
`per_analyte`. **Disclosure rates:** 0.3, 0.6. **Budget grid:** 0, 0.10, 0.20, 0.30, 0.40, 0.50,
0.75, 1.00 as a fraction of total catalogue cost.

## 4. Primary condition — AUNLLC (n = 8,000, support_blind / shared_plus_marginal / mask 0.6)

| rank | policy | AUNLLC (lower better) |
|---|---|---|
| 1 | **fixed_domain_order** | **0.32071** |
| 2 | random_train_frequency | 0.32095 |
| 3 | random_uniform_all | 0.32366 |
| 4 | greedy_eig | 0.32659 |
| 5 | greedy_eig_per_cost | 0.32671 |
| 6 | no_acquisition | 0.32696 |

### NLL vs budget

| policy | 0.0 | 0.10 | 0.20 | 0.30 | 0.40 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|---|---|---|
| no_acquisition | 0.3270 | 0.3270 | 0.3270 | 0.3270 | 0.3270 | 0.3270 | 0.3270 | 0.3270 |
| random_uniform_all | 0.3270 | 0.3269 | 0.3253 | 0.3247 | 0.3243 | 0.3231 | 0.3218 | 0.3216 |
| random_train_frequency | 0.3270 | 0.3268 | 0.3234 | 0.3222 | 0.3207 | 0.3193 | 0.3185 | 0.3183 |
| fixed_domain_order | 0.3270 | 0.3266 | 0.3197 | 0.3197 | 0.3191 | 0.3197 | 0.3197 | 0.3197 |
| greedy_eig | 0.3270 | 0.3270 | 0.3265 | 0.3265 | 0.3264 | 0.3265 | 0.3265 | 0.3266 |
| greedy_eig_per_cost | 0.3270 | 0.3270 | 0.3267 | 0.3267 | 0.3265 | 0.3267 | 0.3267 | 0.3267 |

### Secondary metrics at β = 1.0

| policy | Brier | AUROC | AP | cal. intercept | cal. slope |
|---|---|---|---|---|---|
| no_acquisition | 0.1003 | 0.8109 | 0.4159 | −0.013 | 0.928 |
| random_uniform_all | 0.0987 | 0.8174 | 0.4297 | −0.046 | 0.951 |
| **random_train_frequency** | **0.0977** | **0.8220** | **0.4413** | −0.048 | 0.967 |
| fixed_domain_order | 0.0982 | 0.8208 | 0.4366 | −0.054 | 0.960 |
| greedy_eig | 0.1002 | 0.8125 | 0.4214 | +0.013 | 0.922 |
| greedy_eig_per_cost | 0.1003 | 0.8125 | 0.4201 | +0.016 | 0.923 |

Note the primary and secondary endpoints disagree at the top: `fixed_domain_order` wins on AUNLLC,
`random_train_frequency` wins on Brier/AUROC/AP. The margin between them is small.

### Realized spend, disclosure and failure at β = 1.0

| policy | mean spend | mean cells disclosed | mean requests | failed-request rate |
|---|---|---|---|---|
| no_acquisition | 0.00 | 0.0 | 0.0 | — |
| random_uniform_all | 11.84 | 8.8 | 9.7 | 0.80 |
| random_train_frequency | 11.90 | **11.6** | 8.3 | **0.76** |
| fixed_domain_order | 11.90 | 5.1 | 7.0 | 0.91 |
| greedy_eig | 11.90 | **2.4** | 9.5 | **0.95** |
| greedy_eig_per_cost | 11.90 | 2.3 | 9.5 | 0.95 |

**Failure rates are high (76–95%) across every policy.** This is the headroom limitation flagged
earlier and it is retained as a result, not engineered away.

## 5. Rankings for every condition (n = 2,000)

| condition | ranking (best → worst by AUNLLC) |
|---|---|
| support_blind \| shared_plus_marginal \| 0.3 | fixed_domain_order > random_train_frequency > random_uniform_all > no_acquisition > greedy_eig_per_cost > greedy_eig |
| support_blind \| shared_plus_marginal \| 0.6 | fixed_domain_order > random_train_frequency > random_uniform_all > greedy_eig > greedy_eig_per_cost > no_acquisition |
| support_blind \| uniform_group \| 0.3 | fixed_domain_order > random_train_frequency > random_uniform_all > no_acquisition > greedy_eig > greedy_eig_per_cost |
| support_blind \| uniform_group \| 0.6 | fixed_domain_order > random_train_frequency > random_uniform_all > greedy_eig > greedy_eig_per_cost > no_acquisition |
| support_blind \| ordinal_tier \| 0.3 | **random_train_frequency** > fixed_domain_order > random_uniform_all > no_acquisition > greedy_eig_per_cost > greedy_eig |
| support_blind \| ordinal_tier \| 0.6 | fixed_domain_order > random_train_frequency > random_uniform_all > greedy_eig_per_cost > greedy_eig > no_acquisition |
| support_blind \| per_analyte \| 0.3 | fixed_domain_order > random_train_frequency > random_uniform_all > no_acquisition > greedy_eig > greedy_eig_per_cost |
| support_blind \| per_analyte \| 0.6 | fixed_domain_order > random_train_frequency > random_uniform_all > greedy_eig > greedy_eig_per_cost > no_acquisition |
| support_aware \| shared_plus_marginal \| 0.3 | **random_train_frequency** > random_uniform_all > greedy_eig_per_cost > greedy_eig > fixed_domain_order > no_acquisition |
| support_aware \| shared_plus_marginal \| 0.6 | fixed_domain_order > random_train_frequency > random_uniform_all > greedy_eig_per_cost > greedy_eig > no_acquisition |
| support_aware \| uniform_group \| 0.3 | **random_uniform_all** > random_train_frequency > fixed_domain_order > greedy_eig > greedy_eig_per_cost > no_acquisition |
| support_aware \| uniform_group \| 0.6 | fixed_domain_order > random_train_frequency > random_uniform_all > greedy_eig > greedy_eig_per_cost > no_acquisition |
| support_aware \| ordinal_tier \| 0.3 | **random_train_frequency** > random_uniform_all > fixed_domain_order > greedy_eig_per_cost > greedy_eig > no_acquisition |
| support_aware \| ordinal_tier \| 0.6 | fixed_domain_order > random_train_frequency > random_uniform_all > greedy_eig_per_cost > greedy_eig > no_acquisition |
| support_aware \| per_analyte \| 0.3 | **random_uniform_all** > random_train_frequency > greedy_eig > greedy_eig_per_cost > fixed_domain_order > no_acquisition |
| support_aware \| per_analyte \| 0.6 | **greedy_eig_per_cost** > fixed_domain_order > random_train_frequency > random_uniform_all > greedy_eig > no_acquisition |

Winner tally across 16 conditions: `fixed_domain_order` 10, `random_train_frequency` 3,
`random_uniform_all` 2, `greedy_eig_per_cost` 1.

## 6. Kendall tau-b and winner changes — the decisive split

| pair set | pairs | mean τ-b | min τ-b |
|---|---|---|---|
| **within `support_blind` (fair)** | 28 | **+0.743** | **+0.467** |
| within `support_aware` (oracle) | 28 | +0.524 | +0.067 |
| across protocols | 64 | +0.512 | **−0.067** |
| all pairs | 120 | +0.569 | −0.067 |

Winner changes by locus, with the predeclared classification:

| locus | SUPPORTED REVERSAL | UNRESOLVED / EFFECTIVELY TIED |
|---|---|---|
| within `support_blind` (fair) | **1** | 6 |
| within `support_aware` (oracle) | 6 | 17 |
| across protocols | 27 | 14 |
| **total** | **34** | **37** |

**This is the result that decides the classification.** 34 of 71 winner changes are statistically
supported, which looks dramatic — but **27 of those 34 are across-protocol**, i.e. they compare the
fair protocol against a *diagnostic availability oracle that is explicitly never deployable*.
Instability that only appears when you hand a policy an oracle is not evidence that benchmark
design changes real conclusions.

**Inside the fair `support_blind` protocol, exactly one winner change is statistically supported:**

| from | to | winners | paired ΔAUNLLC |
|---|---|---|---|
| `ordinal_tier \| 0.3` | `per_analyte \| 0.6` | random_train_frequency → fixed_domain_order | in A: +0.00030 [−0.00173, +0.00242] (**includes 0**); in B: −0.00264 [−0.00516, −0.00016] (**excludes 0, barely**) |

That single supported reversal changes **both** the cost regime and the disclosure rate at once, so
it does not isolate either axis, and one of its two intervals includes zero. The remaining six
within-protocol winner changes are **UNRESOLVED / EFFECTIVELY TIED**.

## 7. What the strong adaptive policy did — an honest negative result

`greedy_eig` and `greedy_eig_per_cost` **lose to trivial baselines almost everywhere**, ranking
4th–6th in 14 of 16 conditions and barely separating from `no_acquisition` on the primary endpoint
(AUNLLC 0.3266 vs 0.3270).

The mechanism is visible in the spend table: the surrogate EIG policy issues 9.5 requests but
discloses only **2.4 cells at a 95% failure rate**, while `random_train_frequency` discloses
**11.6 cells at a 76% failure rate**. The surrogate scores a group by how much its *simulated*
completion would move the prediction, which systematically favours rarely-measured, high-leverage
analytes — precisely the ones least likely to have hidden data available. Under support-blind
replay it therefore spends most of its budget on empty requests.

A policy that knows *what would be informative* but not *what is obtainable* is beaten by one that
simply asks for what is usually measured. That is a real finding about surrogate-utility
acquisition under support-blind evaluation, and it is reported as-is.

## 8. Figures

```
experiments/acquisition/results/m4/figures/
  m4_primary_nll_vs_budget.png   NLL and Brier vs budget, primary condition
  m4_rank_flow.png               policy rank across all 16 conditions, with the
                                 fair/oracle boundary marked
```

## 9. Limitations

- One dataset, one 24h cutoff, one split assignment; no external validation.
- **Acquisition headroom is small.** At mask 0.6 the `group_hours` mechanism leaves only ~12% of
  cells hidden by the final boundary, and failed-request rates run 76–95%. Effects are therefore
  compressed, and the predeclared rates were deliberately not changed to enlarge them.
- The stability grid uses n = 2,000, so its confidence intervals are wider than the primary
  condition's n = 8,000.
- `support_aware` is a diagnostic oracle. Its rankings must never be quoted as deployable.
- Primary (AUNLLC) and secondary (Brier/AUROC/AP) endpoints disagree about the top policy; the
  margin between the top two is small in both.
- The surrogate EIG is a coarse 3-point quadrature completed in feature space, not exact EIG.
- Retrospective disclosure replay, not prospective ordering. No causal or deployment claim.
- 71 winner-change comparisons were tested without a multiplicity correction; the within-protocol
  conclusion rests on a single marginal interval and should be treated as suggestive.

## 10. Classification and headline

**M4-B — MODERATE ASSUMPTION SENSITIVITY.**

Under the fair support-blind protocol the ordering is broadly stable (mean τ-b +0.743, min +0.467;
`fixed_domain_order` wins 7 of 8 conditions; only one marginal supported reversal). Margins,
calibration and cost-efficiency do change materially across assumptions, and rankings scatter badly
once the availability oracle is introduced. **M4-A is not supported and is not claimed.**

**Strongest scientifically defensible headline:**

> Under support-blind retrospective disclosure replay on PhysioNet 2012, acquisition-policy
> ordering is largely stable to cost-regime and disclosure-rate assumptions (mean Kendall τ-b
> +0.743 within the fair protocol, one marginal supported winner change out of 28 pairs), but
> becomes unstable when policies are scored against a historical-availability oracle (27 supported
> winner changes across protocols). A surrogate expected-information-gain policy is beaten by a
> training-frequency random baseline and by a fixed domain-motivated ordering, because it spends
> 95% of its requests on information that is not available to disclose.

**BEST OVERALL KILLER RESULT FOUND: NO.**

The measured effect within the fair protocol is a *stability* result, not a fragility result. The
striking instability is confined to comparisons against a diagnostic oracle, which is a weaker and
more easily challenged claim. The most novel and defensible finding is §7 — that surrogate-utility
acquisition is beaten by "ask for what is usually measured" under support-blind replay — but that
is a single-dataset negative result, not a headline.

## 11. M5 recommendation: **MODIFY**

Cliniverse now has three solid, honest results (M2 values-dominate, M3 calibration drift under
selected-analyte loss, M4 ranking stability + the EIG failure) and no killer result. Before any
further milestone I recommend the project owner choose between:

1. **Consolidate.** Accept M4-B, write up the three results as a coherent robustness/negative-result
   study, and spend remaining effort on rigour and a minimal demo.
2. **Attack the headroom limitation.** The 76–95% failure rate is the single biggest constraint on
   every acquisition result. A disclosure regime with genuine headroom would need a predeclared
   design change and an explicit statement that it supersedes the current rates.
3. **Pivot the headline to §7** — availability-blind utility is worse than frequency priors — and
   test whether it replicates under a second masking mechanism.

**STOP. M5 not started.** No frontend, deployment, GCP or presentation work performed.
