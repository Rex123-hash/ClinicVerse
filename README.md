# Cliniverse

**Research software. Not medical advice, not a diagnostic tool, not a treatment recommender.**
All outputs are research/model simulations.

A model that never sees a single lab value reaches **AUROC 0.7224** using only
measurement-presence patterns. Clinical AI can learn the care process itself — not just patient
physiology. **Cliniverse stress-tests whether information-acquisition methods still look reliable
when that historical measurement-policy shortcut is disrupted.**

That figure is an executed result ([E-002](docs/EXPERIMENTS.md), n = 8,000, 5-fold CV,
95% CI [0.707, 0.738]) on the PhysioNet/CinC 2012 development cohort. It is an associational
finding about what a model can predict from measurement-presence patterns — not a causal claim,
and not an assertion that any model infers clinician intent.

## What this is

A benchmark and evaluation harness that runs identical acquisition policies under two disclosure
protocols — **support-aware** (standard replay: only historically recorded groups are requestable)
and **support-blind** (any group may be requested; unavailable requests cost full price and return
nothing) — and reports the paired difference, plus whether policy rankings survive four cost
regimes.

**Group-level acquisition with shared cost is not novel** — see
[Yu et al., ICLR 2023](https://arxiv.org/abs/2302.10261), the closest prior art. We adopt that
setting; we do not claim it. Our contribution is a **measurement** one, not a new method.

## Status

M0 complete; M1 in progress. See [`docs/STATUS.md`](docs/STATUS.md).

## Start here

| Document | Contents |
|---|---|
| [`docs/research_assessment.md`](docs/research_assessment.md) | **Current source of truth** — hypothesis, prior art, retracted claims, dataset, estimand, roadmap |
| [`docs/BENCHMARK_SPEC.md`](docs/BENCHMARK_SPEC.md) | Formal estimand, information boundary, policy-visible vs evaluator-only split |
| [`docs/REVIEW_RESPONSE_0.md`](docs/REVIEW_RESPONSE_0.md) | Independent review findings, verification, and what we retracted |
| [`docs/NOVELTY_REASSESSMENT.md`](docs/NOVELTY_REASSESSMENT.md) | Prior art and the claims we may not make |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Architecture decisions and rejected alternatives |
| [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) | What this system cannot claim — **read before citing any result** |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | Executed runs only. No projected numbers. |

## Data

Primary dataset: **PhysioNet/CinC Challenge 2012** — 12,000 ICU patients, 36 irregularly-sampled
clinical variables over 48 hours, **79.75% missing** on the binned hourly grid.
Open Data Commons Attribution License v1.0, **no credentialing required**, ~20 MB.

Verify access and reproduce the dataset statistics:

```bash
python scripts/verify_physionet2012.py
```

## The central caveat

Cliniverse measures **relative policy performance under a synthetic masking mechanism that we
specify, seed and disclose.** It does **not** estimate the clinical utility or real-world
deployment value of any acquisition policy — that would require identification assumptions
(no-unobserved-confounding) which are false in ICU data, where missingness is informative and
driven by clinician judgement absent from the record.

See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) §2.

## License and attribution

Code license: TBD. Data used under ODC-BY v1.0 (PhysioNet) and Apache-2.0 (Synthea), within
their terms. No credentialed or identifiable patient data is used or committed.
