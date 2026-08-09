# Cliniverse

**Research software. Not medical advice, not a diagnostic tool, not a treatment recommender.**
All outputs are research/model simulations.

Cliniverse is a reproducible benchmark and evaluation harness for **cost-aware, panel-level,
time-aware observation acquisition** under incomplete longitudinal patient data, together with
calibrated-uncertainty baseline models.

The question it tries to answer: *given a patient timeline with missing observations and a
limited budget, which observation should you acquire next — and can any policy do this better
than clinical and statistical heuristics?*

## Status

Pre-implementation. The research assessment is complete; implementation is gated on scope
sign-off. See [`docs/STATUS.md`](docs/STATUS.md).

## Start here

| Document | Contents |
|---|---|
| [`docs/research_assessment.md`](docs/research_assessment.md) | Literature review, prior-art analysis, dataset evaluation, revised architecture, roadmap |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Architecture decisions and rejected alternatives |
| [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) | What this system cannot claim — **read before citing any result** |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | Executed runs only. No projected numbers. |

## Data

Primary dataset: **PhysioNet/CinC Challenge 2012** — 12,000 ICU patients, 36 irregularly-sampled
clinical variables over 48 hours, ≥76.7% missing on an hourly grid.
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
