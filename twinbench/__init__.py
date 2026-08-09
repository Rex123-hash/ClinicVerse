"""TwinBench — a reproducible benchmark for cost-aware, panel-level observation
acquisition under incomplete longitudinal patient data.

Scope note, which is load-bearing rather than boilerplate: TwinBench measures
**relative policy performance under a synthetic masking mechanism that it
specifies, seeds and discloses.** It does not estimate the clinical utility or
real-world deployment value of any acquisition policy. Doing so would require
the no-unobserved-confounding assumption of the active-feature-acquisition
performance-evaluation literature, which is false in ICU data where missingness
is driven by clinician judgement absent from the record.

See ``docs/DECISIONS.md`` D-003 and ``docs/LIMITATIONS.md`` section 2.
"""

from __future__ import annotations

__all__: list[str] = []
