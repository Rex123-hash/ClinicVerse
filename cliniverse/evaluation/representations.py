"""The M2 representation contract.

Three representations are compared on identical patients and identical folds:

``mask_only``    measurement-presence information only — counts, ever-measured
                 flags, recency. No clinical value of any kind.
``values_only``  clinical values only — last/mean/min/max/slope summaries. Every
                 explicit measurement-presence feature is removed.
``values_mask``  both blocks.

Statics (Age, Gender, Height, ICUType, AdmissionWeight) are **excluded from all
three**, so the contrast isolates exactly one thing: presence versus value. They
are reported separately as supplementary representations.

The honest caveat on ``values_only``
------------------------------------
It is impossible to make a values-only representation carry *zero* information
about missingness, and we do not claim otherwise.

- A summary of a never-measured variable does not exist, so something must be
  substituted. Whatever is substituted is, in principle, detectable.
- Median imputation leaves a detectable point mass at the training median. How
  much a fitted model exploits it is an empirical question, not an assumption.
- Native NaN handling in gradient boosting is strictly worse still — the learned
  default direction *is* a missingness indicator.

We therefore (a) never use native NaN routing in ``values_only``, and (b) provide
two diagnostics: a small train-derived jitter around the median and an empirical
marginal draw for each summary column. Marginal draws remove the exact point mass
but can break correlations among last/mean/min/max/slope summaries, so their
performance gap is not an estimate of missingness signal by itself.
"""

from __future__ import annotations

import dataclasses
import enum
import warnings

import numpy as np
import numpy.typing as npt

from cliniverse.data.cohort import Cohort
from cliniverse.encoders import FeatureBlock, FeatureSet, build_features
from cliniverse.exceptions import ConfigError

FloatArray = npt.NDArray[np.float64]


class Representation(enum.StrEnum):
    """Named feature views compared in M2."""

    MASK_ONLY = "mask_only"
    VALUES_ONLY = "values_only"
    VALUES_MASK = "values_mask"
    # Supplementary — not part of the three-way contract.
    STATICS_ONLY = "statics_only"
    VALUES_MASK_STATICS = "values_mask_statics"


#: Blocks composing each representation.
REPRESENTATION_BLOCKS: dict[Representation, tuple[FeatureBlock, ...]] = {
    Representation.MASK_ONLY: (FeatureBlock.AVAILABILITY,),
    Representation.VALUES_ONLY: (FeatureBlock.VALUES,),
    Representation.VALUES_MASK: (FeatureBlock.AVAILABILITY, FeatureBlock.VALUES),
    Representation.STATICS_ONLY: (FeatureBlock.STATICS,),
    Representation.VALUES_MASK_STATICS: (
        FeatureBlock.AVAILABILITY,
        FeatureBlock.VALUES,
        FeatureBlock.STATICS,
    ),
}

#: The three binding representations of the M2 contract.
CORE_REPRESENTATIONS: tuple[Representation, ...] = (
    Representation.MASK_ONLY,
    Representation.VALUES_ONLY,
    Representation.VALUES_MASK,
)


class ImputationStrategy(enum.StrEnum):
    MEDIAN = "median"
    MEDIAN_JITTER = "median_jitter"
    EMPIRICAL_MARGINAL = "empirical_marginal"


@dataclasses.dataclass(frozen=True, slots=True)
class FittedImputer:
    """Imputer fitted on training rows only.

    ``MEDIAN`` substitutes the training median. ``MEDIAN_JITTER`` adds small,
    independent Gaussian noise scaled by 1% of the training IQR. It tests whether
    exact equality to the median is necessary, not whether missingness has been
    removed. ``EMPIRICAL_MARGINAL`` draws independently from observed training
    values in each summary column; it is a diagnostic with a known
    correlation-breaking disadvantage.
    """

    strategy: ImputationStrategy
    medians: FloatArray
    pools: tuple[FloatArray, ...]
    jitter_scales: FloatArray
    seed: int

    @classmethod
    def fit(
        cls,
        x: FloatArray,
        *,
        strategy: ImputationStrategy = ImputationStrategy.MEDIAN,
        seed: int = 0,
    ) -> FittedImputer:
        if x.ndim != 2:
            raise ConfigError(f"expected a 2-D design matrix, got {x.shape}")
        with warnings.catch_warnings():
            # A column unobserved throughout training yields an all-NaN slice.
            # That is expected and handled on the next line, so the warning is
            # noise rather than information.
            warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
            medians = np.nanmedian(x, axis=0)
        # A feature never observed in training has no median; 0.0 is as good as
        # anything and is recorded rather than silently produced by nanmedian.
        medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float64)

        pools: tuple[FloatArray, ...] = ()
        if strategy is ImputationStrategy.EMPIRICAL_MARGINAL:
            pools = tuple(
                (
                    col[np.isfinite(col)]
                    if np.isfinite(col).any()
                    else np.array([medians[j]], dtype=np.float64)
                )
                for j, col in enumerate(x.T.astype(np.float64))
            )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            q25, q75 = np.nanpercentile(x, [25.0, 75.0], axis=0)
            std = np.nanstd(x, axis=0)
        jitter_scales = 0.01 * (q75 - q25)
        jitter_scales = np.where(
            np.isfinite(jitter_scales) & (jitter_scales > 0),
            jitter_scales,
            0.01 * std,
        )
        fallback = np.maximum(np.abs(medians) * 1e-6, 1e-6)
        jitter_scales = np.where(
            np.isfinite(jitter_scales) & (jitter_scales > 0), jitter_scales, fallback
        ).astype(np.float64)
        return cls(
            strategy=strategy,
            medians=medians,
            pools=pools,
            jitter_scales=jitter_scales,
            seed=seed,
        )

    def transform(self, x: FloatArray, *, draw_seed: int) -> FloatArray:
        out = np.array(x, dtype=np.float64, copy=True)
        missing = ~np.isfinite(out)
        if not missing.any():
            return out
        if self.strategy is ImputationStrategy.MEDIAN:
            out[missing] = np.broadcast_to(self.medians, out.shape)[missing]
            return out

        rng = np.random.default_rng(self.seed + draw_seed)
        if self.strategy is ImputationStrategy.MEDIAN_JITTER:
            draws = self.medians[None, :] + rng.normal(size=out.shape) * self.jitter_scales
            out[missing] = draws[missing]
            return out

        for j in range(out.shape[1]):
            sel = missing[:, j]
            n = int(sel.sum())
            if n:
                out[sel, j] = rng.choice(self.pools[j], size=n, replace=True)
        return out


@dataclasses.dataclass(frozen=True, slots=True)
class RepresentationView:
    """A design matrix for one representation, with provenance."""

    representation: Representation
    x: FloatArray
    names: tuple[str, ...]

    @property
    def n_features(self) -> int:
        return int(self.x.shape[1])

    def contains_presence_features(self) -> bool:
        """Whether any explicit measurement-presence feature is present.

        Used by the leakage tests to assert ``values_only`` really is value-only.
        """
        prefixes = ("n_obs::", "ever::", "recency::", "n_distinct_vars::")
        return any(n.startswith(prefixes) for n in self.names)

    def feature_inventory(self) -> list[dict[str, object]]:
        """Machine-readable scientific provenance for every feature column."""
        inventory: list[dict[str, object]] = []
        for name in self.names:
            statistic, source = name.split("::", maxsplit=1)
            explicit = statistic in {"n_obs", "ever", "recency", "n_distinct_vars"}
            numeric = statistic in {"last", "mean", "min", "max", "slope", "static"}
            inventory.append(
                {
                    "name": name,
                    "source_variable": source,
                    "statistic": statistic,
                    "uses_numeric_value": numeric,
                    "measurement_presence": (
                        "explicit" if explicit else "implicit" if numeric else "no"
                    ),
                    "uses_count": statistic in {"n_obs", "n_distinct_vars"},
                    "uses_recency": statistic == "recency",
                    "uses_time_since_last": statistic == "recency",
                    "uses_missing_sentinel": statistic == "recency",
                    "requires_imputation": numeric,
                    "cutoff_safe": True,
                }
            )
        return inventory


def build_representation(cohort: Cohort, representation: Representation) -> RepresentationView:
    """Build one representation from an already-truncated cohort.

    The cohort must be truncated to the decision point before calling this, so
    that no feature can depend on post-cutoff data.
    """
    if representation not in REPRESENTATION_BLOCKS:
        raise ConfigError(f"unknown representation {representation!r}")
    features: FeatureSet = build_features(cohort)
    view = features.subset(*REPRESENTATION_BLOCKS[representation])
    return RepresentationView(
        representation=representation,
        x=view.x.astype(np.float64),
        names=view.names,
    )
