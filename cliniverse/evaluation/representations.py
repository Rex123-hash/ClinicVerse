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
- Median imputation is the worst offender: an imputed cell holds *exactly* the
  training median, and a tree can split on that value to recover the missingness
  indicator almost perfectly.
- Native NaN handling in gradient boosting is strictly worse still — the learned
  default direction *is* a missingness indicator.

We therefore (a) never use native NaN routing in ``values_only``, and (b) provide
``values_only_stochastic``, which imputes by sampling from the training marginal
of each feature. The sampled values are indistinguishable from observed ones, so
the "exactly median" tell disappears. The gap between the two variants bounds how
much of ``values_only``'s performance is residual missingness information rather
than physiology.
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
    STOCHASTIC = "stochastic"


@dataclasses.dataclass(frozen=True, slots=True)
class FittedImputer:
    """Imputer fitted on training rows only.

    ``MEDIAN`` substitutes the training median. ``STOCHASTIC`` samples from the
    observed training values of each feature, which removes the "exactly the
    median" signature that otherwise lets a model recover missingness.
    """

    strategy: ImputationStrategy
    medians: FloatArray
    pools: tuple[FloatArray, ...]
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
        if strategy is ImputationStrategy.STOCHASTIC:
            pools = tuple(
                (
                    col[np.isfinite(col)]
                    if np.isfinite(col).any()
                    else np.array([medians[j]], dtype=np.float64)
                )
                for j, col in enumerate(x.T.astype(np.float64))
            )
        return cls(strategy=strategy, medians=medians, pools=pools, seed=seed)

    def transform(self, x: FloatArray, *, draw_seed: int) -> FloatArray:
        out = np.array(x, dtype=np.float64, copy=True)
        missing = ~np.isfinite(out)
        if not missing.any():
            return out
        if self.strategy is ImputationStrategy.MEDIAN:
            out[missing] = np.broadcast_to(self.medians, out.shape)[missing]
            return out

        rng = np.random.default_rng(self.seed + draw_seed)
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
