"""Summary-feature construction, with availability and value information kept
in strictly separate blocks.

The separation is the point. In ICU data the *fact that a test was ordered*
encodes clinician concern, so a model can score well without reading any
measured value at all. Building availability and value features as independent
blocks lets us train on either alone and measure how much of the signal is
information about the patient versus information about the clinician's
behaviour.

Blocks:
    AVAILABILITY  derived from the observation mask only — counts, ever-measured
                  flags, time since last observation. Contains no measured value.
    VALUES        derived from measured values only — last, mean, min, max, slope.
                  Missing entries are imputed, and no mask indicator is exposed.
    STATICS       admission descriptors.
"""

from __future__ import annotations

import dataclasses
import enum
import warnings

import numpy as np
import numpy.typing as npt

from cliniverse.data.cohort import BoolArray, Cohort, FloatArray

_NEVER_OBSERVED = -1.0


class FeatureBlock(enum.StrEnum):
    """Independently selectable groups of features."""

    AVAILABILITY = "availability"
    VALUES = "values"
    STATICS = "statics"


@dataclasses.dataclass(frozen=True, slots=True)
class FeatureSet:
    """A design matrix plus column provenance."""

    x: FloatArray
    names: tuple[str, ...]
    blocks: tuple[FeatureBlock, ...]

    def __post_init__(self) -> None:
        if self.x.shape[1] != len(self.names) or len(self.names) != len(self.blocks):
            raise ValueError("feature matrix, names and blocks must align")

    @property
    def n_features(self) -> int:
        return int(self.x.shape[1])

    def columns_for(self, *blocks: FeatureBlock) -> npt.NDArray[np.intp]:
        wanted = set(blocks)
        return np.array([i for i, b in enumerate(self.blocks) if b in wanted], dtype=np.intp)

    def subset(self, *blocks: FeatureBlock) -> FeatureSet:
        """Restrict to the named blocks. Used for the availability-only ablation."""
        cols = self.columns_for(*blocks)
        return FeatureSet(
            x=self.x[:, cols],
            names=tuple(self.names[i] for i in cols),
            blocks=tuple(self.blocks[i] for i in cols),
        )


def _availability_features(
    m: BoolArray, variable_names: tuple[str, ...]
) -> tuple[FloatArray, list[str]]:
    """Counts, ever-measured flags and recency — from the mask alone."""
    _, t, _ = m.shape
    counts = m.sum(axis=1).astype(np.float32)
    ever = m.any(axis=1).astype(np.float32)

    # Hours since the most recent observation, relative to the cutoff.
    hour_index = np.arange(t, dtype=np.float32)[None, :, None]
    last_seen = np.where(m, hour_index, -np.inf).max(axis=1)
    recency = np.where(np.isfinite(last_seen), (t - 1) - last_seen, _NEVER_OBSERVED)

    total = counts.sum(axis=1, keepdims=True)
    distinct = ever.sum(axis=1, keepdims=True)

    blocks: list[FloatArray] = [counts, ever, recency.astype(np.float32), total, distinct]
    names = (
        [f"n_obs::{v}" for v in variable_names]
        + [f"ever::{v}" for v in variable_names]
        + [f"recency::{v}" for v in variable_names]
        + ["n_obs::TOTAL", "n_distinct_vars::TOTAL"]
    )
    return np.concatenate(blocks, axis=1).astype(np.float32), names


def _value_features(
    x: FloatArray, m: BoolArray, variable_names: tuple[str, ...]
) -> tuple[FloatArray, list[str]]:
    """Last / mean / min / max / slope of measured values.

    Missing summaries are left as NaN here and imputed later inside a fold, so
    the imputer never sees validation rows.
    """
    n, t, v = x.shape
    masked = np.where(m, x, np.nan)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        # All-NaN columns are expected for never-observed variables and remain
        # NaN for fold-local imputation; suppress only those NumPy warnings.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(masked, axis=1).astype(np.float32)
        vmin = np.nanmin(masked, axis=1).astype(np.float32)
        vmax = np.nanmax(masked, axis=1).astype(np.float32)

    hour_index = np.arange(t, dtype=np.float32)[None, :, None]
    last_idx = np.where(m, hour_index, -np.inf).argmax(axis=1)
    first_idx = np.where(m, hour_index, np.inf).argmin(axis=1)
    rows = np.arange(n)[:, None]
    cols = np.arange(v)[None, :]
    last = x[rows, last_idx, cols]
    first = x[rows, first_idx, cols]

    span = (last_idx - first_idx).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.where(span > 0, (last - first) / np.where(span > 0, span, 1.0), 0.0)
    never = ~m.any(axis=1)
    for arr in (last, first, slope):
        arr[never] = np.nan

    blocks: list[FloatArray] = [last, mean, vmin, vmax, slope]
    names = (
        [f"last::{v_}" for v_ in variable_names]
        + [f"mean::{v_}" for v_ in variable_names]
        + [f"min::{v_}" for v_ in variable_names]
        + [f"max::{v_}" for v_ in variable_names]
        + [f"slope::{v_}" for v_ in variable_names]
    )
    return np.concatenate(blocks, axis=1).astype(np.float32), names


def build_features(cohort: Cohort) -> FeatureSet:
    """Build the full feature set from an already-truncated cohort.

    The cohort must be truncated to the decision point *before* calling this;
    every feature is computed over the whole array it is given.
    """
    avail, avail_names = _availability_features(cohort.m, cohort.variable_names)
    values, value_names = _value_features(cohort.x, cohort.m, cohort.variable_names)
    statics = cohort.statics.astype(np.float32)
    static_names = [f"static::{s}" for s in cohort.static_names]

    return FeatureSet(
        x=np.concatenate([avail, values, statics], axis=1).astype(np.float32),
        names=tuple(avail_names + value_names + static_names),
        blocks=tuple(
            [FeatureBlock.AVAILABILITY] * len(avail_names)
            + [FeatureBlock.VALUES] * len(value_names)
            + [FeatureBlock.STATICS] * len(static_names)
        ),
    )
