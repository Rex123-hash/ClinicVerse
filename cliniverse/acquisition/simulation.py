"""Counterfactual feature simulation for expected-utility policies.

A myopic acquisition policy must ask "what would my prediction look like if I
acquired this group?" *before* the value is disclosed. It therefore needs a model
of the completion, not the completion itself.

This module provides that model directly in feature space: acquiring a group is
simulated as one additional observation, at the current boundary, for each member
variable, taking a value from a training-fold quantile. The summaries that can
be reconstructed from the current feature vector are updated as follows:

    n_obs   += 1              ever    := 1
    recency := 0              last    := q
    mean    := (mean*n + q)/(n+1)
    min     := min(min, q)    max     := max(max, q)

The slope and global distinct-variable count cannot be reconstructed exactly
from this compressed state and are left unchanged. This makes the completion a
representation-incomplete heuristic rather than a simulated acquisition with
fully compatible feature semantics.

Working in feature space keeps the simulation cheap enough to batch across all
patients and all candidate actions, which keeps the entropy heuristic tractable.

**Every quantile comes from training folds only.** Nothing in this module reads a
hidden value, so a policy built on it cannot leak one.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from cliniverse.exceptions import ConfigError

FloatArray = npt.NDArray[np.float64]

#: Prefixes of the summary statistics produced by `build_features`.
_STATS = ("last", "mean", "min", "max", "slope")


@dataclasses.dataclass(frozen=True, slots=True)
class FeatureLayout:
    """Column indices of each summary statistic, per variable."""

    columns: dict[str, dict[str, int]]

    @classmethod
    def from_names(cls, names: tuple[str, ...]) -> FeatureLayout:
        columns: dict[str, dict[str, int]] = {}
        for i, name in enumerate(names):
            if "::" not in name:
                continue
            stat, _, variable = name.partition("::")
            columns.setdefault(variable, {})[stat] = i
        return cls(columns=columns)

    def has(self, variable: str, stat: str) -> bool:
        return stat in self.columns.get(variable, {})


def build_training_quantiles(
    x: FloatArray,
    m: npt.NDArray[np.bool_],
    variable_names: tuple[str, ...],
    *,
    quantiles: tuple[float, ...] = (0.25, 0.50, 0.75),
) -> dict[str, tuple[float, ...]]:
    """Per-variable value quantiles from observed training cells only."""
    table: dict[str, tuple[float, ...]] = {}
    for j, name in enumerate(variable_names):
        observed = x[:, :, j][m[:, :, j]]
        observed = observed[np.isfinite(observed)]
        if observed.size == 0:
            table[name] = tuple(0.0 for _ in quantiles)
        else:
            table[name] = tuple(
                float(v) for v in np.quantile(observed.astype(np.float64), quantiles)
            )
    return table


def make_simulator(
    feature_names: tuple[str, ...],
    group_members: dict[str, tuple[str, ...]],
    quantile_table: dict[str, tuple[float, ...]],
) -> Callable[[FloatArray, str, int], FloatArray]:
    """Return ``simulate(features, action, quantile_index) -> features``.

    The returned callable is pure and batched over patients.
    """
    layout = FeatureLayout.from_names(feature_names)

    def simulate(features: FloatArray, action: str, quantile_index: int) -> FloatArray:
        members = group_members.get(action)
        if members is None:
            raise ConfigError(f"unknown action {action!r}")
        out = np.array(features, dtype=np.float64, copy=True)

        for variable in members:
            cols = layout.columns.get(variable)
            if not cols:
                continue
            qs = quantile_table.get(variable)
            if qs is None or quantile_index >= len(qs):
                continue
            q = qs[quantile_index]

            n_col = cols.get("n_obs")
            n_before = out[:, n_col].copy() if n_col is not None else np.zeros(len(out))
            n_before = np.clip(n_before, 0.0, None)

            if n_col is not None:
                out[:, n_col] = n_before + 1.0
            if (c := cols.get("ever")) is not None:
                out[:, c] = 1.0
            if (c := cols.get("recency")) is not None:
                out[:, c] = 0.0
            if (c := cols.get("last")) is not None:
                out[:, c] = q
            if (c := cols.get("mean")) is not None:
                # Where nothing was observed the stored value is an imputed
                # constant, so the new observation replaces it outright.
                prior = np.where(n_before > 0, out[:, c], 0.0)
                out[:, c] = (prior * n_before + q) / (n_before + 1.0)
            if (c := cols.get("min")) is not None:
                out[:, c] = np.where(n_before > 0, np.minimum(out[:, c], q), q)
            if (c := cols.get("max")) is not None:
                out[:, c] = np.where(n_before > 0, np.maximum(out[:, c], q), q)
        return out

    return simulate
