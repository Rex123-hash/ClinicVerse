"""The in-memory representation of a parsed patient cohort.

A ``Cohort`` is a dense, aligned view of an inherently sparse dataset. The mask
``M`` is not an implementation detail — it *is* the signal for missingness-aware
models and the support over which acquisition policies operate, so it is carried
alongside ``X`` everywhere rather than being folded into imputed values.
"""

from __future__ import annotations

import dataclasses
from typing import Self

import numpy as np
import numpy.typing as npt

from cliniverse.exceptions import DataError

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]


@dataclasses.dataclass(frozen=True, slots=True)
class Cohort:
    """A parsed, hourly-binned cohort.

    Attributes:
        record_ids: ``(n,)`` patient record identifiers.
        source_set: ``(n,)`` originating record set (``'a'``/``'b'``/``'c'``).
        x: ``(n, T, V)`` binned values; ``NaN`` wherever unobserved.
        m: ``(n, T, V)`` observation mask; ``True`` where measured.
        statics: ``(n, S)`` static descriptors; ``NaN`` where unknown.
        statics_mask: ``(n, S)`` ``True`` where the static value is known.
        labels: task name -> ``(n,)`` target array.
        variable_names: length-``V`` column names for ``x``/``m``.
        static_names: length-``S`` column names for ``statics``.
    """

    record_ids: IntArray
    source_set: npt.NDArray[np.str_]
    x: FloatArray
    m: BoolArray
    statics: FloatArray
    statics_mask: BoolArray
    labels: dict[str, FloatArray]
    variable_names: tuple[str, ...]
    static_names: tuple[str, ...]

    def __post_init__(self) -> None:
        n, t, v = self.x.shape
        if self.m.shape != self.x.shape:
            raise DataError(f"mask shape {self.m.shape} != values shape {self.x.shape}")
        if len(self.variable_names) != v:
            raise DataError(
                f"{len(self.variable_names)} variable names for {v} value columns"
            )
        if len(self.static_names) != self.statics.shape[1]:
            raise DataError("static name count does not match statics width")
        for name, arr in (
            ("record_ids", self.record_ids),
            ("source_set", self.source_set),
            ("statics", self.statics),
            ("statics_mask", self.statics_mask),
        ):
            if arr.shape[0] != n:
                raise DataError(f"{name} has {arr.shape[0]} rows, expected {n}")
        for task, y in self.labels.items():
            if y.shape[0] != n:
                raise DataError(f"label {task!r} has {y.shape[0]} rows, expected {n}")
        # Invariant that the rest of the codebase relies on: a cell is masked-in
        # if and only if it holds a finite value.
        if bool(np.any(self.m & ~np.isfinite(self.x))):
            raise DataError("mask marks a cell observed but its value is not finite")
        if bool(np.any(~self.m & np.isfinite(self.x))):
            raise DataError("a finite value exists in a cell marked unobserved")
        del t

    # ------------------------------------------------------------- shape ----
    @property
    def n_patients(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_hours(self) -> int:
        return int(self.x.shape[1])

    @property
    def n_variables(self) -> int:
        return int(self.x.shape[2])

    def variable_index(self, name: str) -> int:
        try:
            return self.variable_names.index(name)
        except ValueError as exc:
            raise DataError(f"unknown variable {name!r}") from exc

    # ---------------------------------------------------------- selection ----
    def select(self, idx: IntArray | BoolArray) -> Self:
        """Return a new cohort containing only the selected patients."""
        return dataclasses.replace(
            self,
            record_ids=self.record_ids[idx],
            source_set=self.source_set[idx],
            x=self.x[idx],
            m=self.m[idx],
            statics=self.statics[idx],
            statics_mask=self.statics_mask[idx],
            labels={k: v[idx] for k, v in self.labels.items()},
        )

    def truncate(self, hours: int) -> Self:
        """Restrict the observation window to the first ``hours`` hours.

        This is how a prediction-time cutoff is enforced. Using it — rather than
        slicing arrays ad hoc at call sites — keeps the "model may not see the
        future" rule in one auditable place.
        """
        if not 0 < hours <= self.n_hours:
            raise DataError(f"hours must be in 1..{self.n_hours}, got {hours}")
        return dataclasses.replace(
            self, x=self.x[:, :hours, :], m=self.m[:, :hours, :]
        )

    # ------------------------------------------------------------ summary ----
    def observation_counts(self) -> IntArray:
        """``(n,)`` number of observed cells per patient."""
        counts: IntArray = self.m.sum(axis=(1, 2)).astype(np.int64)
        return counts

    def coverage(self) -> FloatArray:
        """``(V,)`` fraction of patients in which each variable is ever observed."""
        cov: FloatArray = self.m.any(axis=1).mean(axis=0).astype(np.float32)
        return cov

    def describe(self) -> dict[str, float | int]:
        """Summary statistics, used in logs and reproducibility assertions."""
        occupancy = float(self.m.mean())
        return {
            "n_patients": self.n_patients,
            "n_hours": self.n_hours,
            "n_variables": self.n_variables,
            "observed_cells": int(self.m.sum()),
            "grid_occupancy": occupancy,
            "missing_fraction": 1.0 - occupancy,
            "degenerate_records": int((self.observation_counts() == 0).sum()),
        }
