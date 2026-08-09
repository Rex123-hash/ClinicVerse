"""Masking mechanisms: which observed cells are hidden from the policy.

A mechanism takes the *observed* mask for one patient and returns a boolean
"hidden" array of the same shape. Only observed cells may be hidden — a cell
that was never recorded cannot be hidden, because there is nothing to disclose.

Mechanisms are deterministic given ``(patient_index, seed)``. Randomness is
drawn from a seed derived from both, so a patient's mask does not depend on how
many patients were processed before it, and regenerating any single case
reproduces exactly.

Three mechanisms, in increasing realism:

``mcar_cells``   Hide individual analyte-hour cells independently. The easiest
                 setting, and the one implied by feature-level acquisition.
``panel_events`` Hide whole co-measured events: every analyte of a group
                 recorded in the same hour disappears together. This matches how
                 the data is actually generated (E-001: within-group Jaccard
                 0.78-0.97) and is the default.
``time_blocks``  Hide contiguous hour ranges across all groups — a monitoring
                 gap rather than an unordered test.
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Final

import numpy as np
import numpy.typing as npt

from cliniverse.acquisition.catalogue import PanelCatalogue
from cliniverse.exceptions import ConfigError

BoolArray = npt.NDArray[np.bool_]

#: Mixed into every seed so that mechanism seeds cannot collide with split seeds.
_SEED_SALT: Final = 0x7B10_0001


def _rng(seed: int, patient_index: int) -> np.random.Generator:
    """Per-patient generator, independent of processing order."""
    return np.random.default_rng((seed ^ _SEED_SALT) + 1_000_003 * patient_index)


@dataclasses.dataclass(frozen=True, slots=True)
class MaskingMechanism(abc.ABC):
    """Base class. ``rate`` is the expected fraction of observed cells hidden."""

    rate: float
    seed: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.rate <= 1.0:
            raise ConfigError(f"masking rate must be in [0, 1], got {self.rate}")

    @property
    @abc.abstractmethod
    def mechanism_id(self) -> str:
        """Stable identifier recorded in every case manifest."""

    @abc.abstractmethod
    def _draw(
        self,
        observed: BoolArray,
        rng: np.random.Generator,
        catalogue: PanelCatalogue,
        variable_names: tuple[str, ...],
    ) -> BoolArray: ...

    def hidden_for(
        self,
        observed: BoolArray,
        patient_index: int,
        catalogue: PanelCatalogue,
        variable_names: tuple[str, ...],
    ) -> BoolArray:
        """Return the hidden mask for one patient, shape ``(T, V)``.

        Guarantees ``hidden implies observed``: unobserved cells are never hidden.
        """
        if observed.ndim != 2:
            raise ConfigError(f"expected a (T, V) observed mask, got {observed.shape}")
        hidden = self._draw(
            observed, _rng(self.seed, patient_index), catalogue, variable_names
        )
        hidden &= observed
        return hidden


@dataclasses.dataclass(frozen=True, slots=True)
class McarCells(MaskingMechanism):
    """Hide observed analyte-hour cells independently at random."""

    @property
    def mechanism_id(self) -> str:
        return f"mcar_cells@{self.rate:g}#{self.seed}"

    def _draw(
        self,
        observed: BoolArray,
        rng: np.random.Generator,
        catalogue: PanelCatalogue,
        variable_names: tuple[str, ...],
    ) -> BoolArray:
        del catalogue, variable_names
        return rng.random(observed.shape) < self.rate


@dataclasses.dataclass(frozen=True, slots=True)
class PanelEvents(MaskingMechanism):
    """Hide whole co-measured events: a group's analytes in one hour, together.

    This is the default because it matches how the data is generated. Hiding
    individual analytes from a group that was measured as a unit would create
    patterns that do not occur in the source records.
    """

    @property
    def mechanism_id(self) -> str:
        return f"panel_events@{self.rate:g}#{self.seed}"

    def _draw(
        self,
        observed: BoolArray,
        rng: np.random.Generator,
        catalogue: PanelCatalogue,
        variable_names: tuple[str, ...],
    ) -> BoolArray:
        hidden = np.zeros_like(observed)
        index = {name: i for i, name in enumerate(variable_names)}
        for panel in catalogue.panels.values():
            cols = [index[m] for m in panel.members if m in index]
            if not cols:
                continue
            block = observed[:, cols]
            # An "event" is an hour in which any member of this group was measured.
            event_hours = np.flatnonzero(block.any(axis=1))
            if event_hours.size == 0:
                continue
            chosen = event_hours[rng.random(event_hours.size) < self.rate]
            for hour in chosen:
                hidden[hour, cols] = True
        return hidden


@dataclasses.dataclass(frozen=True, slots=True)
class TimeBlocks(MaskingMechanism):
    """Hide contiguous hour ranges across every variable — a monitoring gap."""

    block_hours: int = 6

    def __post_init__(self) -> None:
        # Explicit base call, not `super()`: with `slots=True` the dataclass
        # decorator builds a replacement class, so the zero-argument `super()`
        # closure cell points at the discarded original and raises TypeError.
        MaskingMechanism.__post_init__(self)
        if self.block_hours < 1:
            raise ConfigError(f"block_hours must be >= 1, got {self.block_hours}")

    @property
    def mechanism_id(self) -> str:
        return f"time_blocks@{self.rate:g}/{self.block_hours}h#{self.seed}"

    def _draw(
        self,
        observed: BoolArray,
        rng: np.random.Generator,
        catalogue: PanelCatalogue,
        variable_names: tuple[str, ...],
    ) -> BoolArray:
        del catalogue, variable_names
        n_hours = observed.shape[0]
        hidden = np.zeros_like(observed)
        n_blocks = max(1, n_hours // self.block_hours)
        for b in range(n_blocks):
            if rng.random() >= self.rate:
                continue
            start = b * self.block_hours
            hidden[start : start + self.block_hours, :] = True
        return hidden


MECHANISMS: Final[dict[str, type[MaskingMechanism]]] = {
    "mcar_cells": McarCells,
    "panel_events": PanelEvents,
    "time_blocks": TimeBlocks,
}


def build_mechanism(name: str, rate: float, seed: int, **kwargs: object) -> MaskingMechanism:
    """Construct a mechanism by name."""
    try:
        cls = MECHANISMS[name]
    except KeyError as exc:
        raise ConfigError(
            f"unknown masking mechanism {name!r}; available: {sorted(MECHANISMS)}"
        ) from exc
    return cls(rate=rate, seed=seed, **kwargs)
