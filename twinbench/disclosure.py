"""The disclosure engine: what a policy may see, request, and be charged for.

This module carries the benchmark's validity. E-002 showed that
measurement-presence patterns alone support AUROC 0.7224 on this cohort, so any
leak of the hidden support would let a policy score well by reading the
historical measurement pattern rather than by acquiring information.

Two invariants, both enforced structurally rather than by convention:

**SO-1** ``PolicyView`` holds no reference to evaluator state. It is built by
copying only disclosed arrays, so there is no attribute path from what a policy
receives to the hidden set, the historical support, or the targets.

**SO-2** A group holding hidden values and a group holding none are
indistinguishable *before purchase*. This is why an unavailable request is
**charged in full and discloses nothing**: if failed requests were free, a
policy could probe availability at no cost and reconstruct the very signal the
protocol exists to remove.

Protocols
---------
``support_aware``  Only groups with at least one hidden value within the current
                   boundary may be requested. This reproduces standard
                   retrospective replay practice, in which the acquirable set is
                   derived from what was historically recorded — so availability
                   is a free signal.
``support_blind``  The whole catalogue may be requested. Availability costs
                   budget to discover.

The paired difference between these two is the benchmark's primary result.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Final

import numpy as np
import numpy.typing as npt

from cliniverse.acquisition.catalogue import PanelCatalogue
from cliniverse.data.cohort import BoolArray, Cohort, FloatArray
from cliniverse.exceptions import BudgetError, ConfigError

IntArray = npt.NDArray[np.int64]


class Protocol(enum.StrEnum):
    """Which groups a policy is permitted to request."""

    SUPPORT_AWARE = "support_aware"
    SUPPORT_BLIND = "support_blind"


DEFAULT_EPOCH_HOURS: Final[tuple[int, ...]] = (12, 18, 24)


@dataclasses.dataclass(frozen=True, slots=True)
class PolicyView:
    """Everything a policy is allowed to see. Nothing here reveals the support.

    Deliberately absent: the hidden set, the historical observation mask, counts
    of hidden values, whether a gap is natural or synthetic, timestamps beyond
    the boundary, and every target.
    """

    disclosed_values: FloatArray
    disclosed_mask: BoolArray
    statics: FloatArray
    statics_mask: BoolArray
    epoch: int
    n_epochs: int
    boundary_hour: int
    spent: float
    remaining: float
    requestable: tuple[str, ...]
    variable_names: tuple[str, ...]
    static_names: tuple[str, ...]
    catalogue: PanelCatalogue

    @property
    def n_disclosed(self) -> int:
        return int(self.disclosed_mask.sum())

    def values_for(self, variable: str) -> FloatArray:
        """Disclosed series for one variable, ``NaN`` where not disclosed."""
        try:
            col = self.variable_names.index(variable)
        except ValueError as exc:
            raise ConfigError(f"unknown variable {variable!r}") from exc
        series: FloatArray = self.disclosed_values[:, col]
        return series


@dataclasses.dataclass(frozen=True, slots=True)
class Purchase:
    """The outcome of one request. Returned only *after* payment."""

    panel: str
    cost: float
    n_disclosed: int
    epoch: int

    @property
    def was_empty(self) -> bool:
        """True when the policy paid and received nothing.

        Not knowable in advance — that is the point of SO-2.
        """
        return self.n_disclosed == 0


class DisclosureEngine:
    """Per-patient disclosure state. Evaluator-side; never handed to a policy.

    Args:
        cohort: the full cohort (evaluator-side, holds the true values).
        patient: index of the patient this episode concerns.
        hidden: ``(T, V)`` mask of cells withheld from the policy.
        catalogue: the orderable groups and their costs.
        budget: total spend allowed across all epochs.
        protocol: which groups are requestable.
        epoch_hours: information boundaries, ascending.
    """

    def __init__(
        self,
        cohort: Cohort,
        patient: int,
        hidden: BoolArray,
        catalogue: PanelCatalogue,
        *,
        budget: float,
        protocol: Protocol = Protocol.SUPPORT_BLIND,
        epoch_hours: tuple[int, ...] = DEFAULT_EPOCH_HOURS,
    ) -> None:
        if budget < 0:
            raise BudgetError(f"budget must be non-negative, got {budget}")
        if not epoch_hours or list(epoch_hours) != sorted(epoch_hours):
            raise ConfigError(f"epoch_hours must be ascending and non-empty: {epoch_hours}")
        if epoch_hours[-1] > cohort.n_hours:
            raise ConfigError(
                f"final boundary {epoch_hours[-1]}h exceeds horizon {cohort.n_hours}h"
            )
        if hidden.shape != cohort.x.shape[1:]:
            raise ConfigError(
                f"hidden mask shape {hidden.shape} != (T, V) {cohort.x.shape[1:]}"
            )
        if bool(np.any(hidden & ~cohort.m[patient])):
            raise ConfigError("hidden mask marks a cell that was never observed")

        # --- evaluator-only state. Nothing below is exposed to a policy. ---
        self._values: FloatArray = cohort.x[patient].copy()
        self._observed: BoolArray = cohort.m[patient].copy()
        self._hidden: BoolArray = hidden.copy()
        self._statics: FloatArray = cohort.statics[patient].copy()
        self._statics_mask: BoolArray = cohort.statics_mask[patient].copy()
        self._variable_names = cohort.variable_names
        self._static_names = cohort.static_names

        self._catalogue = catalogue
        self._budget = float(budget)
        self._protocol = protocol
        self._epoch_hours = epoch_hours
        self._epoch = 0
        self._spent = 0.0
        self._log: list[Purchase] = []

        # Cells the policy can currently see: observed, not hidden, within boundary.
        self._revealed: BoolArray = self._observed & ~self._hidden

    # ------------------------------------------------------------- epochs ----
    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def n_epochs(self) -> int:
        return len(self._epoch_hours)

    @property
    def boundary_hour(self) -> int:
        """The information boundary for the current epoch, ``t_k``."""
        return self._epoch_hours[self._epoch]

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def remaining(self) -> float:
        return self._budget - self._spent

    @property
    def purchases(self) -> tuple[Purchase, ...]:
        return tuple(self._log)

    def advance_epoch(self) -> bool:
        """Move to the next epoch. Returns False when the last one is done."""
        if self._epoch + 1 >= self.n_epochs:
            return False
        self._epoch += 1
        return True

    # --------------------------------------------------------- boundaries ----
    def _within_boundary(self) -> BoolArray:
        """``(T, V)`` mask selecting hours at or before the current boundary."""
        gate = np.zeros_like(self._observed)
        gate[: self.boundary_hour, :] = True
        return gate

    def _panel_columns(self, panel_name: str) -> list[int]:
        panel = self._catalogue.panels.get(panel_name)
        if panel is None:
            raise ConfigError(f"unknown panel {panel_name!r}")
        index = {name: i for i, name in enumerate(self._variable_names)}
        return [index[m] for m in panel.members if m in index]

    def _has_hidden(self, panel_name: str) -> bool:
        """Evaluator-only. Never call this from policy-facing code."""
        cols = self._panel_columns(panel_name)
        if not cols:
            return False
        return bool((self._hidden[: self.boundary_hour, cols]).any())

    def requestable_panels(self) -> tuple[str, ...]:
        """Groups the policy may request under the active protocol.

        Under ``support_blind`` this is the whole catalogue and therefore carries
        no information. Under ``support_aware`` it is filtered by availability —
        which is precisely the leak that protocol is designed to reproduce.
        """
        if self._protocol is Protocol.SUPPORT_BLIND:
            return self._catalogue.panel_names
        return tuple(p for p in self._catalogue.panel_names if self._has_hidden(p))

    # ------------------------------------------------------------- policy ----
    def view(self) -> PolicyView:
        """Build the policy-visible state (SO-1: copies only, no back-reference)."""
        gate = self._within_boundary()
        visible = self._revealed & gate
        values = np.where(visible, self._values, np.nan).astype(np.float32)
        return PolicyView(
            disclosed_values=values,
            disclosed_mask=visible.copy(),
            statics=self._statics.copy(),
            statics_mask=self._statics_mask.copy(),
            epoch=self._epoch,
            n_epochs=self.n_epochs,
            boundary_hour=self.boundary_hour,
            spent=self._spent,
            remaining=self.remaining,
            requestable=self.requestable_panels(),
            variable_names=self._variable_names,
            static_names=self._static_names,
            catalogue=self._catalogue,
        )

    def request(self, panel_name: str) -> Purchase:
        """Request a group. Charged in full whether or not anything is disclosed.

        Raises:
            BudgetError: if the cost exceeds the remaining budget, or the group
                is not requestable under the active protocol.
        """
        if panel_name not in self._catalogue.panels:
            raise ConfigError(f"unknown panel {panel_name!r}")
        if panel_name not in self.requestable_panels():
            raise BudgetError(
                f"panel {panel_name!r} is not requestable under {self._protocol}"
            )
        cost = self._catalogue.cost_of(panel_name)
        if cost > self.remaining + 1e-9:
            raise BudgetError(
                f"panel {panel_name!r} costs {cost} but only {self.remaining} remains"
            )

        cols = self._panel_columns(panel_name)
        gate = self._within_boundary()
        to_disclose = np.zeros_like(self._hidden)
        if cols:
            block = np.zeros_like(self._hidden)
            block[:, cols] = True
            to_disclose = self._hidden & block & gate

        n = int(to_disclose.sum())
        # Charge first, then disclose: an empty result costs exactly the same.
        self._spent += cost
        self._revealed |= to_disclose
        self._hidden &= ~to_disclose

        purchase = Purchase(panel=panel_name, cost=cost, n_disclosed=n, epoch=self._epoch)
        self._log.append(purchase)
        return purchase

    # ---------------------------------------------------------- evaluator ----
    def evaluator_state(self) -> dict[str, object]:
        """Ground truth, for the evaluator only.

        Deliberately a separate method with an explicit name so that any call
        from policy code is obvious in review and in a diff.
        """
        return {
            "hidden_remaining": int(self._hidden.sum()),
            "observed_total": int(self._observed.sum()),
            "revealed_total": int(self._revealed.sum()),
            "spent": self._spent,
            "purchases": self.purchases,
        }
