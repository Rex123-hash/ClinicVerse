"""Running a policy through one patient's decision epochs.

The runner is deliberately thin. It owns the loop and the stopping rules; it
never inspects the engine's private state, and it hands the policy nothing but a
:class:`~twinbench.disclosure.PolicyView`. Keeping it small is what makes the
support-oracle invariants auditable — there is only one place where a policy
could be handed something it should not see, and it is here.

Policies implement :class:`Policy`: given a view, return the name of a group to
request, or ``None`` to stop. Returning a group that is unaffordable or not
requestable ends the episode rather than raising, so that a badly-behaved policy
scores poorly instead of crashing the benchmark.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Protocol as TypingProtocol
from typing import runtime_checkable

import numpy as np

from cliniverse.acquisition.catalogue import PanelCatalogue
from cliniverse.data.cohort import BoolArray
from cliniverse.exceptions import BudgetError, ConfigError
from twinbench.disclosure import DisclosureEngine, PolicyView, Purchase


@runtime_checkable
class Policy(TypingProtocol):
    """An acquisition policy.

    Implementations must be pure functions of the view plus their own internal
    state. They must not import or hold benchmark internals.
    """

    name: str

    def select(self, view: PolicyView) -> object:
        """Return a group name or ``None``; the runner validates runtime output."""
        ...

    def reset(self) -> None:
        """Clear per-episode state. Called once before each episode."""
        ...


class TerminationReason(enum.StrEnum):
    """Explicit reason an evaluator stopped an episode."""

    POLICY_STOP = "policy_stop"
    BUDGET_EXHAUSTED = "budget_exhausted"
    REQUEST_LIMIT = "request_limit"
    MALFORMED_ACTION = "malformed_action"
    UNKNOWN_ACTION = "unknown_action"
    UNAVAILABLE_ACTION = "unavailable_action"
    UNAFFORDABLE_ACTION = "unaffordable_action"


@dataclasses.dataclass(frozen=True, slots=True)
class EpisodeTrace:
    """What happened during one episode. Evaluator-side record."""

    policy: str
    patient_index: int
    protocol: str
    budget: float
    spent: float
    purchases: tuple[Purchase, ...]
    final_view: PolicyView
    termination: TerminationReason

    @property
    def n_requests(self) -> int:
        return len(self.purchases)

    @property
    def n_empty_requests(self) -> int:
        """Requests that cost budget and disclosed nothing.

        Under ``support_blind`` this is the visible price of not knowing what is
        available; under ``support_aware`` it should be zero by construction.
        """
        return sum(1 for p in self.purchases if p.was_empty)

    @property
    def n_disclosed(self) -> int:
        return sum(p.n_disclosed for p in self.purchases)

    @property
    def wasted_spend(self) -> float:
        return sum(p.cost for p in self.purchases if p.was_empty)


def run_episode(
    engine: DisclosureEngine,
    policy: Policy,
    *,
    max_requests_per_epoch: int = 32,
) -> EpisodeTrace:
    """Run ``policy`` against ``engine`` across all epochs.

    The loop advances an epoch when the policy stops requesting, and ends when
    the budget is exhausted or the last epoch closes.

    Args:
        max_requests_per_epoch: guard against a policy that never returns
            ``None``. Reaching it ends the epoch, and is not an error — a policy
            that spends without stopping is simply a bad policy.
    """
    if max_requests_per_epoch < 1:
        raise ConfigError("max_requests_per_epoch must be >= 1")

    policy.reset()
    termination = TerminationReason.POLICY_STOP
    terminate = False
    while True:
        for _ in range(max_requests_per_epoch):
            view = engine.view()
            if view.remaining <= 0:
                termination = TerminationReason.BUDGET_EXHAUSTED
                terminate = True
                break
            choice = policy.select(view)
            if choice is None:
                break
            if not isinstance(choice, str):
                termination = TerminationReason.MALFORMED_ACTION
                terminate = True
                break
            if choice not in view.catalogue.panel_names:
                termination = TerminationReason.UNKNOWN_ACTION
                terminate = True
                break
            if choice not in view.requestable:
                termination = TerminationReason.UNAVAILABLE_ACTION
                terminate = True
                break
            if view.catalogue.cost_of(choice) > view.remaining + 1e-9:
                termination = TerminationReason.UNAFFORDABLE_ACTION
                terminate = True
                break
            try:
                engine.request(choice)
            except (BudgetError, ConfigError):
                # Defensive fallback: all built-in rejection paths are classified
                # above from the policy-visible state.
                termination = TerminationReason.UNAVAILABLE_ACTION
                terminate = True
                break
            if engine.remaining <= 0:
                termination = TerminationReason.BUDGET_EXHAUSTED
                terminate = True
                break
        else:
            termination = TerminationReason.REQUEST_LIMIT
            terminate = True
        if terminate:
            break
        if not engine.advance_epoch():
            break

    return EpisodeTrace(
        policy=policy.name,
        patient_index=engine.patient_index,
        protocol=str(engine.protocol),
        budget=engine.budget,
        spent=engine.spent,
        purchases=engine.purchases,
        final_view=engine.view(),
        termination=termination,
    )


# --------------------------------------------------------------- policies ----
@dataclasses.dataclass
class NoAcquisition:
    """Spends nothing. The zero-budget reference point on every curve."""

    name: str = "no_acquisition"

    def select(self, view: PolicyView) -> str | None:
        del view
        return None

    def reset(self) -> None:
        return None


@dataclasses.dataclass
class RandomUniformAll:
    """Support-blind uniform random over all affordable legal action types."""

    seed: int = 0
    name: str = "random_uniform_all"
    _rng: np.random.Generator = dataclasses.field(
        default_factory=lambda: np.random.default_rng(0), repr=False
    )
    _bought: set[str] = dataclasses.field(default_factory=set, repr=False)

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._bought = set()

    def select(self, view: PolicyView) -> str | None:
        if view.requestable != view.catalogue.panel_names:
            raise ConfigError("random_uniform_all requires the support_blind protocol")
        options = [
            p
            for p in view.requestable
            if p not in self._bought and view.catalogue.cost_of(p) <= view.remaining
        ]
        if not options:
            return None
        choice = str(self._rng.choice(sorted(options)))
        self._bought.add(choice)
        return choice


@dataclasses.dataclass
class RandomSupportOracle:
    """Diagnostic oracle sampling only patient-specific available actions.

    This baseline consumes the availability-filtered action list exposed by the
    support-aware protocol. It is not deployable and is not a fair support-blind
    comparator.
    """

    seed: int = 0
    name: str = "random_support_oracle"
    _rng: np.random.Generator = dataclasses.field(
        default_factory=lambda: np.random.default_rng(0), repr=False
    )
    _bought: set[str] = dataclasses.field(default_factory=set, repr=False)

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._bought = set()

    def select(self, view: PolicyView) -> str | None:
        options = [
            name
            for name in view.requestable
            if name not in self._bought and view.catalogue.cost_of(name) <= view.remaining
        ]
        if not options:
            return None
        choice = str(self._rng.choice(sorted(options)))
        self._bought.add(choice)
        return choice


@dataclasses.dataclass
class RandomTrainFrequency:
    """Support-blind random weighted by feature-group frequency in training data.

    Frequencies are fixed before evaluation and contain no patient-specific
    availability. ``fit`` counts training patient-hours in which any member of a
    group was observed and applies additive smoothing.
    """

    weights: tuple[tuple[str, float], ...]
    seed: int = 0
    name: str = "random_train_frequency"
    _rng: np.random.Generator = dataclasses.field(
        default_factory=lambda: np.random.default_rng(0), repr=False
    )
    _bought: set[str] = dataclasses.field(default_factory=set, repr=False)

    @classmethod
    def fit(
        cls,
        training_mask: BoolArray,
        variable_names: tuple[str, ...],
        catalogue: PanelCatalogue,
        *,
        seed: int = 0,
        smoothing: float = 1.0,
    ) -> RandomTrainFrequency:
        """Fit action weights from a training mask only."""
        if training_mask.ndim != 3:
            raise ConfigError(
                f"training_mask must have shape (N, T, V), got {training_mask.shape}"
            )
        if not np.isfinite(smoothing) or smoothing <= 0:
            raise ConfigError("smoothing must be finite and > 0")
        index = {name: i for i, name in enumerate(variable_names)}
        weights: list[tuple[str, float]] = []
        for name in catalogue.panel_names:
            cols = [
                index[member] for member in catalogue.panels[name].members if member in index
            ]
            count = float(training_mask[:, :, cols].any(axis=2).sum()) if cols else 0.0
            weights.append((name, count + smoothing))
        return cls(weights=tuple(weights), seed=seed)

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._bought = set()

    def select(self, view: PolicyView) -> str | None:
        if view.requestable != view.catalogue.panel_names:
            raise ConfigError("random_train_frequency requires the support_blind protocol")
        weight_map = dict(self.weights)
        if set(weight_map) != set(view.catalogue.panel_names):
            raise ConfigError("training frequencies do not match the action catalogue")
        options = [
            name
            for name in view.catalogue.panel_names
            if name not in self._bought and view.catalogue.cost_of(name) <= view.remaining
        ]
        if not options:
            return None
        probabilities = np.asarray([weight_map[name] for name in options], dtype=np.float64)
        probabilities /= probabilities.sum()
        choice = str(self._rng.choice(options, p=probabilities))
        self._bought.add(choice)
        return choice


@dataclasses.dataclass
class FixedOrder:
    """Requests groups in a fixed, declared order.

    Authored by the engineering team from observed measurement frequency. It is
    **not** clinician-designed or clinician-validated, and must not be described
    as a clinical heuristic.
    """

    order: tuple[str, ...]
    name: str = "fixed_order"
    _position: int = dataclasses.field(default=0, repr=False)

    def reset(self) -> None:
        self._position = 0

    def select(self, view: PolicyView) -> str | None:
        while self._position < len(self.order):
            candidate = self.order[self._position]
            self._position += 1
            if (
                candidate in view.requestable
                and view.catalogue.cost_of(candidate) <= view.remaining
            ):
                return candidate
        return None
