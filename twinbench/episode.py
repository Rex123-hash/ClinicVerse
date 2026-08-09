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
from typing import Protocol as TypingProtocol
from typing import runtime_checkable

import numpy as np

from cliniverse.exceptions import BudgetError, ConfigError
from twinbench.disclosure import DisclosureEngine, PolicyView, Purchase


@runtime_checkable
class Policy(TypingProtocol):
    """An acquisition policy.

    Implementations must be pure functions of the view plus their own internal
    state. They must not import or hold benchmark internals.
    """

    name: str

    def select(self, view: PolicyView) -> str | None:
        """Return the group to request now, or ``None`` to stop this epoch."""
        ...

    def reset(self) -> None:
        """Clear per-episode state. Called once before each episode."""
        ...


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
    while True:
        for _ in range(max_requests_per_epoch):
            view = engine.view()
            if view.remaining <= 0:
                break
            choice = policy.select(view)
            if choice is None:
                break
            try:
                engine.request(choice)
            except (BudgetError, ConfigError):
                # An unaffordable or disallowed choice ends the episode. The
                # policy is penalised by the spend it already made, not by an
                # exception that would abort the whole run.
                break
        if engine.remaining <= 0 or not engine.advance_epoch():
            break

    return EpisodeTrace(
        policy=policy.name,
        patient_index=engine.patient_index,
        protocol=str(engine.protocol),
        budget=engine.budget,
        spent=engine.spent,
        purchases=engine.purchases,
        final_view=engine.view(),
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
class RandomPolicy:
    """Uniformly samples an affordable group it has not already bought.

    The honest floor for any acquisition method: anything that cannot beat this
    is not selecting information, it is just spending budget.
    """

    seed: int = 0
    name: str = "random"
    _rng: np.random.Generator = dataclasses.field(
        default_factory=lambda: np.random.default_rng(0), repr=False
    )
    _bought: set[str] = dataclasses.field(default_factory=set, repr=False)

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._bought = set()

    def select(self, view: PolicyView) -> str | None:
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
