"""The M4 evaluator: drives TwinBench disclosure engines, never replaces them.

Every state transition goes through the tested :class:`DisclosureEngine`. This
module only decides *which* action to ask for and gathers the resulting visible
state; it does not reimplement disclosure, costing, boundary enforcement or
budget accounting.

Two execution paths, identical in semantics:

``run_static``    for policies whose choice does not depend on the model
                  (no-acquisition, random, fixed order, support oracle). Episodes
                  run per patient through the engine.
``run_adaptive``  for the surrogate expected-information-gain policies, which
                  need a model prediction at every step. Patients advance in
                  lockstep so candidate scoring can be batched; the engine calls
                  are unchanged.

The lockstep path exists purely for speed. A patient's action sequence does not
depend on any other patient: scores are computed row-wise and each engine is
advanced independently.
"""

from __future__ import annotations

import dataclasses
import hashlib

import numpy as np
import numpy.typing as npt

from cliniverse.acquisition.policies import BatchPolicy
from cliniverse.data.cohort import Cohort
from cliniverse.exceptions import BudgetError, ConfigError
from twinbench.disclosure import DisclosureEngine

FloatArray = npt.NDArray[np.float64]

#: Hard stop on requests per epoch. A policy that keeps asking is a bad policy,
#: not an error, so the episode simply ends.
MAX_REQUESTS_PER_EPOCH = 16


@dataclasses.dataclass(slots=True)
class ActionRecord:
    """One request, recorded for the reproducible trace.

    Holds counts and hashes only — never a hidden value.
    """

    patient_index: int
    epoch: int
    step: int
    action: str
    cost: float
    success: bool
    n_disclosed: int
    remaining_after: float
    visible_state_hash: str

    def as_row(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def visible_state_hash(values: npt.NDArray[np.floating], mask: npt.NDArray[np.bool_]) -> str:
    """Short hash of the policy-visible state, for trace reproducibility."""
    h = hashlib.sha256()
    h.update(np.nan_to_num(values, nan=-9999.0).astype(np.float32).tobytes())
    h.update(mask.tobytes())
    return h.hexdigest()[:16]


def gather_disclosed(engines: list[DisclosureEngine], template: Cohort) -> Cohort:
    """Assemble a cohort from what the engines currently disclose.

    Authoritative by construction: the arrays come from ``engine.view()``, so the
    features can never see something the policy could not.
    """
    n = len(engines)
    x = np.empty((n, template.n_hours, template.n_variables), dtype=np.float32)
    m = np.empty((n, template.n_hours, template.n_variables), dtype=bool)
    for i, engine in enumerate(engines):
        view = engine.view()
        x[i] = view.disclosed_values
        m[i] = view.disclosed_mask
    return dataclasses.replace(template, x=x, m=m)


def _affordable_mask(
    engines: list[DisclosureEngine], actions: tuple[str, ...]
) -> npt.NDArray[np.bool_]:
    """``(n_patients, n_actions)`` legality mask under the active protocol."""
    legal = np.zeros((len(engines), len(actions)), dtype=bool)
    if not engines:
        return legal
    # Costs are identical across patients, so price the catalogue once rather
    # than materialising a PolicyView per patient purely to read a cost.
    costs = np.array(
        [engines[0].view().catalogue.cost_of(a) for a in actions], dtype=np.float64
    )
    for i, engine in enumerate(engines):
        requestable = set(engine.requestable_panels())
        remaining = engine.remaining
        for j, action in enumerate(actions):
            if action in requestable and costs[j] <= remaining + 1e-9:
                legal[i, j] = True
    return legal


def run_adaptive(
    engines: list[DisclosureEngine],
    policy: BatchPolicy,
    template: Cohort,
    build_features: object,
    *,
    collect_trace: bool = False,
) -> tuple[Cohort, list[ActionRecord]]:
    """Advance every patient in lockstep, batching only the scoring.

    Returns the final disclosed cohort and, optionally, the action trace.
    """
    if not engines:
        raise ConfigError("no engines to run")
    actions = engines[0].view().catalogue.panel_names
    trace: list[ActionRecord] = []
    build = build_features  # Callable[[Cohort], FloatArray]

    while True:
        for step in range(MAX_REQUESTS_PER_EPOCH):
            legal = _affordable_mask(engines, actions)
            if not legal.any():
                break

            disclosed = gather_disclosed(engines, template)
            features = np.asarray(build(disclosed), dtype=np.float64)  # type: ignore[operator]
            scores = policy.score_batch(features, actions, step)
            scores = np.where(legal, scores, -np.inf)

            acted = False
            for i, engine in enumerate(engines):
                row = scores[i]
                if not np.isfinite(row).any():
                    continue
                choice = actions[int(np.argmax(row))]
                view_before = engine.view()
                try:
                    purchase = engine.request(choice)
                except (BudgetError, ConfigError):
                    continue
                acted = True
                if collect_trace:
                    trace.append(
                        ActionRecord(
                            patient_index=engine.patient_index,
                            epoch=engine.epoch,
                            step=step,
                            action=choice,
                            cost=purchase.cost,
                            success=not purchase.was_empty,
                            n_disclosed=purchase.n_disclosed,
                            remaining_after=engine.remaining,
                            visible_state_hash=visible_state_hash(
                                view_before.disclosed_values, view_before.disclosed_mask
                            ),
                        )
                    )
            if not acted:
                break

        # Every engine must advance. `any(... for ...)` would short-circuit and
        # leave all but the first patient frozen at the first boundary.
        advanced = [engine.advance_epoch() for engine in engines]
        if not any(advanced):
            break

    return gather_disclosed(engines, template), trace
