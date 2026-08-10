"""Acquisition policies scored in batch, with engine semantics untouched.

Every policy here implements ``score_batch``: given the current feature matrix
for a set of patients and their policy-visible views, return a score per
(patient, action). The evaluator picks ``argmax`` among affordable actions and
then calls the *tested* ``DisclosureEngine.request`` to perform the transition.
Batching therefore speeds up scoring only; it never reimplements disclosure.

**Leakage boundary.** Scores are computed from the policy-visible view and from
statistics fitted on training folds. No policy sees hidden values, the hidden
support, availability, or the label. `GreedyEIG` in particular integrates over a
*predicted* distribution for the unknown value; it never evaluates the utility of
the value that is actually hidden.
"""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from cliniverse.exceptions import ConfigError

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

#: Batched model predictor: design matrix -> probability per row.
PredictFn = Callable[[FloatArray], FloatArray]
#: Counterfactual completion: (features, action, quantile index) -> features.
SimulateFn = Callable[[FloatArray, str, int], FloatArray]

_EPS = 1e-12


def binary_entropy(p: FloatArray) -> FloatArray:
    q = np.clip(p, _EPS, 1 - _EPS)
    return np.asarray(-(q * np.log(q) + (1 - q) * np.log1p(-q)), dtype=np.float64)


@dataclasses.dataclass
class BatchPolicy(abc.ABC):
    """A policy that scores every action for every patient at once."""

    name: str

    @abc.abstractmethod
    def score_batch(
        self, features: FloatArray, actions: tuple[str, ...], step: int
    ) -> FloatArray:
        """Return an ``(n_patients, n_actions)`` score matrix. Higher is chosen."""

    def config(self) -> dict[str, object]:
        return {"name": self.name}


@dataclasses.dataclass
class NoAcquisitionBatch(BatchPolicy):
    """Never acquires. The budget-zero reference."""

    name: str = "no_acquisition"

    def score_batch(
        self, features: FloatArray, actions: tuple[str, ...], step: int
    ) -> FloatArray:
        del step
        return np.full((features.shape[0], len(actions)), -np.inf)


@dataclasses.dataclass
class RandomUniformBatch(BatchPolicy):
    """Uniform over all legal actions. Support-blind."""

    seed: int = 0
    name: str = "random_uniform_all"

    def score_batch(
        self, features: FloatArray, actions: tuple[str, ...], step: int
    ) -> FloatArray:
        rng = np.random.default_rng(self.seed + 7919 * step)
        return rng.random((features.shape[0], len(actions)))


@dataclasses.dataclass
class RandomTrainFrequencyBatch(BatchPolicy):
    """Samples actions in proportion to training-fold measurement frequency.

    The serious random comparator: it knows which groups are commonly measured
    *in the training data*, but nothing patient-specific and nothing about
    availability.
    """

    weights: dict[str, float] = dataclasses.field(default_factory=dict)
    seed: int = 0
    name: str = "random_train_frequency"

    @classmethod
    def fit(
        cls,
        training_mask: BoolArray,
        variable_names: tuple[str, ...],
        groups: dict[str, list[int]],
        *,
        seed: int = 0,
        smoothing: float = 1.0,
    ) -> RandomTrainFrequencyBatch:
        del variable_names
        weights: dict[str, float] = {}
        for name, cols in groups.items():
            weights[name] = float(training_mask[:, :, cols].sum()) + smoothing
        total = sum(weights.values())
        return cls(weights={k: v / total for k, v in weights.items()}, seed=seed)

    def score_batch(
        self, features: FloatArray, actions: tuple[str, ...], step: int
    ) -> FloatArray:
        if not self.weights:
            raise ConfigError("RandomTrainFrequencyBatch used before fit()")
        rng = np.random.default_rng(self.seed + 6271 * step)
        w = np.array([self.weights.get(a, 0.0) for a in actions], dtype=np.float64)
        w = np.clip(w, 1e-9, None)
        # Gumbel-top-1 sampling gives a weighted draw via a score matrix.
        gumbel = rng.gumbel(size=(features.shape[0], len(actions)))
        return np.log(w)[None, :] + gumbel

    def config(self) -> dict[str, object]:
        return {"name": self.name, "weights": dict(self.weights), "seed": self.seed}


@dataclasses.dataclass
class FixedOrderBatch(BatchPolicy):
    """A predeclared deterministic order. Domain-motivated, not clinician-derived."""

    order: tuple[str, ...] = ()
    name: str = "fixed_domain_order"

    def score_batch(
        self, features: FloatArray, actions: tuple[str, ...], step: int
    ) -> FloatArray:
        del step
        rank = {a: len(self.order) - i for i, a in enumerate(self.order)}
        scores = np.array([rank.get(a, -1.0) for a in actions], dtype=np.float64)
        return np.tile(scores, (features.shape[0], 1))

    def config(self) -> dict[str, object]:
        return {"name": self.name, "order": list(self.order)}


@dataclasses.dataclass
class GreedyEIGBatch(BatchPolicy):
    """Myopic expected information gain about the outcome.

    For each action the policy integrates over a predictive distribution for the
    unknown hidden value, using a fixed three-point quadrature at the training
    25th/50th/75th percentiles of each member variable with equal weights:

        EIG(a) = H(p_now) - mean_k H(p_k)

    where ``p_k`` is the model's outcome probability under the k-th imputed
    completion. Quantiles come from training folds only, and the hidden value is
    never consulted — the choice is made before any disclosure.

    ``per_cost`` divides the score by the action's cost. That variant is the
    closest justified analogue to a cost-sensitive greedy comparator; it is **not**
    a reproduction of Yu et al.
    """

    predict: PredictFn | None = None
    simulate: SimulateFn | None = None
    quantile_index: tuple[int, ...] = (0, 1, 2)
    costs: dict[str, float] = dataclasses.field(default_factory=dict)
    per_cost: bool = False
    name: str = "greedy_eig"

    def score_batch(
        self, features: FloatArray, actions: tuple[str, ...], step: int
    ) -> FloatArray:
        del step
        if self.predict is None or self.simulate is None:
            raise ConfigError("GreedyEIGBatch requires predict and simulate callables")
        predict = self.predict
        simulate = self.simulate

        p_now = np.asarray(predict(features), dtype=np.float64)
        h_now = binary_entropy(p_now)

        scores = np.zeros((features.shape[0], len(actions)), dtype=np.float64)
        for j, action in enumerate(actions):
            expected_h = np.zeros(features.shape[0], dtype=np.float64)
            for q in self.quantile_index:
                simulated = np.asarray(simulate(features, action, q), dtype=np.float64)
                p_k = np.asarray(predict(simulated), dtype=np.float64)
                expected_h += binary_entropy(p_k)
            expected_h /= len(self.quantile_index)
            gain = h_now - expected_h
            if self.per_cost:
                cost = max(self.costs.get(action, 1.0), 1e-9)
                gain = gain / cost
            scores[:, j] = gain
        return scores

    def config(self) -> dict[str, object]:
        return {
            "name": self.name,
            "per_cost": self.per_cost,
            "n_quadrature_points": len(self.quantile_index),
            "quadrature": "train 25th/50th/75th percentiles, equal weights",
        }
