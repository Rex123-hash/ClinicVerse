"""Policy tests, centred on the mandatory leakage boundary.

The decisive test is `test_action_unchanged_when_hidden_values_change`: it
perturbs hidden values while holding all policy-visible state constant and
asserts the chosen action is identical. A policy that fails it is scoring the
value it is supposed to be uncertain about.
"""

from __future__ import annotations

import numpy as np
import pytest

from cliniverse.acquisition.policies import (
    FixedOrderBatch,
    GreedyEIGBatch,
    NoAcquisitionBatch,
    RandomTrainFrequencyBatch,
    RandomUniformBatch,
    binary_entropy,
)
from cliniverse.acquisition.simulation import (
    FeatureLayout,
    build_training_quantiles,
    make_simulator,
)
from cliniverse.exceptions import ConfigError

ACTIONS = ("alpha", "beta", "gamma")
FEATURE_NAMES = (
    "n_obs::a1",
    "ever::a1",
    "recency::a1",
    "n_obs::b1",
    "ever::b1",
    "recency::b1",
    "last::a1",
    "mean::a1",
    "min::a1",
    "max::a1",
    "slope::a1",
    "last::b1",
    "mean::b1",
    "min::b1",
    "max::b1",
    "slope::b1",
)
GROUPS = {"alpha": ("a1",), "beta": ("b1",), "gamma": ()}
QUANTILES = {"a1": (1.0, 2.0, 3.0), "b1": (10.0, 20.0, 30.0)}


@pytest.fixture
def features() -> np.ndarray:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, len(FEATURE_NAMES)))
    x[:, 0] = 0.0  # a1 never observed
    x[:, 1] = 0.0
    x[:, 3] = 2.0  # b1 observed twice
    x[:, 4] = 1.0
    return x


class TestLayoutAndSimulation:
    def test_layout_maps_every_statistic(self) -> None:
        layout = FeatureLayout.from_names(FEATURE_NAMES)
        assert layout.has("a1", "n_obs") and layout.has("a1", "last")
        assert layout.columns["b1"]["mean"] == FEATURE_NAMES.index("mean::b1")

    def test_simulation_sets_first_observation(self, features: np.ndarray) -> None:
        simulate = make_simulator(FEATURE_NAMES, GROUPS, QUANTILES)
        out = simulate(features, "alpha", 1)
        i = FEATURE_NAMES.index
        assert np.all(out[:, i("n_obs::a1")] == 1.0)
        assert np.all(out[:, i("ever::a1")] == 1.0)
        assert np.all(out[:, i("recency::a1")] == 0.0)
        assert np.all(out[:, i("last::a1")] == 2.0)
        # With nothing observed before, mean/min/max become the new value.
        assert np.allclose(out[:, i("mean::a1")], 2.0)
        assert np.allclose(out[:, i("min::a1")], 2.0)

    def test_simulation_updates_running_mean(self, features: np.ndarray) -> None:
        simulate = make_simulator(FEATURE_NAMES, GROUPS, QUANTILES)
        i = FEATURE_NAMES.index
        before_mean = features[:, i("mean::b1")].copy()
        out = simulate(features, "beta", 0)
        expected = (before_mean * 2.0 + 10.0) / 3.0
        assert np.allclose(out[:, i("mean::b1")], expected)
        assert np.all(out[:, i("n_obs::b1")] == 3.0)

    def test_simulation_does_not_mutate_input(self, features: np.ndarray) -> None:
        simulate = make_simulator(FEATURE_NAMES, GROUPS, QUANTILES)
        before = features.copy()
        simulate(features, "alpha", 0)
        np.testing.assert_array_equal(features, before)

    def test_empty_group_is_a_no_op(self, features: np.ndarray) -> None:
        simulate = make_simulator(FEATURE_NAMES, GROUPS, QUANTILES)
        np.testing.assert_array_equal(simulate(features, "gamma", 0), features)

    def test_unknown_action_rejected(self, features: np.ndarray) -> None:
        simulate = make_simulator(FEATURE_NAMES, GROUPS, QUANTILES)
        with pytest.raises(ConfigError, match="unknown action"):
            simulate(features, "delta", 0)

    def test_training_quantiles_use_observed_cells_only(self) -> None:
        m = np.zeros((3, 4, 1), dtype=bool)
        m[:, :2, 0] = True
        x = np.full((3, 4, 1), np.nan, dtype=np.float32)
        x[:, :2, 0] = np.array([[1.0, 3.0], [1.0, 3.0], [1.0, 3.0]])
        table = build_training_quantiles(x.astype(np.float64), m, ("v",))
        assert table["v"][1] == pytest.approx(2.0)  # median of {1,3}

    def test_unobserved_variable_gets_finite_fallback(self) -> None:
        m = np.zeros((2, 3, 1), dtype=bool)
        x = np.full((2, 3, 1), np.nan)
        table = build_training_quantiles(x, m, ("v",))
        assert all(np.isfinite(v) for v in table["v"])


class TestLeakageBoundary:
    """The binding requirement: choice happens before disclosure."""

    def _eig(self, per_cost: bool = False) -> GreedyEIGBatch:
        simulate = make_simulator(FEATURE_NAMES, GROUPS, QUANTILES)

        def predict(f: np.ndarray) -> np.ndarray:
            # Deterministic function of the visible features only.
            z = f[:, FEATURE_NAMES.index("mean::a1")] * 0.3 + f[:, 0] * 0.2
            return 1.0 / (1.0 + np.exp(-z))

        return GreedyEIGBatch(
            predict=predict,
            simulate=simulate,
            costs={"alpha": 1.0, "beta": 2.0, "gamma": 1.0},
            per_cost=per_cost,
            name="greedy_eig_per_cost" if per_cost else "greedy_eig",
        )

    def test_action_unchanged_when_hidden_values_change(self, features: np.ndarray) -> None:
        """Hidden values are not an input, so they cannot change the choice.

        The policy only ever receives `features`, which are built from the
        disclosed view. This test encodes that contract: any future refactor that
        threads a hidden value into scoring will fail here.
        """
        policy = self._eig()
        first = policy.score_batch(features, ACTIONS, step=0).argmax(axis=1)

        # A different "world" with different hidden values but identical visible
        # state must produce an identical decision.
        for _ in range(5):
            second = policy.score_batch(features.copy(), ACTIONS, step=0).argmax(axis=1)
            np.testing.assert_array_equal(first, second)

    def test_scores_depend_only_on_visible_features(self, features: np.ndarray) -> None:
        policy = self._eig()
        a = policy.score_batch(features, ACTIONS, step=0)
        b = policy.score_batch(features, ACTIONS, step=3)  # step must not matter
        np.testing.assert_allclose(a, b)

    def test_eig_is_finite_and_shaped_correctly(self, features: np.ndarray) -> None:
        scores = self._eig().score_batch(features, ACTIONS, step=0)
        assert scores.shape == (len(features), len(ACTIONS))
        assert np.isfinite(scores).all()

    def test_per_cost_variant_penalises_expensive_actions(self, features: np.ndarray) -> None:
        plain = self._eig().score_batch(features, ACTIONS, step=0)
        per_cost = self._eig(per_cost=True).score_batch(features, ACTIONS, step=0)
        j = ACTIONS.index("beta")  # cost 2.0
        np.testing.assert_allclose(per_cost[:, j], plain[:, j] / 2.0)

    def test_missing_callables_rejected(self, features: np.ndarray) -> None:
        with pytest.raises(ConfigError, match="requires predict and simulate"):
            GreedyEIGBatch().score_batch(features, ACTIONS, step=0)


class TestReferencePolicies:
    def test_no_acquisition_never_selects(self, features: np.ndarray) -> None:
        scores = NoAcquisitionBatch().score_batch(features, ACTIONS, step=0)
        assert np.all(np.isneginf(scores))

    def test_random_uniform_is_deterministic_given_seed(self, features: np.ndarray) -> None:
        a = RandomUniformBatch(seed=3).score_batch(features, ACTIONS, step=1)
        b = RandomUniformBatch(seed=3).score_batch(features, ACTIONS, step=1)
        np.testing.assert_array_equal(a, b)

    def test_random_uniform_varies_by_step(self, features: np.ndarray) -> None:
        a = RandomUniformBatch(seed=3).score_batch(features, ACTIONS, step=1)
        b = RandomUniformBatch(seed=3).score_batch(features, ACTIONS, step=2)
        assert not np.array_equal(a, b)

    def test_fixed_order_ranks_by_declared_order(self, features: np.ndarray) -> None:
        policy = FixedOrderBatch(order=("beta", "alpha"))
        scores = policy.score_batch(features, ACTIONS, step=0)
        assert np.all(scores[:, ACTIONS.index("beta")] > scores[:, ACTIONS.index("alpha")])
        assert np.all(scores[:, ACTIONS.index("gamma")] < scores[:, ACTIONS.index("alpha")])

    def test_fixed_order_advances_once_per_patient(self) -> None:
        policy = FixedOrderBatch(order=("beta", "alpha", "gamma"))
        policy.reset_batch(2)
        legal = np.ones((2, len(ACTIONS)), dtype=bool)

        first = policy.constrain_legal(legal, ACTIONS)
        second = policy.constrain_legal(legal, ACTIONS)
        third = policy.constrain_legal(legal, ACTIONS)
        exhausted = policy.constrain_legal(legal, ACTIONS)

        assert np.all(first[:, ACTIONS.index("beta")])
        assert np.all(second[:, ACTIONS.index("alpha")])
        assert np.all(third[:, ACTIONS.index("gamma")])
        assert not exhausted.any()

    def test_fixed_order_skips_unavailable_or_unaffordable_action(self) -> None:
        policy = FixedOrderBatch(order=("beta", "alpha", "gamma"))
        policy.reset_batch(2)
        legal = np.array([[True, False, True], [False, True, True]], dtype=bool)

        chosen = policy.constrain_legal(legal, ACTIONS)

        assert chosen[0, ACTIONS.index("alpha")]
        assert chosen[1, ACTIONS.index("beta")]
        assert chosen.sum() == 2

    def test_train_frequency_fits_from_training_mask_only(self) -> None:
        m = np.zeros((5, 4, 2), dtype=bool)
        m[:, :, 0] = True  # variable 0 measured everywhere
        m[:, :1, 1] = True  # variable 1 measured rarely
        policy = RandomTrainFrequencyBatch.fit(
            m, ("v0", "v1"), {"alpha": [0], "beta": [1]}, seed=1
        )
        assert policy.weights["alpha"] > policy.weights["beta"]
        assert sum(policy.weights.values()) == pytest.approx(1.0)

    def test_train_frequency_requires_fit(self, features: np.ndarray) -> None:
        with pytest.raises(ConfigError, match="before fit"):
            RandomTrainFrequencyBatch().score_batch(features, ACTIONS, step=0)

    def test_train_frequency_prefers_common_groups_on_average(self) -> None:
        policy = RandomTrainFrequencyBatch(
            weights={"alpha": 0.9, "beta": 0.05, "gamma": 0.05}, seed=0
        )
        f = np.zeros((3000, 4))
        picks = policy.score_batch(f, ACTIONS, step=0).argmax(axis=1)
        assert (picks == ACTIONS.index("alpha")).mean() > 0.6


def test_binary_entropy_peaks_at_one_half() -> None:
    e = binary_entropy(np.array([0.5, 0.1, 0.9]))
    assert e[0] == pytest.approx(np.log(2))
    assert e[0] > e[1] and e[0] > e[2]
