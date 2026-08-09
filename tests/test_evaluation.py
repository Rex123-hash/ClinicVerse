"""Tests for the M2 evaluation layer: metrics, representations, imputation, artifacts.

The imputation and representation paths are new preprocessing, so they carry
regression tests asserting the properties the M2 contract depends on: values-only
really excludes presence features, imputers never see validation rows, and paired
inference reuses identical resamples.
"""

from __future__ import annotations

import numpy as np
import pytest

from cliniverse.data.cohort import Cohort
from cliniverse.evaluation.artifacts import cohort_fingerprint, split_hash, stable_hash
from cliniverse.evaluation.metrics import (
    METRIC_FUNCTIONS,
    auroc,
    bootstrap_metric,
    brier_score,
    calibration_intercept,
    calibration_slope,
    classification_metrics,
    negative_log_likelihood,
    paired_bootstrap_difference,
    reliability_curve,
)
from cliniverse.evaluation.representations import (
    CORE_REPRESENTATIONS,
    FittedImputer,
    ImputationStrategy,
    Representation,
    build_representation,
)
from cliniverse.exceptions import ConfigError


@pytest.fixture
def scored() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.2, size=500).astype(float)
    # A predictor correlated with the label, plus noise.
    p = np.clip(0.2 + 0.4 * y + rng.normal(0, 0.15, size=500), 0.01, 0.99)
    return y, p


class TestMetricValidation:
    def test_shape_mismatch_rejected(self) -> None:
        with pytest.raises(ConfigError, match="differ in shape"):
            auroc(np.array([0.0, 1.0]), np.array([0.5]))

    def test_non_finite_predictions_rejected(self) -> None:
        with pytest.raises(ConfigError, match="non-finite"):
            brier_score(np.array([0.0, 1.0]), np.array([0.5, np.nan]))

    def test_non_binary_labels_rejected(self) -> None:
        with pytest.raises(ConfigError, match="binary"):
            auroc(np.array([0.0, 2.0]), np.array([0.5, 0.5]))

    def test_empty_rejected(self) -> None:
        with pytest.raises(ConfigError, match="empty"):
            auroc(np.array([]), np.array([]))

    def test_nll_survives_extreme_predictions(self) -> None:
        """A single confident mistake must not produce infinity."""
        value = negative_log_likelihood(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
        assert np.isfinite(value)


class TestMetricCorrectness:
    def test_perfect_predictor(self) -> None:
        y = np.array([0.0, 0.0, 1.0, 1.0])
        p = np.array([0.01, 0.02, 0.98, 0.99])
        assert auroc(y, p) == pytest.approx(1.0)

    def test_brier_of_constant_prevalence(self) -> None:
        y = np.array([0.0, 0.0, 0.0, 1.0])
        p = np.full(4, 0.25)
        assert brier_score(y, p) == pytest.approx(0.1875)

    def test_calibration_of_a_perfectly_calibrated_predictor(self) -> None:
        """Labels drawn from the predicted probability give slope ~1, intercept ~0."""
        rng = np.random.default_rng(3)
        p = rng.uniform(0.05, 0.95, size=20000)
        y = rng.binomial(1, p).astype(float)
        assert calibration_slope(y, p) == pytest.approx(1.0, abs=0.08)
        assert calibration_intercept(y, p) == pytest.approx(0.0, abs=0.08)

    def test_overconfident_predictor_has_slope_below_one(self) -> None:
        rng = np.random.default_rng(4)
        p_true = rng.uniform(0.1, 0.9, size=20000)
        y = rng.binomial(1, p_true).astype(float)
        # Push probabilities toward the extremes: same ranking, overconfident.
        sharpened = np.clip(0.5 + (p_true - 0.5) * 2.5, 0.001, 0.999)
        assert calibration_slope(y, sharpened) < 0.9

    def test_metrics_bundle_reports_prevalence_and_n(
        self, scored: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y, p = scored
        m = classification_metrics(y, p)
        assert m.n == len(y)
        assert m.prevalence == pytest.approx(float(y.mean()))


class TestReliability:
    def test_bins_are_equal_mass_and_cover_everyone(
        self, scored: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y, p = scored
        curve = reliability_curve(y, p, n_bins=10)
        assert sum(curve["count"]) == len(y)
        counts = np.array(curve["count"])
        assert counts.max() - counts.min() <= 0.25 * counts.mean() + 5

    def test_constant_predictor_degrades_gracefully(self) -> None:
        y = np.array([0.0, 1.0, 0.0, 1.0])
        curve = reliability_curve(y, np.full(4, 0.5))
        assert curve["mean_predicted"] == []

    def test_too_few_bins_rejected(self, scored: tuple[np.ndarray, np.ndarray]) -> None:
        y, p = scored
        with pytest.raises(ConfigError, match="n_bins"):
            reliability_curve(y, p, n_bins=1)


class TestBootstrap:
    def test_interval_brackets_point_estimate(
        self, scored: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y, p = scored
        interval = bootstrap_metric(y, p, auroc, n_boot=300, seed=1)
        assert interval.low <= interval.point <= interval.high

    def test_bootstrap_is_deterministic(self, scored: tuple[np.ndarray, np.ndarray]) -> None:
        y, p = scored
        a = bootstrap_metric(y, p, auroc, n_boot=200, seed=5)
        b = bootstrap_metric(y, p, auroc, n_boot=200, seed=5)
        assert (a.point, a.low, a.high) == (b.point, b.low, b.high)

    def test_paired_difference_of_identical_predictors_is_exactly_zero(
        self, scored: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """The whole point of pairing: shared variance must cancel completely."""
        y, p = scored
        diff = paired_bootstrap_difference(
            y,
            p,
            p,
            auroc,
            metric_name="auroc",
            name_a="a",
            name_b="b",
            n_boot=200,
            seed=2,
        )
        assert diff.difference == pytest.approx(0.0)
        assert diff.low == pytest.approx(0.0)
        assert diff.high == pytest.approx(0.0)
        assert not diff.excludes_zero

    def test_paired_interval_is_tighter_than_independent_intervals(
        self, scored: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Paired inference is used precisely because it is more sensitive."""
        y, p = scored
        rng = np.random.default_rng(9)
        p2 = np.clip(p + rng.normal(0, 0.01, size=len(p)), 0.01, 0.99)

        paired = paired_bootstrap_difference(
            y,
            p,
            p2,
            auroc,
            metric_name="auroc",
            name_a="a",
            name_b="b",
            n_boot=400,
            seed=3,
        )
        a = bootstrap_metric(y, p, auroc, n_boot=400, seed=3)
        b = bootstrap_metric(y, p2, auroc, n_boot=400, seed=3)
        paired_width = paired.high - paired.low
        naive_width = (a.high - a.low) + (b.high - b.low)
        assert paired_width < naive_width

    def test_direction_convention_is_b_minus_a(self) -> None:
        y = np.array([0.0, 0.0, 1.0, 1.0])
        weak = np.array([0.4, 0.6, 0.4, 0.6])
        strong = np.array([0.1, 0.2, 0.8, 0.9])
        diff = paired_bootstrap_difference(
            y,
            weak,
            strong,
            auroc,
            metric_name="auroc",
            name_a="weak",
            name_b="strong",
            n_boot=100,
            seed=1,
        )
        assert diff.difference > 0

    def test_every_registered_metric_runs(self, scored: tuple[np.ndarray, np.ndarray]) -> None:
        y, p = scored
        for name, fn in METRIC_FUNCTIONS.items():
            assert np.isfinite(fn(y, p)), name


class TestImputer:
    def test_median_imputation_uses_training_medians_only(self) -> None:
        train = np.array([[1.0], [3.0], [5.0]])
        imputer = FittedImputer.fit(train, strategy=ImputationStrategy.MEDIAN)
        out = imputer.transform(np.array([[np.nan], [100.0]]), draw_seed=0)
        assert out[0, 0] == pytest.approx(3.0)
        assert out[1, 0] == pytest.approx(100.0)

    def test_imputer_never_sees_validation_rows(self) -> None:
        """A wild validation value must not shift the imputed constant."""
        train = np.array([[1.0], [1.0], [1.0]])
        imputer = FittedImputer.fit(train, strategy=ImputationStrategy.MEDIAN)
        out = imputer.transform(np.array([[np.nan], [1e6]]), draw_seed=0)
        assert out[0, 0] == pytest.approx(1.0)

    def test_stochastic_imputation_draws_from_the_training_pool(self) -> None:
        train = np.array([[1.0], [2.0], [3.0]])
        imputer = FittedImputer.fit(train, strategy=ImputationStrategy.STOCHASTIC, seed=7)
        out = imputer.transform(np.full((200, 1), np.nan), draw_seed=0)
        assert set(np.unique(out[:, 0]).tolist()) <= {1.0, 2.0, 3.0}

    def test_stochastic_imputation_is_not_constant(self) -> None:
        """This is the property that removes the 'exactly the median' signature."""
        train = np.array([[1.0], [2.0], [3.0], [4.0]])
        imputer = FittedImputer.fit(train, strategy=ImputationStrategy.STOCHASTIC, seed=7)
        out = imputer.transform(np.full((200, 1), np.nan), draw_seed=0)
        assert len(np.unique(out[:, 0])) > 1

    def test_stochastic_imputation_is_deterministic_given_seeds(self) -> None:
        train = np.array([[1.0], [2.0], [3.0]])
        imputer = FittedImputer.fit(train, strategy=ImputationStrategy.STOCHASTIC, seed=7)
        a = imputer.transform(np.full((50, 1), np.nan), draw_seed=4)
        b = imputer.transform(np.full((50, 1), np.nan), draw_seed=4)
        np.testing.assert_array_equal(a, b)

    def test_all_nan_training_column_falls_back_without_crashing(self) -> None:
        train = np.array([[np.nan], [np.nan]])
        imputer = FittedImputer.fit(train, strategy=ImputationStrategy.MEDIAN)
        out = imputer.transform(np.array([[np.nan]]), draw_seed=0)
        assert np.isfinite(out).all()

    def test_output_has_no_missing_values(self) -> None:
        rng = np.random.default_rng(1)
        train = rng.normal(size=(50, 4))
        train[rng.random(train.shape) < 0.3] = np.nan
        imputer = FittedImputer.fit(train, strategy=ImputationStrategy.MEDIAN)
        assert np.isfinite(imputer.transform(train, draw_seed=0)).all()

    def test_rejects_non_2d(self) -> None:
        with pytest.raises(ConfigError, match="2-D"):
            FittedImputer.fit(np.zeros((2, 2, 2)))


class TestRepresentations:
    @pytest.fixture
    def small(self) -> Cohort:
        n, t, v = 6, 4, 3
        m = np.zeros((n, t, v), dtype=bool)
        m[:, 0, 0] = True
        m[:3, 1, 1] = True
        x = np.where(m, 1.0, np.nan).astype(np.float32)
        return Cohort(
            record_ids=np.arange(n, dtype=np.int64),
            source_set=np.array(["a"] * n, dtype=np.str_),
            x=x,
            m=m,
            statics=np.zeros((n, 2), dtype=np.float32),
            statics_mask=np.ones((n, 2), dtype=bool),
            labels={"mortality": np.array([0, 1] * 3, dtype=np.float32)},
            variable_names=("v1", "v2", "v3"),
            static_names=("Age", "Gender"),
        )

    def test_values_only_contains_no_presence_features(self, small: Cohort) -> None:
        """The binding property of the M2 contract."""
        view = build_representation(small, Representation.VALUES_ONLY)
        assert not view.contains_presence_features()

    def test_mask_only_contains_only_presence_features(self, small: Cohort) -> None:
        view = build_representation(small, Representation.MASK_ONLY)
        assert view.contains_presence_features()
        assert all(
            n.startswith(("n_obs::", "ever::", "recency::", "n_distinct_vars::"))
            for n in view.names
        )

    def test_core_representations_exclude_statics(self, small: Cohort) -> None:
        """Statics are excluded so the contrast isolates presence versus value."""
        for rep in CORE_REPRESENTATIONS:
            view = build_representation(small, rep)
            assert not any(n.startswith("static::") for n in view.names)

    def test_values_mask_is_the_union(self, small: Cohort) -> None:
        mask = build_representation(small, Representation.MASK_ONLY)
        values = build_representation(small, Representation.VALUES_ONLY)
        both = build_representation(small, Representation.VALUES_MASK)
        assert both.n_features == mask.n_features + values.n_features
        assert set(both.names) == set(mask.names) | set(values.names)

    def test_unknown_representation_rejected(self, small: Cohort) -> None:
        with pytest.raises(ConfigError, match="unknown representation"):
            build_representation(small, "telepathy")  # type: ignore[arg-type]

    def test_truncation_changes_the_representation(self, small: Cohort) -> None:
        """Guards that features really are computed over the truncated window."""
        full = build_representation(small, Representation.MASK_ONLY)
        early = build_representation(small.truncate(1), Representation.MASK_ONLY)
        assert not np.array_equal(full.x, early.x)


class TestArtifacts:
    def test_stable_hash_is_order_independent_for_mappings(self) -> None:
        assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})

    def test_stable_hash_detects_change(self) -> None:
        assert stable_hash({"a": 1}) != stable_hash({"a": 2})

    def test_cohort_fingerprint_changes_with_truncation(self, toy_cohort: Cohort) -> None:
        assert cohort_fingerprint(toy_cohort) != cohort_fingerprint(toy_cohort.truncate(1))

    def test_split_hash_is_deterministic(self, toy_cohort: Cohort) -> None:
        from cliniverse.data.splits import stratified_folds

        a = stratified_folds(toy_cohort, n_folds=2, seed=1)
        b = stratified_folds(toy_cohort, n_folds=2, seed=1)
        assert split_hash(a) == split_hash(b)

    def test_split_hash_changes_with_membership(self, toy_cohort: Cohort) -> None:
        """Asserted on membership rather than seed: a 4-patient cohort has few
        distinct 2-fold partitions, so two seeds can legitimately coincide."""
        import dataclasses

        from cliniverse.data.splits import stratified_folds

        folds = stratified_folds(toy_cohort, n_folds=2, seed=1)
        moved = [
            dataclasses.replace(
                folds[0], train=folds[0].validation, validation=folds[0].train
            ),
            folds[1],
        ]
        assert split_hash(folds) != split_hash(moved)
