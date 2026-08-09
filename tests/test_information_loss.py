"""Information-loss, calibration and selective-prediction tests.

The load-bearing property is severity matching: if group loss removed more cells
than the matched cell condition, "group loss is worse" would be explained by
amount rather than structure, and the whole M3 contrast would be void.
"""

from __future__ import annotations

import numpy as np
import pytest

from cliniverse.acquisition.catalogue import Panel, PanelCatalogue
from cliniverse.data.cohort import Cohort
from cliniverse.evaluation.calibration import (
    MIN_ISOTONIC_CALIBRATION_N,
    CalibratorKind,
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    build_calibrator,
)
from cliniverse.evaluation.information_loss import (
    LossCondition,
    apply_information_loss,
    eligible_columns,
    matched_pair,
)
from cliniverse.evaluation.selective import (
    aurc,
    confidence,
    mean_predictive_entropy,
    predictive_entropy,
    risk_coverage_curve,
)
from cliniverse.exceptions import ConfigError

CATALOGUE = PanelCatalogue(
    version="test",
    panels={
        "alpha": Panel(name="alpha", label="Alpha", members=("a1", "a2"), cost=1.0),
        "beta": Panel(name="beta", label="Beta", members=("b1", "b2"), cost=1.0),
    },
)
VARIABLES = ("a1", "a2", "b1", "b2", "vital")


@pytest.fixture
def cohort() -> Cohort:
    """12 patients, 8 hours, 5 variables; the last is a vital (not eligible)."""
    n, t, v = 12, 8, 5
    rng = np.random.default_rng(0)
    m = rng.random((n, t, v)) < 0.6
    m[:, :, 4] = True  # vital always present
    x = np.where(m, rng.normal(size=(n, t, v)), np.nan).astype(np.float32)
    return Cohort(
        record_ids=np.arange(n, dtype=np.int64),
        source_set=np.array(["a"] * n, dtype=np.str_),
        x=x,
        m=m,
        statics=np.zeros((n, 1), dtype=np.float32),
        statics_mask=np.ones((n, 1), dtype=bool),
        labels={"mortality": np.array([0, 1] * (n // 2), dtype=np.float32)},
        variable_names=VARIABLES,
        static_names=("Age",),
    )


class TestEligibility:
    def test_only_catalogue_variables_are_eligible(self, cohort: Cohort) -> None:
        cols, groups = eligible_columns(cohort, CATALOGUE)
        assert cols.tolist() == [0, 1, 2, 3]  # the vital at index 4 is excluded
        assert set(groups) == {"alpha", "beta"}

    def test_vitals_are_never_removed(self, cohort: Cohort) -> None:
        out = apply_information_loss(
            cohort, LossCondition.GROUP_STRUCTURED, 1.0, CATALOGUE, seed=1
        )
        np.testing.assert_array_equal(out.cohort.m[:, :, 4], cohort.m[:, :, 4])

    def test_catalogue_covering_nothing_is_rejected(self, cohort: Cohort) -> None:
        empty = PanelCatalogue(
            version="t",
            panels={"z": Panel(name="z", label="Z", members=("nope",), cost=1.0)},
        )
        with pytest.raises(ConfigError, match="covers no variables"):
            eligible_columns(cohort, empty)


class TestLossMechanics:
    def test_none_condition_changes_nothing(self, cohort: Cohort) -> None:
        out = apply_information_loss(cohort, LossCondition.NONE, 0.0, CATALOGUE, seed=1)
        np.testing.assert_array_equal(out.cohort.m, cohort.m)
        assert out.removed_cells.sum() == 0

    def test_removed_cells_become_unobserved_and_nan(self, cohort: Cohort) -> None:
        out = apply_information_loss(
            cohort, LossCondition.GROUP_STRUCTURED, 0.5, CATALOGUE, seed=1
        )
        removed = cohort.m & ~out.cohort.m
        assert removed.any()
        assert bool(np.isnan(out.cohort.x[removed]).all())

    def test_loss_never_creates_observations(self, cohort: Cohort) -> None:
        out = apply_information_loss(cohort, LossCondition.CELL_RANDOM, 0.5, CATALOGUE, seed=1)
        assert not bool((out.cohort.m & ~cohort.m).any())

    def test_group_loss_removes_whole_groups(self, cohort: Cohort) -> None:
        """A removed group must lose every observed cell, not a subset."""
        out = apply_information_loss(
            cohort, LossCondition.GROUP_STRUCTURED, 1.0, CATALOGUE, seed=3
        )
        _, groups = eligible_columns(cohort, CATALOGUE)
        for i in range(cohort.n_patients):
            for cols in groups.values():
                before = cohort.m[i][:, cols]
                after = out.cohort.m[i][:, cols]
                if before.any() and not after.any():
                    continue  # fully removed
                np.testing.assert_array_equal(before, after)  # or untouched

    def test_full_severity_removes_all_eligible(self, cohort: Cohort) -> None:
        out = apply_information_loss(
            cohort, LossCondition.GROUP_STRUCTURED, 1.0, CATALOGUE, seed=1
        )
        assert not bool(out.cohort.m[:, :, :4].any())

    def test_original_cohort_is_not_mutated(self, cohort: Cohort) -> None:
        before = cohort.m.copy()
        apply_information_loss(cohort, LossCondition.GROUP_STRUCTURED, 0.5, CATALOGUE, seed=1)
        np.testing.assert_array_equal(cohort.m, before)

    def test_determinism(self, cohort: Cohort) -> None:
        a = apply_information_loss(
            cohort, LossCondition.GROUP_STRUCTURED, 0.5, CATALOGUE, seed=7
        )
        b = apply_information_loss(
            cohort, LossCondition.GROUP_STRUCTURED, 0.5, CATALOGUE, seed=7
        )
        np.testing.assert_array_equal(a.cohort.m, b.cohort.m)

    def test_invalid_severity_rejected(self, cohort: Cohort) -> None:
        with pytest.raises(ConfigError, match="severity"):
            apply_information_loss(cohort, LossCondition.CELL_RANDOM, 1.5, CATALOGUE, seed=1)


class TestMatchedSeverity:
    """The control the entire M3 contrast depends on."""

    @pytest.mark.parametrize("severity", [0.25, 0.5, 0.75])
    def test_group_and_cell_remove_identical_counts_per_patient(
        self, cohort: Cohort, severity: float
    ) -> None:
        group, cell = matched_pair(cohort, severity, CATALOGUE, seed=11)
        np.testing.assert_array_equal(group.removed_cells, cell.removed_cells)

    def test_matched_conditions_differ_in_structure(self, cohort: Cohort) -> None:
        """Same amount removed, different cells — otherwise there is no contrast."""
        group, cell = matched_pair(cohort, 0.5, CATALOGUE, seed=11)
        assert not np.array_equal(group.cohort.m, cell.cohort.m)

    def test_realized_severity_is_recorded_and_bounded(self, cohort: Cohort) -> None:
        group, _ = matched_pair(cohort, 0.5, CATALOGUE, seed=11)
        realized = group.realized_severity
        assert np.all(realized >= 0.0) and np.all(realized <= 1.0)

    def test_greedy_rule_does_not_wildly_overshoot(self, cohort: Cohort) -> None:
        """Accept a group only if it lands nearer the target than stopping.

        The earlier 'remove until target reached' rule turned a requested 0.25
        into a realized 0.46 on the real cohort.
        """
        group, _ = matched_pair(cohort, 0.25, CATALOGUE, seed=5)
        eligible = group.eligible_cells > 0
        realized = group.realized_severity[eligible]
        # With only two groups the granularity is coarse, so this is a loose
        # bound; it still fails for a rule that always overshoots.
        assert float(realized.mean()) <= 0.6

    def test_matching_survives_patients_with_no_eligible_cells(self) -> None:
        n, t, v = 3, 4, 5
        m = np.zeros((n, t, v), dtype=bool)
        m[:, :, 4] = True  # only the vital
        x = np.where(m, 1.0, np.nan).astype(np.float32)
        cohort = Cohort(
            record_ids=np.arange(n, dtype=np.int64),
            source_set=np.array(["a"] * n, dtype=np.str_),
            x=x,
            m=m,
            statics=np.zeros((n, 1), dtype=np.float32),
            statics_mask=np.ones((n, 1), dtype=bool),
            labels={"mortality": np.array([0, 1, 0], dtype=np.float32)},
            variable_names=VARIABLES,
            static_names=("Age",),
        )
        group, cell = matched_pair(cohort, 0.5, CATALOGUE, seed=1)
        assert group.removed_cells.sum() == 0
        np.testing.assert_array_equal(group.removed_cells, cell.removed_cells)


class TestSelectivePrediction:
    def test_perfect_confidence_ordering_gives_low_aurc(self) -> None:
        y = np.array([0.0, 0.0, 1.0, 1.0])
        good = np.array([0.01, 0.02, 0.98, 0.99])
        bad = np.array([0.99, 0.98, 0.02, 0.01])
        assert aurc(y, good) < aurc(y, bad)

    def test_risk_coverage_is_monotone_in_length(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.binomial(1, 0.3, 200).astype(float)
        p = np.clip(rng.random(200), 0.01, 0.99)
        curve = risk_coverage_curve(y, p)
        assert curve.coverage[0] < curve.coverage[-1]
        assert curve.coverage[-1] == pytest.approx(1.0)
        assert len(curve.risk) == len(y)

    def test_entropy_is_maximal_at_one_half(self) -> None:
        e = predictive_entropy(np.array([0.5, 0.1, 0.9, 0.01]))
        assert e[0] == pytest.approx(np.log(2))
        assert e[0] > e[1] and e[0] > e[2] and e[0] > e[3]

    def test_confidence_is_zero_at_the_boundary(self) -> None:
        assert confidence(np.array([0.5]))[0] == pytest.approx(0.0)
        assert confidence(np.array([1.0]))[0] == pytest.approx(1.0)

    def test_mean_entropy_ignores_labels(self) -> None:
        p = np.array([0.2, 0.8])
        a = mean_predictive_entropy(np.array([0.0, 1.0]), p)
        b = mean_predictive_entropy(np.array([1.0, 0.0]), p)
        assert a == pytest.approx(b)

    def test_unknown_loss_rejected(self) -> None:
        with pytest.raises(ConfigError, match="unknown loss"):
            risk_coverage_curve(np.array([0.0, 1.0]), np.array([0.2, 0.8]), loss="hinge")


class TestCalibrators:
    @pytest.fixture
    def calib_data(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(2)
        n = 4000
        p_true = rng.uniform(0.02, 0.9, n)
        y = rng.binomial(1, p_true).astype(float)
        # Systematically miscalibrated: predictions shifted too low.
        p = np.clip(p_true * 0.5, 1e-6, 1 - 1e-6)
        return y, p

    def test_identity_is_a_no_op(self, calib_data) -> None:
        y, p = calib_data
        out = IdentityCalibrator().fit(p, y).transform(p)
        np.testing.assert_array_equal(out, p)

    def test_platt_corrects_a_systematic_shift(self, calib_data) -> None:
        y, p = calib_data
        cal = PlattCalibrator().fit(p, y)
        out = cal.transform(p)
        assert abs(out.mean() - y.mean()) < abs(p.mean() - y.mean())

    def test_platt_preserves_ranking(self, calib_data) -> None:
        """Monotone map: it cannot change AUROC."""
        y, p = calib_data
        out = PlattCalibrator().fit(p, y).transform(p)
        assert np.array_equal(np.argsort(p), np.argsort(out))

    def test_platt_before_fit_raises(self) -> None:
        with pytest.raises(ConfigError, match="before fit"):
            PlattCalibrator().transform(np.array([0.5]))

    def test_single_class_calibration_set_rejected(self) -> None:
        with pytest.raises(ConfigError, match="single class"):
            PlattCalibrator().fit(np.array([0.2, 0.3]), np.array([0.0, 0.0]))

    def test_isotonic_refuses_small_calibration_sets(self) -> None:
        rng = np.random.default_rng(0)
        n = MIN_ISOTONIC_CALIBRATION_N - 1
        with pytest.raises(ConfigError, match="at least"):
            IsotonicCalibrator().fit(rng.random(n), rng.binomial(1, 0.3, n).astype(float))

    def test_isotonic_output_never_saturates(self, calib_data) -> None:
        """Exact 0 or 1 would make log-loss infinite."""
        y, p = calib_data
        out = IsotonicCalibrator().fit(p, y).transform(p)
        assert out.min() > 0.0 and out.max() < 1.0

    def test_builder_returns_each_kind(self) -> None:
        for kind in CalibratorKind:
            assert build_calibrator(kind).kind is kind
