"""M5 failure-search tests.

Two properties are load-bearing and are tested hardest.

**Amount matching.** Every reported M5 quantity is an excess over a control that
removed the same number of cells from the same patient. If matching silently
failed, "this configuration is damaging" would collapse back into "this
configuration removed more", which is exactly the M3-A error that repair #3
corrected.

**The lock.** Selection must be computable from discovery scores alone. A test
constructs a configuration that looks unremarkable on discovery and spectacular
on confirmation, and asserts it is not selected.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import rankdata, spearmanr

from cliniverse.acquisition.catalogue import Panel, PanelCatalogue
from cliniverse.data.cohort import Cohort
from cliniverse.evaluation.failure_search import (
    ConfigurationScore,
    _rankdata,
    apply_group_subset_loss,
    control_seed,
    enumerate_group_subsets,
    holm_bonferroni,
    matched_random_control,
    select_top_k,
    spearman_permutation_test,
    subset_columns,
    subset_removal_mask,
)
from cliniverse.evaluation.metrics import (
    brier_score,
    negative_log_likelihood,
    paired_bootstrap_difference,
    paired_mean_difference_bootstrap,
    per_patient_log_loss,
    per_patient_squared_error,
)
from cliniverse.exceptions import ConfigError

CATALOGUE = PanelCatalogue(
    version="test",
    panels={
        "alpha": Panel(name="alpha", label="Alpha", members=("a1", "a2"), cost=1.0),
        "beta": Panel(name="beta", label="Beta", members=("b1", "b2"), cost=1.0),
        "gamma": Panel(name="gamma", label="Gamma", members=("g1",), cost=1.0),
    },
)
VARIABLES = ("a1", "a2", "b1", "b2", "g1", "vital")


@pytest.fixture
def cohort() -> Cohort:
    """16 patients, 8 hours, 6 variables; the last is a vital (never eligible)."""
    n, t, v = 16, 8, 6
    rng = np.random.default_rng(11)
    m = rng.random((n, t, v)) < 0.55
    m[:, :, 5] = True
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


class TestEnumeration:
    def test_covers_every_non_empty_subset(self) -> None:
        subsets = enumerate_group_subsets(("alpha", "beta", "gamma"))
        assert len(subsets) == 2**3 - 1 == 7
        assert len(set(subsets)) == len(subsets)
        assert all(len(s) >= 1 for s in subsets)

    def test_real_catalogue_yields_1023_configurations(self) -> None:
        subsets = enumerate_group_subsets(tuple(f"g{i}" for i in range(10)))
        assert len(subsets) == 1023

    def test_order_is_deterministic_and_size_major(self) -> None:
        first = enumerate_group_subsets(("beta", "alpha", "gamma"))
        second = enumerate_group_subsets(("gamma", "beta", "alpha"))
        assert first == second
        assert [len(s) for s in first] == sorted(len(s) for s in first)
        assert first[0] == ("alpha",)
        assert first[-1] == ("alpha", "beta", "gamma")

    def test_members_are_sorted_within_a_configuration(self) -> None:
        for subset in enumerate_group_subsets(("beta", "alpha", "gamma")):
            assert list(subset) == sorted(subset)

    def test_empty_group_set_rejected(self) -> None:
        with pytest.raises(ConfigError, match="empty group set"):
            enumerate_group_subsets(())


class TestSubsetRemoval:
    def test_removes_only_observed_cells_of_named_groups(self, cohort: Cohort) -> None:
        removal = subset_removal_mask(cohort, ("alpha",), CATALOGUE)
        columns = subset_columns(cohort, ("alpha",), CATALOGUE)
        assert set(columns.tolist()) == {0, 1}
        assert not bool((removal & ~cohort.m).any()), "withheld a never-observed cell"
        other = np.ones(len(VARIABLES), dtype=bool)
        other[columns] = False
        assert not bool(removal[:, :, other].any()), "touched a column outside the subset"

    def test_removes_every_occurrence_across_the_window(self, cohort: Cohort) -> None:
        loss = apply_group_subset_loss(cohort, ("alpha", "gamma"), CATALOGUE)
        for column in (0, 1, 4):
            assert not bool(loss.cohort.m[:, :, column].any())
            assert bool(np.isnan(loss.cohort.x[:, :, column]).all())

    def test_untouched_variables_are_bit_identical(self, cohort: Cohort) -> None:
        loss = apply_group_subset_loss(cohort, ("alpha",), CATALOGUE)
        assert np.array_equal(loss.cohort.m[:, :, 2:], cohort.m[:, :, 2:])
        kept = loss.cohort.x[:, :, 2:]
        original = cohort.x[:, :, 2:]
        assert np.array_equal(np.isnan(kept), np.isnan(original))
        assert np.allclose(kept[~np.isnan(kept)], original[~np.isnan(original)])

    def test_vitals_are_never_eligible(self, cohort: Cohort) -> None:
        loss = apply_group_subset_loss(cohort, ("alpha", "beta", "gamma"), CATALOGUE)
        assert bool(loss.cohort.m[:, :, 5].all()), "a vital was withheld"

    def test_counts_agree_with_the_mask(self, cohort: Cohort) -> None:
        loss = apply_group_subset_loss(cohort, ("beta",), CATALOGUE)
        removal = subset_removal_mask(cohort, ("beta",), CATALOGUE)
        assert np.array_equal(loss.removed_cells, removal.sum(axis=(1, 2)))
        assert np.array_equal(loss.removed_by_variable, removal.sum(axis=1))
        assert int(loss.removed_cells.sum()) == int(removal.sum())

    def test_is_deterministic(self, cohort: Cohort) -> None:
        first = apply_group_subset_loss(cohort, ("alpha", "beta"), CATALOGUE)
        second = apply_group_subset_loss(cohort, ("alpha", "beta"), CATALOGUE)
        assert np.array_equal(first.cohort.m, second.cohort.m)
        assert np.array_equal(first.removed_cells, second.removed_cells)

    def test_realized_severity_is_a_fraction(self, cohort: Cohort) -> None:
        loss = apply_group_subset_loss(cohort, ("alpha", "beta", "gamma"), CATALOGUE)
        severity = loss.realized_severity
        assert np.all((severity >= 0.0) & (severity <= 1.0))
        # Withholding every group removes every eligible cell.
        assert np.allclose(severity[loss.eligible_cells > 0], 1.0)

    def test_empty_configuration_rejected(self, cohort: Cohort) -> None:
        with pytest.raises(ConfigError, match="at least one group"):
            apply_group_subset_loss(cohort, (), CATALOGUE)

    def test_unknown_group_rejected(self, cohort: Cohort) -> None:
        with pytest.raises(ConfigError, match="absent from this cohort"):
            apply_group_subset_loss(cohort, ("delta",), CATALOGUE)

    def test_repeated_group_rejected(self, cohort: Cohort) -> None:
        with pytest.raises(ConfigError, match="repeats a group"):
            apply_group_subset_loss(cohort, ("alpha", "alpha"), CATALOGUE)


class TestAmountMatchedControl:
    def test_removes_exactly_the_requested_count_per_patient(self, cohort: Cohort) -> None:
        loss = apply_group_subset_loss(cohort, ("alpha",), CATALOGUE)
        control = matched_random_control(cohort, loss.removed_cells, CATALOGUE, seed=3)
        removed = (cohort.m & ~control.m).sum(axis=(1, 2))
        assert np.array_equal(removed, loss.removed_cells)

    def test_only_observed_eligible_cells_are_removed(self, cohort: Cohort) -> None:
        loss = apply_group_subset_loss(cohort, ("beta",), CATALOGUE)
        control = matched_random_control(cohort, loss.removed_cells, CATALOGUE, seed=5)
        removed = cohort.m & ~control.m
        assert not bool(removed[:, :, 5].any()), "the control removed a vital"
        assert not bool((removed & ~cohort.m).any())

    def test_control_generally_differs_from_the_configuration(self, cohort: Cohort) -> None:
        loss = apply_group_subset_loss(cohort, ("alpha",), CATALOGUE)
        control = matched_random_control(cohort, loss.removed_cells, CATALOGUE, seed=7)
        assert not np.array_equal(control.m, loss.cohort.m), (
            "the control reproduced the configuration mask, so the excess "
            "statistic would be identically zero by construction"
        )

    def test_same_seed_reproduces_same_control(self, cohort: Cohort) -> None:
        loss = apply_group_subset_loss(cohort, ("alpha",), CATALOGUE)
        a = matched_random_control(cohort, loss.removed_cells, CATALOGUE, seed=13)
        b = matched_random_control(cohort, loss.removed_cells, CATALOGUE, seed=13)
        assert np.array_equal(a.m, b.m)

    def test_different_seeds_give_different_controls(self, cohort: Cohort) -> None:
        loss = apply_group_subset_loss(cohort, ("alpha",), CATALOGUE)
        a = matched_random_control(cohort, loss.removed_cells, CATALOGUE, seed=13)
        b = matched_random_control(cohort, loss.removed_cells, CATALOGUE, seed=14)
        assert not np.array_equal(a.m, b.m)

    def test_infeasible_request_is_rejected(self, cohort: Cohort) -> None:
        impossible = np.full(cohort.n_patients, 10_000, dtype=np.int64)
        with pytest.raises(ConfigError, match="amount matching failed"):
            matched_random_control(cohort, impossible, CATALOGUE, seed=1)


class TestControlSeed:
    def test_is_deterministic(self) -> None:
        assert control_seed(7, ("alpha", "beta"), 0) == control_seed(7, ("alpha", "beta"), 0)

    def test_separates_configurations_and_repetitions(self) -> None:
        seeds = {
            control_seed(20260809, subset, repetition)
            for subset in (("alpha",), ("beta",), ("alpha", "beta"))
            for repetition in range(3)
        }
        assert len(seeds) == 9

    def test_stays_in_positive_int32_range(self) -> None:
        for repetition in range(5):
            seed = control_seed(20260809, ("alpha", "gamma"), repetition)
            assert 0 <= seed < 2**31 - 1


def _score(
    index: int, subset: tuple[str, ...], excess: float, auroc: float
) -> ConfigurationScore:
    return ConfigurationScore(
        index=index,
        subset=subset,
        n_groups=len(subset),
        mean_removed_cells=1.0,
        mean_realized_severity=0.1,
        delta_nll_excess=excess,
        delta_brier_excess=excess / 10.0,
        nll=0.3,
        nll_control=0.3 - excess,
        brier=0.1,
        brier_control=0.1,
        auroc=auroc,
        auroc_control_last_draw=auroc,
        auprc=0.4,
        calibration_intercept=0.0,
        calibration_slope=1.0,
        mean_predicted_risk=0.14,
    )


class TestLock:
    def test_selects_highest_discovery_excess(self) -> None:
        scores = [
            _score(0, ("alpha",), 0.001, 0.82),
            _score(1, ("beta",), 0.009, 0.82),
            _score(2, ("gamma",), 0.005, 0.82),
        ]
        assert select_top_k(scores, k=2, clean_auroc=0.83, delta=0.02) == (
            ("beta",),
            ("gamma",),
        )

    def test_auroc_constraint_excludes_loud_failures(self) -> None:
        scores = [
            _score(0, ("alpha",), 0.500, 0.10),  # enormous damage, but not silent
            _score(1, ("beta",), 0.002, 0.82),
        ]
        assert select_top_k(scores, k=1, clean_auroc=0.83, delta=0.02) == (("beta",),)

    def test_cannot_see_confirmation_scores(self) -> None:
        """The selection is computed from discovery alone.

        `bait` is unremarkable on discovery and spectacular on confirmation. A
        selection that peeked would pick it; the locked selection must not.
        """
        bait = ("alpha",)
        discovery = [_score(0, bait, 0.0001, 0.82), _score(1, ("beta",), 0.004, 0.82)]
        confirmation = [_score(0, bait, 99.0, 0.82), _score(1, ("beta",), 0.004, 0.82)]
        selected = select_top_k(discovery, k=1, clean_auroc=0.83, delta=0.02)
        assert selected == (("beta",),)
        assert bait not in selected
        # The confirmation table exists and disagrees; the lock ignored it.
        assert max(s.delta_nll_excess for s in confirmation) == 99.0

    def test_ties_break_by_enumeration_index(self) -> None:
        scores = [
            _score(5, ("beta",), 0.004, 0.82),
            _score(2, ("alpha",), 0.004, 0.82),
        ]
        assert select_top_k(scores, k=2, clean_auroc=0.83, delta=0.02) == (
            ("alpha",),
            ("beta",),
        )

    def test_k_larger_than_eligible_returns_all_eligible(self) -> None:
        scores = [_score(0, ("alpha",), 0.004, 0.82)]
        assert len(select_top_k(scores, k=5, clean_auroc=0.83, delta=0.02)) == 1

    def test_no_eligible_configuration_raises(self) -> None:
        scores = [_score(0, ("alpha",), 0.5, 0.10)]
        with pytest.raises(ConfigError, match="preserved AUROC"):
            select_top_k(scores, k=1, clean_auroc=0.83, delta=0.02)

    def test_invalid_k_and_delta_rejected(self) -> None:
        scores = [_score(0, ("alpha",), 0.004, 0.82)]
        with pytest.raises(ConfigError, match="k must be positive"):
            select_top_k(scores, k=0, clean_auroc=0.83, delta=0.02)
        with pytest.raises(ConfigError, match="delta must be"):
            select_top_k(scores, k=1, clean_auroc=0.83, delta=-1.0)


class TestGeneralizationStatistic:
    def test_rankdata_matches_scipy_including_ties(self) -> None:
        values = np.array([3.0, 1.0, 2.0, 3.0, 1.0, 5.0])
        assert np.allclose(_rankdata(values), rankdata(values))

    def test_perfect_agreement_is_detected(self) -> None:
        a = np.arange(30, dtype=np.float64)
        result = spearman_permutation_test(a, a.copy(), n_permutations=500, seed=1)
        assert result["spearman_rho"] == pytest.approx(1.0)
        assert result["p_value_one_sided"] < 0.01

    def test_rho_matches_scipy(self) -> None:
        rng = np.random.default_rng(4)
        a = rng.normal(size=40)
        b = a + rng.normal(size=40) * 0.5
        result = spearman_permutation_test(a, b, n_permutations=200, seed=1)
        assert result["spearman_rho"] == pytest.approx(spearmanr(a, b).statistic)

    def test_anticorrelation_does_not_pass(self) -> None:
        a = np.arange(30, dtype=np.float64)
        result = spearman_permutation_test(a, -a, n_permutations=500, seed=1)
        assert result["spearman_rho"] == pytest.approx(-1.0)
        assert result["p_value_one_sided"] > 0.9

    def test_independent_vectors_do_not_pass(self) -> None:
        rng = np.random.default_rng(9)
        result = spearman_permutation_test(
            rng.normal(size=60), rng.normal(size=60), n_permutations=1000, seed=2
        )
        assert result["p_value_one_sided"] > 0.05

    def test_p_value_is_never_zero(self) -> None:
        a = np.arange(50, dtype=np.float64)
        result = spearman_permutation_test(a, a.copy(), n_permutations=100, seed=1)
        assert result["p_value_one_sided"] > 0.0

    def test_shape_and_size_validation(self) -> None:
        with pytest.raises(ConfigError, match="differ in shape"):
            spearman_permutation_test(np.zeros(4), np.zeros(5), n_permutations=10, seed=1)
        with pytest.raises(ConfigError, match="at least three"):
            spearman_permutation_test(np.zeros(2), np.zeros(2), n_permutations=10, seed=1)


class TestHolmBonferroni:
    def test_rejects_both_when_both_are_small(self) -> None:
        assert holm_bonferroni([0.001, 0.004]) == [True, True]

    def test_step_down_stops_at_the_first_failure(self) -> None:
        assert holm_bonferroni([0.001, 0.4]) == [True, False]

    def test_rejects_nothing_when_all_are_large(self) -> None:
        assert holm_bonferroni([0.2, 0.9]) == [False, False]

    def test_smallest_uses_the_strictest_threshold(self) -> None:
        # 0.03 > 0.05/2, so even the smaller p-value survives the null.
        assert holm_bonferroni([0.03, 0.04]) == [False, False]


class TestPerPatientMetrics:
    @pytest.fixture
    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(3)
        y = (rng.random(400) < 0.2).astype(np.float64)
        p = np.clip(rng.random(400), 0.01, 0.99)
        return y, p

    def test_log_loss_mean_matches_the_scalar_metric(
        self, sample: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y, p = sample
        assert per_patient_log_loss(y, p).mean() == pytest.approx(
            negative_log_likelihood(y, p), abs=1e-12
        )

    def test_squared_error_mean_matches_brier(
        self, sample: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y, p = sample
        assert per_patient_squared_error(y, p).mean() == pytest.approx(
            brier_score(y, p), abs=1e-15
        )

    def test_paired_mean_bootstrap_matches_the_audited_general_path(
        self, sample: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """The fast paired path must not change any interval it replaces."""
        y, p_a = sample
        rng = np.random.default_rng(21)
        p_b = np.clip(p_a + rng.normal(scale=0.05, size=p_a.size), 0.01, 0.99)

        general = paired_bootstrap_difference(
            y,
            p_a,
            p_b,
            negative_log_likelihood,
            metric_name="nll",
            name_a="a",
            name_b="b",
            n_boot=300,
            seed=99,
        )
        fast = paired_mean_difference_bootstrap(
            y,
            per_patient_log_loss(y, p_b) - per_patient_log_loss(y, p_a),
            metric_name="nll",
            name_a="a",
            name_b="b",
            n_boot=300,
            seed=99,
        )
        assert fast.difference == pytest.approx(general.difference, abs=1e-12)
        assert fast.low == pytest.approx(general.low, abs=1e-12)
        assert fast.high == pytest.approx(general.high, abs=1e-12)
        assert fast.n_boot == general.n_boot
        assert fast.excludes_zero == general.excludes_zero

    def test_wider_percentiles_give_a_wider_interval(
        self, sample: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y, p = sample
        difference = per_patient_log_loss(y, p) - 0.3
        narrow = paired_mean_difference_bootstrap(
            y, difference, metric_name="d", name_a="a", name_b="b", n_boot=300
        )
        wide = paired_mean_difference_bootstrap(
            y,
            difference,
            metric_name="d",
            name_a="a",
            name_b="b",
            n_boot=300,
            percentiles=(0.5, 99.5),
        )
        assert wide.low <= narrow.low
        assert wide.high >= narrow.high

    def test_zero_difference_includes_zero(self) -> None:
        y = np.array([0.0, 1.0] * 50)
        result = paired_mean_difference_bootstrap(
            y, np.zeros(100), metric_name="d", name_a="a", name_b="b", n_boot=200
        )
        assert result.difference == 0.0
        assert not result.excludes_zero
        assert result.p_value == pytest.approx(1.0)

    def test_validation(self) -> None:
        y = np.array([0.0, 1.0, 0.0, 1.0])
        with pytest.raises(ConfigError, match="differ in shape"):
            paired_mean_difference_bootstrap(
                y, np.zeros(3), metric_name="d", name_a="a", name_b="b"
            )
        with pytest.raises(ConfigError, match="non-finite"):
            paired_mean_difference_bootstrap(
                y,
                np.array([0.0, np.nan, 0.0, 0.0]),
                metric_name="d",
                name_a="a",
                name_b="b",
            )
        with pytest.raises(ConfigError, match="percentiles must be"):
            paired_mean_difference_bootstrap(
                y,
                np.zeros(4),
                metric_name="d",
                name_a="a",
                name_b="b",
                percentiles=(90.0, 10.0),
            )
