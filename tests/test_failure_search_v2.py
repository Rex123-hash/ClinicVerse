"""M5-v2 stability-aware selection tests.

The properties that carry the v2 result are the ones M5-v1 got wrong, so they are
tested hardest:

- **parsimony**: a candidate that is a bigger sibling of an equally good one must
  never be selected. This is the exact v1 pathology where `BMP_like+TroponinI`
  outranked plain `BMP_like` on noise.
- **tie-band width**: the band comes from the leader's fold dispersion, and a
  candidate outside it must not win on sparsity alone.
- **null-control separation**: null-region candidates compete in the same pool, so
  the selector must be able to pick them when they genuinely win — that is what
  makes the sanity gate informative rather than decorative.
- **detectability**: the MDE must scale correctly, or the gate protecting set-c is
  worthless.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from cliniverse.acquisition.catalogue import Panel, PanelCatalogue
from cliniverse.data.cohort import Cohort
from cliniverse.evaluation.failure_search import (
    analyte_columns,
    apply_analyte_subset_loss,
    apply_group_subset_loss,
    enumerate_group_subsets,
    fold_dispersion,
    matched_random_control,
    minimum_detectable_effect,
    pooled_auroc_on_folds,
    select_one_se_parsimonious,
    selection_frequency,
)
from cliniverse.exceptions import ConfigError

CATALOGUE = PanelCatalogue(
    version="test",
    panels={
        "alpha": Panel(name="alpha", label="Alpha", members=("a1", "a2", "a3"), cost=1.0),
        "beta": Panel(name="beta", label="Beta", members=("b1", "b2"), cost=1.0),
    },
)
VARIABLES = ("a1", "a2", "a3", "b1", "b2", "vital")


@pytest.fixture
def cohort() -> Cohort:
    """16 patients, 8 hours, 6 variables; the last is a vital (never eligible)."""
    n, t, v = 16, 8, 6
    rng = np.random.default_rng(5)
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


class TestAnalyteAddressing:
    def test_columns_resolve_by_analyte_name(self, cohort: Cohort) -> None:
        assert analyte_columns(cohort, ("a1", "a3"), CATALOGUE).tolist() == [0, 2]

    def test_order_of_names_does_not_matter(self, cohort: Cohort) -> None:
        first = analyte_columns(cohort, ("a3", "a1"), CATALOGUE)
        second = analyte_columns(cohort, ("a1", "a3"), CATALOGUE)
        assert first.tolist() == second.tolist()

    def test_uncovered_variable_rejected(self, cohort: Cohort) -> None:
        with pytest.raises(ConfigError, match="not covered by the catalogue"):
            analyte_columns(cohort, ("vital",), CATALOGUE)

    def test_unknown_and_empty_and_duplicate_rejected(self, cohort: Cohort) -> None:
        with pytest.raises(ConfigError, match="at least one analyte"):
            analyte_columns(cohort, (), CATALOGUE)
        with pytest.raises(ConfigError, match="repeats an analyte"):
            analyte_columns(cohort, ("a1", "a1"), CATALOGUE)
        with pytest.raises(ConfigError, match="not covered by the catalogue"):
            analyte_columns(cohort, ("nope",), CATALOGUE)


class TestAnalyteWithholding:
    def test_removes_only_the_named_analytes(self, cohort: Cohort) -> None:
        loss = apply_analyte_subset_loss(cohort, ("a1",), CATALOGUE)
        assert not bool(loss.cohort.m[:, :, 0].any())
        assert np.array_equal(loss.cohort.m[:, :, 1:], cohort.m[:, :, 1:])

    def test_whole_window_removal(self, cohort: Cohort) -> None:
        loss = apply_analyte_subset_loss(cohort, ("a1", "b2"), CATALOGUE)
        for column in (0, 4):
            assert bool(np.isnan(loss.cohort.x[:, :, column]).all())

    def test_full_group_matches_the_group_level_path(self, cohort: Cohort) -> None:
        """Withholding a group's analytes must equal withholding the group.

        This is what keeps v2 comparable with v1 and with M3.
        """
        by_analyte = apply_analyte_subset_loss(cohort, ("a1", "a2", "a3"), CATALOGUE)
        by_group = apply_group_subset_loss(cohort, ("alpha",), CATALOGUE)
        assert np.array_equal(by_analyte.cohort.m, by_group.cohort.m)
        assert np.array_equal(by_analyte.removed_cells, by_group.removed_cells)
        assert np.array_equal(by_analyte.eligible_cells, by_group.eligible_cells)

    def test_vital_is_never_withheld(self, cohort: Cohort) -> None:
        loss = apply_analyte_subset_loss(cohort, ("a1", "a2", "a3", "b1", "b2"), CATALOGUE)
        assert bool(loss.cohort.m[:, :, 5].all())

    def test_eligible_pool_is_all_catalogue_analytes(self, cohort: Cohort) -> None:
        """Eligibility is the whole catalogue, not just the withheld analytes.

        The amount-matched control draws from this pool, so narrowing it would
        silently change the control and break comparability with M3 and v1.
        """
        loss = apply_analyte_subset_loss(cohort, ("a1",), CATALOGUE)
        expected = cohort.m[:, :, [0, 1, 2, 3, 4]].sum(axis=(1, 2))
        assert np.array_equal(loss.eligible_cells, expected)

    def test_is_deterministic(self, cohort: Cohort) -> None:
        a = apply_analyte_subset_loss(cohort, ("a1", "b1"), CATALOGUE)
        b = apply_analyte_subset_loss(cohort, ("a1", "b1"), CATALOGUE)
        assert np.array_equal(a.cohort.m, b.cohort.m)

    def test_control_matches_analyte_pattern_amount(self, cohort: Cohort) -> None:
        loss = apply_analyte_subset_loss(cohort, ("a1", "a2"), CATALOGUE)
        control = matched_random_control(cohort, loss.removed_cells, CATALOGUE, seed=9)
        removed = (cohort.m & ~control.m).sum(axis=(1, 2))
        assert np.array_equal(removed, loss.removed_cells)


class TestFoldDispersion:
    def test_matches_standard_error_formula(self) -> None:
        values = np.array([0.010, 0.012, 0.008, 0.011, 0.009])
        assert fold_dispersion(values) == pytest.approx(values.std(ddof=1) / np.sqrt(5))

    def test_identical_folds_give_zero(self) -> None:
        assert fold_dispersion(np.full(5, 0.01)) == pytest.approx(0.0)

    def test_needs_at_least_two_folds(self) -> None:
        with pytest.raises(ConfigError, match="at least two folds"):
            fold_dispersion(np.array([0.01]))


class TestOneSeParsimonyRule:
    def test_prefers_the_sparsest_candidate_inside_the_band(self) -> None:
        """The exact M5-v1 pathology, as a regression test.

        `('BMP',)` and `('BMP', 'rare')` are effectively tied; v1 would have taken
        the marginally higher one. The rule must take the sparser.
        """
        means = {("BMP",): 0.0130, ("BMP", "rare"): 0.0132, ("CBC",): 0.0001}
        dispersions = dict.fromkeys(means, 0.001)
        selected = select_one_se_parsimonious(means, dispersions, set(means))
        assert selected == ("BMP",)

    def test_does_not_reach_outside_the_band_for_sparsity(self) -> None:
        means = {("BMP",): 0.0050, ("BMP", "x"): 0.0132, ("BMP", "x", "y"): 0.0131}
        dispersions = dict.fromkeys(means, 0.001)
        selected = select_one_se_parsimonious(means, dispersions, set(means))
        assert selected == ("BMP", "x")

    def test_wider_dispersion_widens_the_band(self) -> None:
        means = {("BMP",): 0.0050, ("BMP", "x"): 0.0132}
        tight = select_one_se_parsimonious(means, dict.fromkeys(means, 0.001), set(means))
        loose = select_one_se_parsimonious(means, dict.fromkeys(means, 0.01), set(means))
        assert tight == ("BMP", "x")
        assert loose == ("BMP",)

    def test_ineligible_candidates_cannot_be_selected(self) -> None:
        means = {("loud",): 0.5, ("quiet",): 0.001}
        dispersions = dict.fromkeys(means, 0.0001)
        selected = select_one_se_parsimonious(means, dispersions, {("quiet",)})
        assert selected == ("quiet",)

    def test_null_control_can_win_when_it_genuinely_wins(self) -> None:
        """The sanity gate is only informative if the selector *can* pick a null.

        A selector that structurally excluded the null region would make G1
        vacuous, so this asserts the pool is genuinely shared.
        """
        means = {("BMP",): 0.0001, ("CBC",): 0.0200}
        dispersions = dict.fromkeys(means, 0.0005)
        assert select_one_se_parsimonious(means, dispersions, set(means)) == ("CBC",)

    def test_ties_break_deterministically(self) -> None:
        means = {("b",): 0.01, ("a",): 0.01}
        dispersions = dict.fromkeys(means, 0.001)
        first = select_one_se_parsimonious(means, dispersions, set(means))
        second = select_one_se_parsimonious(
            dict(reversed(list(means.items()))), dispersions, set(means)
        )
        assert first == second == ("a",)

    def test_no_eligible_candidate_raises(self) -> None:
        means = {("BMP",): 0.01}
        with pytest.raises(ConfigError, match="discrimination-silent"):
            select_one_se_parsimonious(means, {("BMP",): 0.001}, set())

    def test_pinned_m5v2_top_level_selection_counts(self) -> None:
        artifact = Path("experiments/robustness/results/m5v2/m5v2_tables.npz")
        with np.load(artifact, allow_pickle=False) as tables:
            deltas = tables["deltas"]
            candidate_auroc = tables["candidate_auroc"]
            clean_auroc = tables["clean_auroc"]

        patterns = [
            *enumerate_group_subsets(
                ("BUN", "Creatinine", "Glucose", "HCO3", "K", "Mg", "Na")
            ),
            *enumerate_group_subsets(("HCT", "Platelets", "WBC")),
            *enumerate_group_subsets(("PaCO2", "PaO2", "pH")),
        ]
        selections = []
        for resplit in range(20):
            means = {
                pattern: float(deltas[index, resplit].mean())
                for index, pattern in enumerate(patterns)
            }
            dispersions = {
                pattern: fold_dispersion(deltas[index, resplit])
                for index, pattern in enumerate(patterns)
            }
            eligible = {
                pattern
                for index, pattern in enumerate(patterns)
                if clean_auroc[resplit] - candidate_auroc[index, resplit] <= 0.02
            }
            selections.append(select_one_se_parsimonious(means, dispersions, eligible))

        assert Counter(selections) == Counter(
            {
                ("BUN", "Glucose", "Na"): 11,
                ("BUN", "Glucose"): 4,
                ("BUN",): 1,
                ("BUN", "Glucose", "HCO3"): 1,
                ("BUN", "Glucose", "Mg"): 1,
                ("BUN", "Glucose", "HCO3", "Mg"): 1,
                ("BUN", "Glucose", "HCO3", "Na"): 1,
            }
        )


class TestSelectionFrequency:
    def test_counts_and_normalises(self) -> None:
        picks = [("a",), ("a",), ("b",), ("a",)]
        frequency = selection_frequency(picks)
        assert frequency[("a",)] == pytest.approx(0.75)
        assert frequency[("b",)] == pytest.approx(0.25)
        assert sum(frequency.values()) == pytest.approx(1.0)

    def test_majority_threshold_arithmetic(self) -> None:
        """11/20 is a strict majority; 10/20 is not."""
        assert round(selection_frequency([("a",)] * 11 + [("b",)] * 9)[("a",)] * 20) == 11
        assert round(selection_frequency([("a",)] * 10 + [("b",)] * 10)[("a",)] * 20) == 10

    def test_empty_rejected(self) -> None:
        with pytest.raises(ConfigError, match="no resplits"):
            selection_frequency([])


class TestNestedAurocIsolation:
    def test_held_out_fold_cannot_change_selection_auroc(self) -> None:
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float64)
        predictions = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
        fold_assignment = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)

        baseline = pooled_auroc_on_folds(labels, predictions, fold_assignment, folds=(0, 1, 2))
        altered = predictions.copy()
        altered[fold_assignment == 3] = altered[fold_assignment == 3][::-1]

        assert pooled_auroc_on_folds(
            labels, altered, fold_assignment, folds=(0, 1, 2)
        ) == pytest.approx(baseline)

    def test_included_fold_does_change_selection_auroc(self) -> None:
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float64)
        predictions = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
        fold_assignment = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)

        baseline = pooled_auroc_on_folds(labels, predictions, fold_assignment, folds=(0, 1, 2))
        altered = predictions.copy()
        altered[fold_assignment == 2] = altered[fold_assignment == 2][::-1]

        assert (
            pooled_auroc_on_folds(labels, altered, fold_assignment, folds=(0, 1, 2)) < baseline
        )

    def test_rejects_duplicate_or_empty_fold_selection(self) -> None:
        labels = np.array([0, 1], dtype=np.float64)
        predictions = np.array([0.1, 0.9], dtype=np.float64)
        fold_assignment = np.array([0, 0], dtype=np.int64)
        with pytest.raises(ConfigError, match="at least one fold"):
            pooled_auroc_on_folds(labels, predictions, fold_assignment, folds=())
        with pytest.raises(ConfigError, match="duplicates"):
            pooled_auroc_on_folds(labels, predictions, fold_assignment, folds=(0, 0))


class TestMinimumDetectableEffect:
    def test_matches_the_closed_form(self) -> None:
        # (z_0.95 + z_0.80) = 1.644854 + 0.841621 = 2.486475
        assert minimum_detectable_effect(0.21, 4000) == pytest.approx(
            2.486475 * 0.21 / np.sqrt(4000), rel=1e-5
        )

    def test_reproduces_the_predeclared_expectation(self) -> None:
        """Design section 7 predicted MDE ~= 0.008 at sigma ~= 0.21, n = 4000."""
        assert minimum_detectable_effect(0.21, 4000) == pytest.approx(0.00826, abs=5e-5)

    def test_shrinks_with_sample_size(self) -> None:
        assert minimum_detectable_effect(0.2, 16000) < minimum_detectable_effect(0.2, 4000)

    def test_grows_with_dispersion(self) -> None:
        assert minimum_detectable_effect(0.4, 4000) > minimum_detectable_effect(0.2, 4000)

    def test_validation(self) -> None:
        with pytest.raises(ConfigError, match="sigma must be"):
            minimum_detectable_effect(0.0, 4000)
        with pytest.raises(ConfigError, match="n must be positive"):
            minimum_detectable_effect(0.2, 0)
        with pytest.raises(ConfigError, match="implausible"):
            minimum_detectable_effect(0.2, 4000, power=0.2)


class TestSearchSpaceShape:
    def test_bmp_like_yields_127_patterns(self) -> None:
        members = ("BUN", "Creatinine", "Glucose", "HCO3", "K", "Mg", "Na")
        assert len(enumerate_group_subsets(members)) == 127

    def test_null_control_regions_yield_seven_each(self) -> None:
        assert len(enumerate_group_subsets(("HCT", "Platelets", "WBC"))) == 7
        assert len(enumerate_group_subsets(("PaCO2", "PaO2", "pH"))) == 7

    def test_total_pool_is_141(self) -> None:
        total = (
            len(
                enumerate_group_subsets(
                    ("BUN", "Creatinine", "Glucose", "HCO3", "K", "Mg", "Na")
                )
            )
            + len(enumerate_group_subsets(("HCT", "Platelets", "WBC")))
            + len(enumerate_group_subsets(("PaCO2", "PaO2", "pH")))
        )
        assert total == 141
