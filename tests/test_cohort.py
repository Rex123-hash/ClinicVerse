"""Cohort invariants and selection semantics."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from cliniverse.data.cohort import Cohort
from cliniverse.exceptions import DataError


class TestInvariants:
    def test_mask_value_consistency_enforced(self, toy_cohort: Cohort) -> None:
        """A cell marked observed must hold a finite value, and vice versa.

        Everything downstream (imputation, masking, acquisition) assumes this.
        """
        bad_x = toy_cohort.x.copy()
        bad_x[3, 0, 0] = 1.0  # finite value in a cell the mask says is unobserved
        with pytest.raises(DataError, match="unobserved"):
            dataclasses.replace(toy_cohort, x=bad_x)

    def test_observed_cell_must_be_finite(self, toy_cohort: Cohort) -> None:
        bad_m = toy_cohort.m.copy()
        bad_m[3, 0, 0] = True  # mask says observed, value is NaN
        with pytest.raises(DataError, match="not finite"):
            dataclasses.replace(toy_cohort, m=bad_m)

    def test_shape_mismatch_rejected(self, toy_cohort: Cohort) -> None:
        with pytest.raises(DataError, match="variable names"):
            dataclasses.replace(toy_cohort, variable_names=("only_one",))

    def test_label_length_mismatch_rejected(self, toy_cohort: Cohort) -> None:
        with pytest.raises(DataError, match="label"):
            dataclasses.replace(
                toy_cohort, labels={"mortality": np.array([0.0], dtype=np.float32)}
            )


class TestSummaries:
    def test_observation_counts(self, toy_cohort: Cohort) -> None:
        assert toy_cohort.observation_counts().tolist() == [2, 3, 1, 0]

    def test_degenerate_record_detected(self, toy_cohort: Cohort) -> None:
        assert toy_cohort.describe()["degenerate_records"] == 1

    def test_coverage(self, toy_cohort: Cohort) -> None:
        # alpha observed in patients 0,1 -> 0.5 ; beta in 0,2 -> 0.5
        np.testing.assert_allclose(toy_cohort.coverage(), [0.5, 0.5])

    def test_occupancy_matches_mask_mean(self, toy_cohort: Cohort) -> None:
        d = toy_cohort.describe()
        assert d["observed_cells"] == 6
        assert d["grid_occupancy"] == pytest.approx(6 / (4 * 3 * 2))
        assert d["missing_fraction"] == pytest.approx(1 - 6 / (4 * 3 * 2))


class TestSelection:
    def test_select_subsets_every_aligned_array(self, toy_cohort: Cohort) -> None:
        sub = toy_cohort.select(np.array([0, 2], dtype=np.int64))
        assert sub.n_patients == 2
        assert sub.record_ids.tolist() == [0, 2]
        assert sub.source_set.tolist() == ["a", "b"]
        assert sub.labels["mortality"].tolist() == [0.0, 0.0]
        assert sub.observation_counts().tolist() == [2, 1]

    def test_truncate_restricts_window(self, toy_cohort: Cohort) -> None:
        trunc = toy_cohort.truncate(1)
        assert trunc.n_hours == 1
        # patient 1 had 3 observations across 3 hours; only hour 0 survives
        assert trunc.observation_counts().tolist() == [2, 1, 0, 0]

    def test_truncate_rejects_out_of_range(self, toy_cohort: Cohort) -> None:
        with pytest.raises(DataError, match="hours must be"):
            toy_cohort.truncate(0)
        with pytest.raises(DataError, match="hours must be"):
            toy_cohort.truncate(99)

    def test_truncate_does_not_leak_future(self, toy_cohort: Cohort) -> None:
        """Patient 2's only observation is at hour 2 and must vanish at cutoff 2."""
        assert toy_cohort.truncate(2).observation_counts()[2] == 0
        assert toy_cohort.truncate(3).observation_counts()[2] == 1

    def test_variable_index_roundtrip(self, toy_cohort: Cohort) -> None:
        assert toy_cohort.variable_index("beta") == 1
        with pytest.raises(DataError, match="unknown variable"):
            toy_cohort.variable_index("gamma")
