"""PhysioNet 2012 parser and loader tests.

The `slow` tests pin the loader against statistics independently verified from
the raw files by `scripts/verify_physionet2012.py`. If the parser silently
changes behaviour, these fail.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from cliniverse.config import VariableConfig
from cliniverse.data.cohort import Cohort
from cliniverse.data.physionet2012 import (
    LEAKY_OUTCOME_COLUMNS,
    cohort_fingerprint,
    parse_record,
)
from cliniverse.exceptions import DataError

# Verified from the raw files on 2026-08-09 (see docs/EXPERIMENTS.md E-000).
SET_A_RECORDS = 4000
SET_A_DEATHS = 554
SET_A_DEGENERATE = {140501, 140936, 141264}
N_TIMESERIES_VARIABLES = 36
RAW_OCCUPANCY_UPPER_BOUND = 0.2328


# --------------------------------------------------------------- parsing ----
class TestParseRecord:
    def _write(self, tmp_path: pathlib.Path, body: str) -> pathlib.Path:
        path = tmp_path / "132539.txt"
        path.write_text("Time,Parameter,Value\n" + body, encoding="utf-8")
        return path

    def test_parses_statics_and_observations(
        self, tmp_path: pathlib.Path, variable_config: VariableConfig
    ) -> None:
        path = self._write(
            tmp_path,
            "00:00,RecordID,132539\n"
            "00:00,Age,54\n"
            "00:00,Gender,0\n"
            "00:00,Height,-1\n"
            "00:07,HR,73\n"
            "01:30,HR,80\n",
        )
        statics, obs = parse_record(path, variable_config)
        assert statics == {"Age": 54.0, "Gender": 0.0}  # Height sentinel dropped
        assert obs == [(0, "HR", 73.0), (1, "HR", 80.0)]

    def test_record_id_is_not_a_measurement(
        self, tmp_path: pathlib.Path, variable_config: VariableConfig
    ) -> None:
        path = self._write(tmp_path, "00:00,RecordID,132539\n")
        statics, obs = parse_record(path, variable_config)
        assert statics == {} and obs == []

    def test_observations_beyond_horizon_dropped(
        self, tmp_path: pathlib.Path, variable_config: VariableConfig
    ) -> None:
        path = self._write(tmp_path, "47:59,HR,70\n48:00,HR,71\n99:00,HR,72\n")
        _, obs = parse_record(path, variable_config)
        assert obs == [(47, "HR", 70.0)]

    def test_implausible_values_dropped(
        self, tmp_path: pathlib.Path, variable_config: VariableConfig
    ) -> None:
        path = self._write(tmp_path, "00:10,HR,73\n00:20,HR,400\n00:30,HR,-3\n")
        _, obs = parse_record(path, variable_config)
        assert obs == [(0, "HR", 73.0)]

    def test_unit_repairs_applied_during_parse(
        self, tmp_path: pathlib.Path, variable_config: VariableConfig
    ) -> None:
        path = self._write(tmp_path, "00:10,Temp,98.6\n00:20,pH,735\n00:30,FiO2,40\n")
        _, obs = parse_record(path, variable_config)
        by_name = {name: value for _, name, value in obs}
        assert by_name["Temp"] == pytest.approx(37.0, abs=1e-6)
        assert by_name["pH"] == pytest.approx(7.35)
        assert by_name["FiO2"] == pytest.approx(0.40)

    def test_malformed_rows_tolerated(
        self, tmp_path: pathlib.Path, variable_config: VariableConfig
    ) -> None:
        path = self._write(
            tmp_path,
            "00:10,HR,73\ngarbage\n00:20,HR,notanumber\nbad:time,HR,80\n00:30,HR,75\n",
        )
        _, obs = parse_record(path, variable_config)
        assert obs == [(0, "HR", 73.0), (0, "HR", 75.0)]

    def test_empty_file_yields_nothing(
        self, tmp_path: pathlib.Path, variable_config: VariableConfig
    ) -> None:
        path = tmp_path / "1.txt"
        path.write_text("", encoding="utf-8")
        assert parse_record(path, variable_config) == ({}, [])

    def test_bad_header_rejected(
        self, tmp_path: pathlib.Path, variable_config: VariableConfig
    ) -> None:
        path = tmp_path / "1.txt"
        path.write_text("a,b,c\n", encoding="utf-8")
        with pytest.raises(DataError, match="unexpected header"):
            parse_record(path, variable_config)


# ------------------------------------------------------- real dataset ----
@pytest.mark.slow
class TestSetA:
    def test_record_and_label_counts(self, cohort_a: Cohort) -> None:
        assert cohort_a.n_patients == SET_A_RECORDS
        assert int(cohort_a.labels["mortality"].sum()) == SET_A_DEATHS

    def test_variable_count(self, cohort_a: Cohort) -> None:
        assert cohort_a.n_variables == N_TIMESERIES_VARIABLES
        assert cohort_a.n_hours == 48

    def test_degenerate_records_retained_not_dropped(self, cohort_a: Cohort) -> None:
        empty = set(cohort_a.record_ids[cohort_a.observation_counts() == 0].tolist())
        assert empty == SET_A_DEGENERATE

    def test_occupancy_below_raw_upper_bound(self, cohort_a: Cohort) -> None:
        """Binned occupancy must sit below the raw row-count bound.

        Raw rows can collide within an hour-cell, and implausible values are
        dropped, so binned occupancy is strictly lower. A value above the bound
        would mean the binner is inventing observations.
        """
        occupancy = cohort_a.describe()["grid_occupancy"]
        assert 0.15 < occupancy < RAW_OCCUPANCY_UPPER_BOUND

    def test_mask_matches_finite_values(self, cohort_a: Cohort) -> None:
        assert np.array_equal(cohort_a.m, np.isfinite(cohort_a.x))

    def test_all_values_within_plausible_bounds(
        self, cohort_a: Cohort, variable_config: VariableConfig
    ) -> None:
        for i, name in enumerate(cohort_a.variable_names):
            column = cohort_a.x[:, :, i]
            observed = column[np.isfinite(column)]
            if observed.size == 0:
                continue
            lo, hi = variable_config.variables[name].plausible
            assert observed.min() >= lo, f"{name} below bound"
            assert observed.max() <= hi, f"{name} above bound"

    def test_loading_is_deterministic(
        self, cohort_a: Cohort, data_cache: pathlib.Path
    ) -> None:
        from cliniverse.data.physionet2012 import load_cohort

        again = load_cohort(data_cache, sets=("a",), download=False)
        assert cohort_fingerprint(again) == cohort_fingerprint(cohort_a)

    def test_prolonged_stay_label_is_derived_correctly(self, cohort_a: Cohort) -> None:
        los = cohort_a.labels["length_of_stay"]
        prolonged = cohort_a.labels["prolonged_stay"]
        known = los >= 0
        np.testing.assert_array_equal(prolonged[known], (los[known] > 3).astype(np.float32))
        # The -1 sentinel must become NaN, not a spurious "short stay".
        assert bool(np.isnan(prolonged[~known]).all())


# -------------------------------------------------------------- leakage ----
class TestLeakageGuards:
    def test_severity_scores_are_not_loaded_as_features(self, toy_cohort: Cohort) -> None:
        """SAPS-I/SOFA are computed from the same 48h window as the features.

        They are legitimate *baselines* but would be leakage as model inputs, so
        they must never appear as cohort variables or labels.
        """
        assert {"SAPS-I", "SOFA", "Survival"} == LEAKY_OUTCOME_COLUMNS
        for column in LEAKY_OUTCOME_COLUMNS:
            assert column not in toy_cohort.variable_names
            assert column not in toy_cohort.static_names
            assert column not in toy_cohort.labels

    @pytest.mark.slow
    def test_real_cohort_carries_no_severity_scores(self, cohort_a: Cohort) -> None:
        for column in LEAKY_OUTCOME_COLUMNS:
            assert column not in cohort_a.variable_names
            assert column not in cohort_a.static_names
            assert column not in cohort_a.labels
