"""Temporal-leakage regression tests.

These exist because of a confirmed, severe bug: `Weight` is recorded throughout
the ICU stay (95.9% of set-a Weight rows are after hour 0, 52.1% at hour >= 24),
but was classified as a static descriptor. A model with a 24h cutoff therefore
received weights measured as late as hour 47.

The tests below are deliberately *general* rather than Weight-specific: they
assert the property "no model input may depend on data after the cutoff" so that
any future variable with the same shape is caught automatically.
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import pytest

from cliniverse.config import VariableConfig
from cliniverse.data.cohort import Cohort
from cliniverse.data.physionet2012 import parse_record
from cliniverse.encoders import build_features


class TestStaticRoutingIsTimeGated:
    """A static field may only be filled from its pinned hour."""

    def _write(self, tmp_path: pathlib.Path, body: str) -> pathlib.Path:
        path = tmp_path / "132539.txt"
        path.write_text("Time,Parameter,Value\n" + body, encoding="utf-8")
        return path

    def test_late_weight_is_timeseries_not_static(
        self, tmp_path: pathlib.Path, variable_config: VariableConfig
    ) -> None:
        path = self._write(tmp_path, "00:00,Age,54\n30:00,Weight,80\n")
        statics, obs = parse_record(path, variable_config)
        assert "AdmissionWeight" not in statics, "post-admission Weight leaked into statics"
        assert (30, "Weight", 80.0) in obs

    def test_hour_zero_weight_is_both_static_and_observation(
        self, tmp_path: pathlib.Path, variable_config: VariableConfig
    ) -> None:
        """An admission weight is a legitimate static *and* a real hour-0 reading."""
        path = self._write(tmp_path, "00:00,Weight,75\n")
        statics, obs = parse_record(path, variable_config)
        assert statics["AdmissionWeight"] == pytest.approx(75.0)
        assert (0, "Weight", 75.0) in obs

    def test_later_weight_does_not_overwrite_admission_weight(
        self, tmp_path: pathlib.Path, variable_config: VariableConfig
    ) -> None:
        path = self._write(tmp_path, "00:00,Weight,75\n36:00,Weight,91\n")
        statics, obs = parse_record(path, variable_config)
        assert statics["AdmissionWeight"] == pytest.approx(75.0)
        assert sorted(o[0] for o in obs if o[1] == "Weight") == [0, 36]

    def test_every_static_is_pinned_to_hour_zero(
        self, variable_config: VariableConfig
    ) -> None:
        for raw, (field, hour) in variable_config.static_sources.items():
            assert hour == 0, f"static {field} sourced from {raw} is not pinned to hour 0"

    def test_weight_is_a_timeseries_variable(self, variable_config: VariableConfig) -> None:
        """Regression guard: reclassifying Weight as static would reintroduce the leak."""
        assert "Weight" in variable_config.variables
        assert "Weight" not in variable_config.statics


class TestTruncationRemovesFuture:
    def test_truncate_drops_all_post_cutoff_observations(self, toy_cohort: Cohort) -> None:
        cutoff = 2
        trunc = toy_cohort.truncate(cutoff)
        assert trunc.x.shape[1] == cutoff
        # Every surviving observation must come from before the cutoff.
        assert np.array_equal(trunc.m, toy_cohort.m[:, :cutoff, :])
        assert np.array_equal(
            np.nan_to_num(trunc.x, nan=-1.0),
            np.nan_to_num(toy_cohort.x[:, :cutoff, :], nan=-1.0),
        )

    def test_every_summary_feature_is_invariant_to_post_cutoff_changes(
        self, toy_cohort: Cohort
    ) -> None:
        cutoff = 2
        changed_x = toy_cohort.x.copy()
        changed_m = toy_cohort.m.copy()
        changed_x[:, cutoff:, :] = 12345.0
        changed_m[:, cutoff:, :] = True
        changed = dataclasses.replace(toy_cohort, x=changed_x, m=changed_m)

        original_features = build_features(toy_cohort.truncate(cutoff))
        changed_features = build_features(changed.truncate(cutoff))
        np.testing.assert_array_equal(
            np.nan_to_num(original_features.x, nan=-9999.0),
            np.nan_to_num(changed_features.x, nan=-9999.0),
        )


@pytest.mark.slow
class TestRealCohortHasNoPostCutoffLeak:
    def test_admission_weight_differs_from_late_weight(self, cohort_a: Cohort) -> None:
        """AdmissionWeight must not track weights recorded after the cutoff.

        If AdmissionWeight were still absorbing the last observed weight, it
        would frequently equal a post-cutoff time-series value while differing
        from the hour-0 value. Here it must equal the hour-0 reading whenever one
        exists.
        """
        w = cohort_a.variable_index("Weight")
        aw = cohort_a.static_names.index("AdmissionWeight")

        hour0_observed = cohort_a.m[:, 0, w]
        hour0_value = cohort_a.x[:, 0, w]
        admission = cohort_a.statics[:, aw]

        both = hour0_observed & cohort_a.statics_mask[:, aw]
        assert bool(both.any()), "no patient has both an hour-0 weight and AdmissionWeight"
        np.testing.assert_allclose(
            admission[both],
            hour0_value[both],
            rtol=1e-5,
            err_msg="AdmissionWeight does not match the hour-0 Weight reading",
        )

    def test_no_admission_weight_without_hour_zero_reading(self, cohort_a: Cohort) -> None:
        w = cohort_a.variable_index("Weight")
        aw = cohort_a.static_names.index("AdmissionWeight")
        has_static = cohort_a.statics_mask[:, aw]
        has_hour0 = cohort_a.m[:, 0, w]
        orphans = int((has_static & ~has_hour0).sum())
        assert orphans == 0, (
            f"{orphans} patients have AdmissionWeight with no hour-0 Weight reading, "
            "meaning a later value was routed into the static vector"
        )

    def test_truncated_cohort_carries_no_late_weight(self, cohort_a: Cohort) -> None:
        """The decisive end-to-end check at the 24h decision point."""
        truncated = cohort_a.truncate(24)
        w = truncated.variable_index("Weight")
        assert truncated.m[:, :, w].shape[1] == 24
        # Statics must be reachable from data at or before the cutoff only.
        aw = truncated.static_names.index("AdmissionWeight")
        has_static = truncated.statics_mask[:, aw]
        has_hour0 = truncated.m[:, 0, w]
        assert int((has_static & ~has_hour0).sum()) == 0
