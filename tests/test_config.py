"""Variable-schema and unit-repair tests.

The repair rules silently rewrite data, so each one is pinned by a test. A
regression here would corrupt every downstream number without any error.
"""

from __future__ import annotations

import pytest

from cliniverse.config import VariableConfig, load_variable_config
from cliniverse.exceptions import ConfigError


def test_config_loads_expected_shape(variable_config: VariableConfig) -> None:
    assert variable_config.dataset == "physionet-cinc-2012"
    assert variable_config.horizon_hours == 48
    assert len(variable_config.variables) == 36
    assert len(variable_config.statics) == 5


def test_variable_order_is_sorted_and_stable(variable_config: VariableConfig) -> None:
    names = variable_config.variable_names
    assert names == tuple(sorted(names))
    assert names == load_variable_config().variable_names


def test_missing_config_file_raises() -> None:
    import pathlib

    with pytest.raises(ConfigError, match="not found"):
        load_variable_config(pathlib.Path("configs/does-not-exist.yaml"))


class TestRepairs:
    def test_fahrenheit_temperature_converted(self, variable_config: VariableConfig) -> None:
        # 98.6 F is 37.0 C
        result = variable_config.clean("Temp", 98.6)
        assert result is not None
        assert result == pytest.approx(37.0, abs=1e-6)

    def test_celsius_temperature_untouched(self, variable_config: VariableConfig) -> None:
        assert variable_config.clean("Temp", 37.0) == pytest.approx(37.0)

    def test_temperature_repair_band_does_not_swallow_valid_celsius(
        self, variable_config: VariableConfig
    ) -> None:
        # The repair band starts at 90, above the plausible Celsius ceiling (45),
        # so no genuine Celsius reading can be mistakenly converted.
        assert variable_config.clean("Temp", 45.0) == pytest.approx(45.0)
        assert variable_config.clean("Temp", 46.0) is None  # implausible, dropped

    def test_ph_recorded_times_100(self, variable_config: VariableConfig) -> None:
        result = variable_config.clean("pH", 735.0)
        assert result is not None
        assert result == pytest.approx(7.35)

    def test_ph_recorded_divided_by_10(self, variable_config: VariableConfig) -> None:
        result = variable_config.clean("pH", 0.735)
        assert result is not None
        assert result == pytest.approx(7.35)

    def test_ph_normal_untouched(self, variable_config: VariableConfig) -> None:
        assert variable_config.clean("pH", 7.4) == pytest.approx(7.4)

    def test_fio2_percentage_converted(self, variable_config: VariableConfig) -> None:
        result = variable_config.clean("FiO2", 40.0)
        assert result is not None
        assert result == pytest.approx(0.40)

    def test_fio2_fraction_untouched(self, variable_config: VariableConfig) -> None:
        assert variable_config.clean("FiO2", 0.5) == pytest.approx(0.5)


class TestPlausibility:
    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("HR", 400.0),  # impossible heart rate
            ("HR", -5.0),
            ("Na", 20.0),  # incompatible with life
            ("GCS", 2.0),  # GCS floor is 3
            ("GCS", 16.0),
            ("pH", 5.0),
            ("Platelets", 5000.0),
        ],
    )
    def test_implausible_dropped(
        self, variable_config: VariableConfig, name: str, value: float
    ) -> None:
        assert variable_config.clean(name, value) is None

    @pytest.mark.parametrize(
        ("name", "value"),
        [("HR", 72.0), ("Na", 140.0), ("GCS", 15.0), ("GCS", 3.0), ("Creatinine", 1.1)],
    )
    def test_plausible_kept(
        self, variable_config: VariableConfig, name: str, value: float
    ) -> None:
        assert variable_config.clean(name, value) == pytest.approx(value)

    def test_extreme_but_real_values_survive(self, variable_config: VariableConfig) -> None:
        # ICU cohorts contain genuine extremes; bounds must not winsorize them.
        assert variable_config.clean("Lactate", 18.0) == pytest.approx(18.0)
        assert variable_config.clean("Creatinine", 12.0) == pytest.approx(12.0)
        assert variable_config.clean("Glucose", 800.0) == pytest.approx(800.0)

    def test_static_missing_sentinel(self, variable_config: VariableConfig) -> None:
        assert variable_config.clean("Height", -1.0) is None
        assert variable_config.clean("Gender", -1.0) is None

    def test_unknown_variable_returns_none(self, variable_config: VariableConfig) -> None:
        assert variable_config.clean("NotAVariable", 1.0) is None
