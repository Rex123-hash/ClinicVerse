"""Panel catalogue tests."""

from __future__ import annotations

import numpy as np
import pytest

from cliniverse.acquisition import load_panel_catalogue
from cliniverse.acquisition.catalogue import Panel, PanelCatalogue
from cliniverse.config import VariableConfig
from cliniverse.exceptions import ConfigError

REGIMES = ("uniform_group", "shared_plus_marginal", "ordinal_tier", "per_analyte")


@pytest.fixture(scope="module")
def catalogue() -> PanelCatalogue:
    return load_panel_catalogue()


def test_catalogue_matches_variable_schema(
    catalogue: PanelCatalogue, variable_config: VariableConfig
) -> None:
    """Every lab is purchasable; nothing non-lab is."""
    catalogue.validate_against(variable_config)


def test_panels_partition_the_analytes(catalogue: PanelCatalogue) -> None:
    members = [m for p in catalogue.panels.values() for m in p.members]
    assert len(members) == len(set(members)), "an analyte appears in two panels"


def test_overlapping_panels_rejected() -> None:
    with pytest.raises(ValueError, match="partition"):
        PanelCatalogue(
            version="test",
            panels={
                "a": Panel(name="a", label="A", members=("X", "Y"), cost=1.0),
                "b": Panel(name="b", label="B", members=("Y", "Z"), cost=1.0),
            },
        )


def test_naming_does_not_claim_real_orders(catalogue: PanelCatalogue) -> None:
    """Guard against regressing to names that imply recorded laboratory orders."""
    forbidden = {"BMP", "CBC", "ABG", "LFT"}
    assert not (forbidden & set(catalogue.panel_names))
    for panel in catalogue.panels.values():
        assert "panel" not in panel.label.lower() or "-like" in panel.label.lower()


class TestCostRegimes:
    @pytest.mark.parametrize("regime", REGIMES)
    def test_regime_prices_every_panel(self, catalogue: PanelCatalogue, regime: str) -> None:
        repriced = catalogue.with_schedule(regime)
        assert set(repriced.panels) == set(catalogue.panels)
        assert all(p.cost > 0 for p in repriced.panels.values())
        assert repriced.schedule_name == regime

    def test_unknown_regime_rejected(self, catalogue: PanelCatalogue) -> None:
        with pytest.raises(ConfigError, match="unknown cost schedule"):
            catalogue.with_schedule("free_labs")

    def test_uniform_regime_is_actually_uniform(self, catalogue: PanelCatalogue) -> None:
        costs = {p.cost for p in catalogue.with_schedule("uniform_group").panels.values()}
        assert costs == {1.0}

    def test_per_analyte_regime_matches_panel_sizes(self, catalogue: PanelCatalogue) -> None:
        repriced = catalogue.with_schedule("per_analyte")
        for name, panel in repriced.panels.items():
            assert panel.cost == pytest.approx(float(catalogue.panels[name].size))

    def test_regimes_disagree_on_ordering(self, catalogue: PanelCatalogue) -> None:
        """Sensitivity analysis is only meaningful if the regimes actually differ."""
        uniform = catalogue.with_schedule("uniform_group")
        per_analyte = catalogue.with_schedule("per_analyte")
        assert uniform.cost_of("BMP_like") != per_analyte.cost_of("BMP_like")

    @pytest.mark.parametrize("cost", [0.0, -1.0, np.nan, np.inf])
    def test_invalid_panel_cost_rejected(self, cost: float) -> None:
        with pytest.raises(ValueError):
            Panel(name="bad", label="Bad", members=("X",), cost=cost)

    @pytest.mark.parametrize("cost", [0.0, -1.0, np.nan, np.inf])
    def test_invalid_alternative_schedule_cost_rejected(self, cost: float) -> None:
        catalogue = PanelCatalogue(
            version="test",
            panels={"a": Panel(name="a", label="A", members=("X",), cost=1.0)},
            alternative_schedules={"bad": {"a": cost}},
        )
        with pytest.raises(ConfigError, match="non-positive or non-finite"):
            catalogue.with_schedule("bad")


class TestLookup:
    def test_panel_for_variable(self, catalogue: PanelCatalogue) -> None:
        assert catalogue.panel_for("Creatinine").name == "BMP_like"
        assert catalogue.panel_for("Platelets").name == "CBC_like"
        assert catalogue.panel_for("pH").name == "ABG_like"

    def test_sao2_is_a_singleton_not_part_of_abg(self, catalogue: PanelCatalogue) -> None:
        """Empirically separated from the ABG-like subset; see E-001."""
        assert catalogue.panel_for("SaO2").is_singleton
        assert "SaO2" not in catalogue.panels["ABG_like"].members

    def test_unknown_variable_raises(self, catalogue: PanelCatalogue) -> None:
        with pytest.raises(ConfigError, match="not in any panel"):
            catalogue.panel_for("HR")  # continuously monitored, not orderable

    def test_duplicate_purchase_charged_once(self, catalogue: PanelCatalogue) -> None:
        single = catalogue.total_cost(("BMP_like",))
        repeated = catalogue.total_cost(("BMP_like", "BMP_like"))
        assert repeated == pytest.approx(single)

    def test_total_cost_sums_distinct_panels(self, catalogue: PanelCatalogue) -> None:
        expected = catalogue.cost_of("BMP_like") + catalogue.cost_of("CBC_like")
        assert catalogue.total_cost(("BMP_like", "CBC_like")) == pytest.approx(expected)
