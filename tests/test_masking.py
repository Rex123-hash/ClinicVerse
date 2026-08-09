"""Masking mechanism tests.

The properties that matter: only observed cells may be hidden, mechanisms are
deterministic given (seed, patient), a patient's mask does not depend on
processing order, and panel-event masking hides co-measured groups as units.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from cliniverse.acquisition.catalogue import Panel, PanelCatalogue
from cliniverse.exceptions import ConfigError
from twinbench.masking import (
    MECHANISMS,
    GroupHours,
    MaskingMechanism,
    McarCells,
    TimeBlocks,
    build_mechanism,
)

CATALOGUE = PanelCatalogue(
    version="test",
    panels={
        "alpha": Panel(name="alpha", label="Alpha", members=("a1", "a2"), cost=1.0),
        "beta": Panel(name="beta", label="Beta", members=("b1", "b2"), cost=1.0),
    },
)
VARIABLES = ("a1", "a2", "b1", "b2")


def observed_all(t: int = 12, v: int = 4) -> np.ndarray:
    return np.ones((t, v), dtype=bool)


ALL_MECHANISMS = [
    McarCells(rate=0.5, seed=1),
    GroupHours(rate=0.5, seed=1),
    TimeBlocks(rate=0.5, seed=1, block_hours=3),
]


@pytest.mark.parametrize("mech", ALL_MECHANISMS, ids=lambda m: m.mechanism_id)
class TestUniversalProperties:
    def test_only_observed_cells_can_be_hidden(self, mech: MaskingMechanism) -> None:
        observed = np.zeros((12, 4), dtype=bool)
        observed[:6, :2] = True
        hidden = mech.hidden_for(observed, 0, CATALOGUE, VARIABLES)
        assert not bool((hidden & ~observed).any()), "hid a cell that was never observed"

    def test_deterministic_for_same_seed_and_patient(self, mech: MaskingMechanism) -> None:
        observed = observed_all()
        a = mech.hidden_for(observed, 3, CATALOGUE, VARIABLES)
        b = mech.hidden_for(observed, 3, CATALOGUE, VARIABLES)
        np.testing.assert_array_equal(a, b)

    def test_independent_of_processing_order(self, mech: MaskingMechanism) -> None:
        """Patient 7's mask is the same whether or not others were drawn first."""
        observed = observed_all()
        direct = mech.hidden_for(observed, 7, CATALOGUE, VARIABLES)
        for i in range(7):
            mech.hidden_for(observed, i, CATALOGUE, VARIABLES)
        after = mech.hidden_for(observed, 7, CATALOGUE, VARIABLES)
        np.testing.assert_array_equal(direct, after)

    def test_different_patients_get_different_masks(self, mech: MaskingMechanism) -> None:
        observed = observed_all()
        masks = [mech.hidden_for(observed, i, CATALOGUE, VARIABLES) for i in range(8)]
        assert any(not np.array_equal(masks[0], m) for m in masks[1:])

    def test_zero_rate_hides_nothing(self, mech: MaskingMechanism) -> None:
        # dataclasses.replace, not `mech.__dict__`: these are slotted dataclasses.
        zero = dataclasses.replace(mech, rate=0.0)
        hidden = zero.hidden_for(observed_all(), 0, CATALOGUE, VARIABLES)
        assert hidden.sum() == 0

    def test_shape_preserved(self, mech: MaskingMechanism) -> None:
        observed = observed_all(t=9, v=4)
        assert mech.hidden_for(observed, 0, CATALOGUE, VARIABLES).shape == observed.shape

    def test_rejects_non_2d_input(self, mech: MaskingMechanism) -> None:
        with pytest.raises(ConfigError, match=r"\(T, V\)"):
            mech.hidden_for(np.ones((2, 3, 4), dtype=bool), 0, CATALOGUE, VARIABLES)


class TestGroupHours:
    def test_hides_whole_groups_together(self) -> None:
        """A group measured in one hour must vanish as a unit, not analyte-wise."""
        mech = GroupHours(rate=1.0, seed=5)
        observed = observed_all()
        hidden = mech.hidden_for(observed, 0, CATALOGUE, VARIABLES)
        # a1 and a2 belong to 'alpha'; their hidden patterns must be identical.
        np.testing.assert_array_equal(hidden[:, 0], hidden[:, 1])
        np.testing.assert_array_equal(hidden[:, 2], hidden[:, 3])

    def test_rate_one_hides_every_measured_event(self) -> None:
        mech = GroupHours(rate=1.0, seed=5)
        observed = observed_all()
        hidden = mech.hidden_for(observed, 0, CATALOGUE, VARIABLES)
        np.testing.assert_array_equal(hidden, observed)

    def test_group_with_no_observations_is_untouched(self) -> None:
        observed = observed_all()
        observed[:, 2:] = False  # beta never measured
        hidden = GroupHours(rate=1.0, seed=5).hidden_for(observed, 0, CATALOGUE, VARIABLES)
        assert hidden[:, 2:].sum() == 0

    def test_partially_measured_group_hides_only_measured_cells(self) -> None:
        observed = observed_all()
        observed[3, 1] = False  # a2 missing at hour 3, a1 present
        hidden = GroupHours(rate=1.0, seed=5).hidden_for(observed, 0, CATALOGUE, VARIABLES)
        assert hidden[3, 0]
        assert not hidden[3, 1]

    def test_approximate_rate(self) -> None:
        mech = GroupHours(rate=0.3, seed=11)
        observed = observed_all(t=48)
        fractions = [
            mech.hidden_for(observed, i, CATALOGUE, VARIABLES).mean() for i in range(200)
        ]
        assert 0.25 < float(np.mean(fractions)) < 0.35


class TestMcarCells:
    def test_cells_hidden_independently(self) -> None:
        """Unlike group_hours, group members should differ."""
        hidden = McarCells(rate=0.5, seed=7).hidden_for(
            observed_all(t=48), 0, CATALOGUE, VARIABLES
        )
        assert not np.array_equal(hidden[:, 0], hidden[:, 1])

    def test_approximate_rate(self) -> None:
        mech = McarCells(rate=0.4, seed=3)
        observed = observed_all(t=48)
        fractions = [
            mech.hidden_for(observed, i, CATALOGUE, VARIABLES).mean() for i in range(100)
        ]
        assert 0.35 < float(np.mean(fractions)) < 0.45


class TestTimeBlocks:
    def test_hides_contiguous_hours_across_all_variables(self) -> None:
        hidden = TimeBlocks(rate=1.0, seed=2, block_hours=3).hidden_for(
            observed_all(t=12), 0, CATALOGUE, VARIABLES
        )
        np.testing.assert_array_equal(hidden, observed_all(t=12))
        # Every hour is either fully hidden or fully visible.
        per_hour = hidden.sum(axis=1)
        assert set(np.unique(per_hour).tolist()) <= {0, 4}

    def test_rejects_bad_block_length(self) -> None:
        with pytest.raises(ConfigError, match="block_hours"):
            TimeBlocks(rate=0.5, seed=1, block_hours=0)

    def test_partial_final_block_is_not_skipped(self) -> None:
        hidden = TimeBlocks(rate=1.0, seed=2, block_hours=3).hidden_for(
            observed_all(t=10), 0, CATALOGUE, VARIABLES
        )
        np.testing.assert_array_equal(hidden, observed_all(t=10))


class TestFactory:
    @pytest.mark.parametrize("name", sorted(MECHANISMS))
    def test_build_every_registered_mechanism(self, name: str) -> None:
        mech = build_mechanism(name, rate=0.25, seed=1)
        assert mech.mechanism_id.startswith(name)
        assert mech.rate == 0.25

    def test_unknown_mechanism_rejected(self) -> None:
        with pytest.raises(ConfigError, match="unknown masking mechanism"):
            build_mechanism("telepathy", rate=0.5, seed=1)

    def test_invalid_rate_rejected(self) -> None:
        with pytest.raises(ConfigError, match=r"rate must be in \[0, 1\]"):
            build_mechanism("mcar_cells", rate=1.5, seed=1)

    @pytest.mark.parametrize("rate", [np.nan, np.inf, -np.inf])
    def test_non_finite_rate_rejected(self, rate: float) -> None:
        with pytest.raises(ConfigError, match="rate must be"):
            build_mechanism("mcar_cells", rate=rate, seed=1)

    def test_mechanism_id_is_stable_and_distinguishing(self) -> None:
        assert build_mechanism("mcar_cells", 0.3, 1).mechanism_id != (
            build_mechanism("mcar_cells", 0.3, 2).mechanism_id
        )
        assert build_mechanism("mcar_cells", 0.3, 1).mechanism_id == (
            build_mechanism("mcar_cells", 0.3, 1).mechanism_id
        )
