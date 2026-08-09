"""Disclosure engine tests, with emphasis on the support-oracle invariants.

E-002 established that measurement-presence patterns alone reach AUROC 0.7224 on
this cohort. A policy that can tell which groups hold hidden values therefore has
access to a strong predictor that has nothing to do with acquiring information.
These tests are what stop that happening.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from cliniverse.acquisition.catalogue import Panel, PanelCatalogue
from cliniverse.data.cohort import Cohort
from cliniverse.exceptions import BudgetError, ConfigError
from twinbench.disclosure import DisclosureEngine, PolicyView, Protocol

# Two groups over four variables, so a "panel" is a real set, not a synonym for
# a single analyte.
CATALOGUE = PanelCatalogue(
    version="test",
    panels={
        "alpha": Panel(name="alpha", label="Alpha", members=("a1", "a2"), cost=1.0),
        "beta": Panel(name="beta", label="Beta", members=("b1", "b2"), cost=2.0),
    },
)
VARIABLES = ("a1", "a2", "b1", "b2")
EPOCHS = (2, 4)


def make_cohort(observed: np.ndarray, values: np.ndarray | None = None) -> Cohort:
    """One patient, 4 hours, 4 variables, from an explicit observation mask."""
    m = observed.astype(bool)[None, :, :]
    if values is None:
        values = np.arange(observed.size, dtype=np.float32).reshape(observed.shape) + 1.0
    x = np.where(m[0], values.astype(np.float32), np.nan)[None, :, :]
    return Cohort(
        record_ids=np.array([1], dtype=np.int64),
        source_set=np.array(["a"], dtype=np.str_),
        x=x.astype(np.float32),
        m=m,
        statics=np.zeros((1, 1), dtype=np.float32),
        statics_mask=np.ones((1, 1), dtype=bool),
        labels={"mortality": np.array([1.0], dtype=np.float32)},
        variable_names=VARIABLES,
        static_names=("Age",),
    )


@pytest.fixture
def full_cohort() -> Cohort:
    return make_cohort(np.ones((4, 4)))


def engine(
    cohort: Cohort,
    hidden: np.ndarray,
    *,
    budget: float = 10.0,
    protocol: Protocol = Protocol.SUPPORT_BLIND,
) -> DisclosureEngine:
    return DisclosureEngine(
        cohort,
        0,
        hidden.astype(bool),
        CATALOGUE,
        budget=budget,
        protocol=protocol,
        epoch_hours=EPOCHS,
    )


# --------------------------------------------------------- SO-1: no oracle ---
class TestPolicyViewCarriesNoEvaluatorState:
    def test_view_has_no_reference_to_engine_or_hidden_set(self, full_cohort: Cohort) -> None:
        """SO-1: no attribute path from the view back to evaluator state."""
        hidden = np.zeros((4, 4), dtype=bool)
        hidden[0, 0] = True
        view = engine(full_cohort, hidden).view()

        exposed = {f.name for f in dataclasses.fields(view)}
        forbidden = {
            "hidden",
            "_hidden",
            "observed",
            "_observed",
            "engine",
            "_engine",
            "support",
            "cohort",
            "labels",
            "targets",
        }
        assert not (exposed & forbidden)

        # Nothing reachable on the view may be a DisclosureEngine.
        for name in exposed:
            assert not isinstance(getattr(view, name), DisclosureEngine)

    def test_disclosed_mask_hides_the_hidden_cells(self, full_cohort: Cohort) -> None:
        hidden = np.zeros((4, 4), dtype=bool)
        hidden[0, 0] = True
        view = engine(full_cohort, hidden).view()
        assert not view.disclosed_mask[0, 0]
        assert np.isnan(view.disclosed_values[0, 0])

    def test_view_never_shows_beyond_the_boundary(self, full_cohort: Cohort) -> None:
        """BR-1: at epoch 0 the boundary is hour 2, so hours 2-3 are invisible."""
        view = engine(full_cohort, np.zeros((4, 4), dtype=bool)).view()
        assert view.boundary_hour == 2
        assert not view.disclosed_mask[2:, :].any()
        assert bool(np.isnan(view.disclosed_values[2:, :]).all())

    def test_advancing_epoch_extends_the_boundary(self, full_cohort: Cohort) -> None:
        eng = engine(full_cohort, np.zeros((4, 4), dtype=bool))
        before = eng.view().n_disclosed
        assert eng.advance_epoch() is True
        after = eng.view()
        assert after.boundary_hour == 4
        assert after.n_disclosed > before
        assert eng.advance_epoch() is False  # no epoch beyond the last


class TestIndistinguishability:
    """SO-2: hidden-bearing and empty groups look identical before purchase."""

    def test_support_blind_requestable_set_is_constant(self) -> None:
        """The requestable set must not vary with what is actually available."""
        rich = make_cohort(np.ones((4, 4)))
        # A patient where 'beta' was never measured at all.
        sparse_mask = np.ones((4, 4))
        sparse_mask[:, 2:] = 0
        sparse = make_cohort(sparse_mask)

        hidden_rich = np.zeros((4, 4), dtype=bool)
        hidden_rich[0, 2] = True  # beta has hidden data
        hidden_sparse = np.zeros((4, 4), dtype=bool)  # beta has nothing

        a = engine(rich, hidden_rich).view().requestable
        b = engine(sparse, hidden_sparse).view().requestable
        assert a == b == CATALOGUE.panel_names

    def test_support_blind_view_identical_when_only_availability_differs(self) -> None:
        """Two patients differing only in hidden availability yield equal views."""
        values = np.ones((4, 4), dtype=np.float32)
        rich = make_cohort(np.ones((4, 4)), values)
        sparse_mask = np.ones((4, 4))
        sparse_mask[0, 2] = 0  # b1 at hour 0 never measured
        sparse = make_cohort(sparse_mask, values)

        hidden_rich = np.zeros((4, 4), dtype=bool)
        hidden_rich[0, 2] = True  # measured but hidden -> policy sees nothing there
        hidden_sparse = np.zeros((4, 4), dtype=bool)  # never measured -> also nothing

        v1 = engine(rich, hidden_rich).view()
        v2 = engine(sparse, hidden_sparse).view()

        np.testing.assert_array_equal(v1.disclosed_mask, v2.disclosed_mask)
        np.testing.assert_array_equal(
            np.nan_to_num(v1.disclosed_values, nan=-1.0),
            np.nan_to_num(v2.disclosed_values, nan=-1.0),
        )
        assert v1.requestable == v2.requestable

    def test_support_aware_deliberately_leaks_availability(self) -> None:
        """The support-aware protocol is *meant* to leak; that is what it models."""
        cohort = make_cohort(np.ones((4, 4)))
        only_alpha = np.zeros((4, 4), dtype=bool)
        only_alpha[0, 0] = True
        aware = engine(cohort, only_alpha, protocol=Protocol.SUPPORT_AWARE)
        assert aware.view().requestable == ("alpha",)
        blind = engine(cohort, only_alpha, protocol=Protocol.SUPPORT_BLIND)
        assert set(blind.view().requestable) == {"alpha", "beta"}


# ------------------------------------------------------------- purchasing ---
class TestPurchasing:
    def test_purchase_discloses_hidden_cells_of_that_group_only(
        self, full_cohort: Cohort
    ) -> None:
        hidden = np.zeros((4, 4), dtype=bool)
        hidden[0, 0] = True  # a1, group alpha
        hidden[0, 2] = True  # b1, group beta
        eng = engine(full_cohort, hidden)

        eng.request("alpha")
        view = eng.view()
        assert view.disclosed_mask[0, 0], "alpha's hidden cell should be disclosed"
        assert not view.disclosed_mask[0, 2], "beta's cell must stay hidden"

    def test_empty_request_costs_full_price(self, full_cohort: Cohort) -> None:
        """SO-2's enforcement: probing availability is never free."""
        eng = engine(full_cohort, np.zeros((4, 4), dtype=bool))
        purchase = eng.request("beta")
        assert purchase.n_disclosed == 0
        assert purchase.was_empty
        assert purchase.cost == pytest.approx(2.0)
        assert eng.spent == pytest.approx(2.0)

    def test_successful_and_empty_requests_cost_the_same(self, full_cohort: Cohort) -> None:
        hidden = np.zeros((4, 4), dtype=bool)
        hidden[0, 0] = True
        productive = engine(full_cohort, hidden).request("alpha")
        empty = engine(full_cohort, np.zeros((4, 4), dtype=bool)).request("alpha")
        assert productive.cost == empty.cost
        assert productive.n_disclosed > 0 and empty.n_disclosed == 0

    def test_purchase_cannot_disclose_beyond_the_boundary(self, full_cohort: Cohort) -> None:
        """BR-1 holds through purchases, not just the initial view."""
        hidden = np.zeros((4, 4), dtype=bool)
        hidden[3, 0] = True  # a1 at hour 3, beyond the epoch-0 boundary of 2
        eng = engine(full_cohort, hidden)
        purchase = eng.request("alpha")
        assert purchase.n_disclosed == 0
        assert not eng.view().disclosed_mask[3, 0]

        # After advancing, the same cell becomes purchasable.
        eng.advance_epoch()
        assert eng.request("alpha").n_disclosed == 1

    def test_repeat_purchase_costs_again_and_discloses_nothing(
        self, full_cohort: Cohort
    ) -> None:
        hidden = np.zeros((4, 4), dtype=bool)
        hidden[0, 0] = True
        eng = engine(full_cohort, hidden)
        first = eng.request("alpha")
        second = eng.request("alpha")
        assert first.n_disclosed == 1
        assert second.n_disclosed == 0
        assert eng.spent == pytest.approx(2.0)

    def test_budget_is_enforced(self, full_cohort: Cohort) -> None:
        eng = engine(full_cohort, np.zeros((4, 4), dtype=bool), budget=1.5)
        eng.request("alpha")  # costs 1.0
        with pytest.raises(BudgetError, match="only"):
            eng.request("beta")  # costs 2.0, only 0.5 remains

    def test_unrequestable_panel_rejected_under_support_aware(
        self, full_cohort: Cohort
    ) -> None:
        only_alpha = np.zeros((4, 4), dtype=bool)
        only_alpha[0, 0] = True
        eng = engine(full_cohort, only_alpha, protocol=Protocol.SUPPORT_AWARE)
        with pytest.raises(BudgetError, match="not requestable"):
            eng.request("beta")

    def test_unknown_panel_rejected(self, full_cohort: Cohort) -> None:
        eng = engine(full_cohort, np.zeros((4, 4), dtype=bool))
        with pytest.raises(ConfigError, match="unknown panel"):
            eng.request("gamma")


# ------------------------------------------------------------ validation ----
class TestConstruction:
    def test_hidden_cell_must_have_been_observed(self) -> None:
        cohort = make_cohort(np.zeros((4, 4)))
        hidden = np.zeros((4, 4), dtype=bool)
        hidden[0, 0] = True
        with pytest.raises(ConfigError, match="never observed"):
            engine(cohort, hidden)

    def test_negative_budget_rejected(self, full_cohort: Cohort) -> None:
        with pytest.raises(BudgetError, match="non-negative"):
            engine(full_cohort, np.zeros((4, 4), dtype=bool), budget=-1.0)

    def test_unsorted_epochs_rejected(self, full_cohort: Cohort) -> None:
        with pytest.raises(ConfigError, match="ascending"):
            DisclosureEngine(
                full_cohort,
                0,
                np.zeros((4, 4), dtype=bool),
                CATALOGUE,
                budget=1.0,
                epoch_hours=(4, 2),
            )

    def test_boundary_beyond_horizon_rejected(self, full_cohort: Cohort) -> None:
        with pytest.raises(ConfigError, match="exceeds horizon"):
            DisclosureEngine(
                full_cohort,
                0,
                np.zeros((4, 4), dtype=bool),
                CATALOGUE,
                budget=1.0,
                epoch_hours=(2, 99),
            )

    def test_engine_does_not_mutate_the_cohort(self, full_cohort: Cohort) -> None:
        before = full_cohort.m.copy()
        hidden = np.zeros((4, 4), dtype=bool)
        hidden[0, 0] = True
        eng = engine(full_cohort, hidden)
        eng.request("alpha")
        np.testing.assert_array_equal(full_cohort.m, before)


def test_policy_view_is_frozen() -> None:
    """A policy must not be able to edit its own view into a different state."""
    assert dataclasses.fields(PolicyView)  # sanity
    cohort = make_cohort(np.ones((4, 4)))
    view = engine(cohort, np.zeros((4, 4), dtype=bool)).view()
    with pytest.raises(dataclasses.FrozenInstanceError):
        view.spent = 999.0  # type: ignore[misc]
