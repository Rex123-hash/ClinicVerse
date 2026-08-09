"""Case manifest and episode-runner tests.

The load-bearing property here is reproducibility: a manifest regenerated from
the same seed must hash identically, and a case must rebuild the same hidden
mask without any patient data being stored.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from cliniverse.acquisition.catalogue import Panel, PanelCatalogue
from cliniverse.data.cohort import Cohort
from cliniverse.exceptions import ConfigError
from twinbench.cases import CaseManifest, build_manifest, engine_for, hidden_mask_for
from twinbench.disclosure import Protocol
from twinbench.episode import FixedOrder, NoAcquisition, Policy, RandomPolicy, run_episode
from twinbench.masking import PanelEvents, build_mechanism

CATALOGUE = PanelCatalogue(
    version="test",
    panels={
        "alpha": Panel(name="alpha", label="Alpha", members=("a1", "a2"), cost=1.0),
        "beta": Panel(name="beta", label="Beta", members=("b1", "b2"), cost=2.0),
    },
    schedule_name="shared_plus_marginal",
    alternative_schedules={
        "shared_plus_marginal": {"alpha": 1.0, "beta": 2.0},
        "uniform_event": {"alpha": 1.0, "beta": 1.0},
    },
)
VARIABLES = ("a1", "a2", "b1", "b2")
EPOCHS = (2, 4)


@pytest.fixture
def cohort() -> Cohort:
    n, t, v = 6, 4, 4
    m = np.ones((n, t, v), dtype=bool)
    x = np.arange(n * t * v, dtype=np.float32).reshape(n, t, v)
    return Cohort(
        record_ids=np.arange(100, 100 + n, dtype=np.int64),
        source_set=np.array(["a"] * n, dtype=np.str_),
        x=x,
        m=m,
        statics=np.zeros((n, 1), dtype=np.float32),
        statics_mask=np.ones((n, 1), dtype=bool),
        labels={"mortality": np.array([0, 1] * (n // 2), dtype=np.float32)},
        variable_names=VARIABLES,
        static_names=("Age",),
    )


@pytest.fixture
def mechanism() -> PanelEvents:
    return PanelEvents(rate=0.5, seed=42)


def manifest_for(cohort: Cohort, mechanism: PanelEvents, **kw) -> CaseManifest:
    return build_manifest(cohort, mechanism, epoch_hours=EPOCHS, **kw)


class TestManifestReproducibility:
    def test_same_seed_gives_same_content_hash(
        self, cohort: Cohort, mechanism: PanelEvents
    ) -> None:
        a = manifest_for(cohort, mechanism)
        b = manifest_for(cohort, PanelEvents(rate=0.5, seed=42))
        assert a.content_hash == b.content_hash

    def test_different_seed_changes_content_hash(self, cohort: Cohort) -> None:
        a = manifest_for(cohort, PanelEvents(rate=0.5, seed=1))
        b = manifest_for(cohort, PanelEvents(rate=0.5, seed=2))
        assert a.content_hash != b.content_hash

    def test_protocol_changes_content_hash(
        self, cohort: Cohort, mechanism: PanelEvents
    ) -> None:
        aware = manifest_for(cohort, mechanism, protocol=Protocol.SUPPORT_AWARE)
        blind = manifest_for(cohort, mechanism, protocol=Protocol.SUPPORT_BLIND)
        assert aware.content_hash != blind.content_hash

    def test_cost_regime_changes_content_hash(
        self, cohort: Cohort, mechanism: PanelEvents
    ) -> None:
        a = manifest_for(cohort, mechanism, cost_regime="shared_plus_marginal")
        b = manifest_for(cohort, mechanism, cost_regime="uniform_event")
        assert a.content_hash != b.content_hash

    def test_hash_is_order_independent(self, cohort: Cohort, mechanism: PanelEvents) -> None:
        original = manifest_for(cohort, mechanism)
        shuffled = original.model_copy(update={"cases": tuple(reversed(original.cases))})
        assert shuffled.content_hash == original.content_hash

    def test_roundtrip_through_disk(
        self, cohort: Cohort, mechanism: PanelEvents, tmp_path: pathlib.Path
    ) -> None:
        original = manifest_for(cohort, mechanism)
        path = original.write(tmp_path / "manifest.json")
        restored = CaseManifest.read(path)
        assert restored.content_hash == original.content_hash
        assert restored.cases == original.cases

    def test_tampered_manifest_is_rejected(
        self, cohort: Cohort, mechanism: PanelEvents, tmp_path: pathlib.Path
    ) -> None:
        path = manifest_for(cohort, mechanism).write(tmp_path / "manifest.json")
        text = path.read_text(encoding="utf-8").replace('"budget": 5.0', '"budget": 9.0')
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ConfigError, match="content hash mismatch"):
            CaseManifest.read(path)

    def test_manifest_stores_no_patient_data(
        self, cohort: Cohort, mechanism: PanelEvents
    ) -> None:
        """Only descriptions are persisted, never values."""
        text = manifest_for(cohort, mechanism).to_json()
        for value in cohort.x.ravel()[:20]:
            assert f'"{value}"' not in text


class TestHiddenMaskRebuild:
    def test_mask_rebuilds_identically(self, cohort: Cohort, mechanism: PanelEvents) -> None:
        case = manifest_for(cohort, mechanism).cases[3]
        a = hidden_mask_for(cohort, case, mechanism, CATALOGUE)
        b = hidden_mask_for(cohort, case, PanelEvents(rate=0.5, seed=42), CATALOGUE)
        np.testing.assert_array_equal(a, b)

    def test_mismatched_mechanism_rejected(
        self, cohort: Cohort, mechanism: PanelEvents
    ) -> None:
        case = manifest_for(cohort, mechanism).cases[0]
        with pytest.raises(ConfigError, match="does not match case"):
            hidden_mask_for(cohort, case, PanelEvents(rate=0.9, seed=42), CATALOGUE)

    def test_engine_uses_the_case_cost_regime(
        self, cohort: Cohort, mechanism: PanelEvents
    ) -> None:
        case = manifest_for(cohort, mechanism, cost_regime="uniform_event").cases[0]
        engine = engine_for(cohort, case, mechanism, CATALOGUE)
        assert engine.view().catalogue.cost_of("beta") == pytest.approx(1.0)


class TestEpisodeRunner:
    def _engine(self, cohort: Cohort, mechanism: PanelEvents, **kw):
        case = manifest_for(cohort, mechanism, **kw).cases[0]
        return engine_for(cohort, case, mechanism, CATALOGUE)

    def test_no_acquisition_spends_nothing(
        self, cohort: Cohort, mechanism: PanelEvents
    ) -> None:
        trace = run_episode(self._engine(cohort, mechanism), NoAcquisition())
        assert trace.spent == 0.0
        assert trace.n_requests == 0

    def test_random_policy_spends_within_budget(
        self, cohort: Cohort, mechanism: PanelEvents
    ) -> None:
        engine = self._engine(cohort, mechanism, budget=3.0)
        trace = run_episode(engine, RandomPolicy(seed=1))
        assert 0 < trace.spent <= 3.0
        assert trace.n_requests >= 1

    def test_random_policy_is_deterministic(
        self, cohort: Cohort, mechanism: PanelEvents
    ) -> None:
        a = run_episode(self._engine(cohort, mechanism), RandomPolicy(seed=7))
        b = run_episode(self._engine(cohort, mechanism), RandomPolicy(seed=7))
        assert [p.panel for p in a.purchases] == [p.panel for p in b.purchases]

    def test_fixed_order_respects_its_order(
        self, cohort: Cohort, mechanism: PanelEvents
    ) -> None:
        engine = self._engine(cohort, mechanism, budget=10.0)
        trace = run_episode(engine, FixedOrder(order=("beta", "alpha")))
        panels = [p.panel for p in trace.purchases]
        assert panels[:2] == ["beta", "alpha"]

    def test_trace_records_protocol_and_patient(
        self, cohort: Cohort, mechanism: PanelEvents
    ) -> None:
        engine = self._engine(cohort, mechanism, protocol=Protocol.SUPPORT_AWARE)
        trace = run_episode(engine, NoAcquisition())
        assert trace.protocol == "support_aware"
        assert trace.patient_index == 0

    def test_policy_cannot_overspend(self, cohort: Cohort, mechanism: PanelEvents) -> None:
        """A greedy policy asking for everything must still respect the budget."""
        engine = self._engine(cohort, mechanism, budget=1.0)
        trace = run_episode(engine, FixedOrder(order=("beta", "alpha", "beta")))
        assert trace.spent <= 1.0

    def test_empty_requests_are_counted_and_charged(self, cohort: Cohort) -> None:
        """Under support_blind, wasted spend must be visible in the trace."""
        nothing_hidden = PanelEvents(rate=0.0, seed=3)
        case = manifest_for(cohort, nothing_hidden, budget=10.0).cases[0]
        engine = engine_for(cohort, case, nothing_hidden, CATALOGUE)
        trace = run_episode(engine, FixedOrder(order=("alpha", "beta")))
        assert trace.n_disclosed == 0
        assert trace.n_empty_requests == 2
        assert trace.wasted_spend == pytest.approx(3.0)

    def test_runner_rejects_bad_guard(self, cohort: Cohort, mechanism: PanelEvents) -> None:
        with pytest.raises(ConfigError, match="max_requests_per_epoch"):
            run_episode(
                self._engine(cohort, mechanism), NoAcquisition(), max_requests_per_epoch=0
            )

    def test_builtin_policies_satisfy_the_protocol(self) -> None:
        for policy in (NoAcquisition(), RandomPolicy(), FixedOrder(order=("alpha",))):
            assert isinstance(policy, Policy)


class TestFactoryIntegration:
    def test_manifest_from_named_mechanism(self, cohort: Cohort) -> None:
        mech = build_mechanism("panel_events", rate=0.4, seed=9)
        manifest = manifest_for(cohort, mech)
        assert len(manifest.cases) == cohort.n_patients
        assert all(c.mechanism_id == mech.mechanism_id for c in manifest.cases)
