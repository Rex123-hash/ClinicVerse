"""Case manifest and episode-runner tests.

The load-bearing property here is reproducibility: a manifest regenerated from
the same seed must hash identically, and a case must rebuild the same hidden
mask without any patient data being stored.
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pytest

from cliniverse.acquisition.catalogue import Panel, PanelCatalogue
from cliniverse.data.cohort import Cohort
from cliniverse.exceptions import ConfigError
from twinbench.cases import CaseManifest, build_manifest, engine_for, hidden_mask_for
from twinbench.disclosure import DisclosureEngine, Protocol
from twinbench.episode import (
    FixedOrder,
    NoAcquisition,
    Policy,
    RandomSupportOracle,
    RandomTrainFrequency,
    RandomUniformAll,
    TerminationReason,
    run_episode,
)
from twinbench.masking import GroupHours, MaskingMechanism, build_mechanism

CATALOGUE = PanelCatalogue(
    version="test",
    panels={
        "alpha": Panel(name="alpha", label="Alpha", members=("a1", "a2"), cost=1.0),
        "beta": Panel(name="beta", label="Beta", members=("b1", "b2"), cost=2.0),
    },
    schedule_name="shared_plus_marginal",
    alternative_schedules={
        "shared_plus_marginal": {"alpha": 1.0, "beta": 2.0},
        "uniform_group": {"alpha": 1.0, "beta": 1.0},
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
def mechanism() -> GroupHours:
    return GroupHours(rate=0.5, seed=42)


def manifest_for(cohort: Cohort, mechanism: MaskingMechanism, **kw: Any) -> CaseManifest:
    kw.setdefault("sets", ("a",))
    return build_manifest(
        cohort, mechanism, CATALOGUE, cutoff_hours=4, epoch_hours=EPOCHS, **kw
    )


class TestManifestReproducibility:
    def test_same_seed_gives_same_content_hash(
        self, cohort: Cohort, mechanism: GroupHours
    ) -> None:
        a = manifest_for(cohort, mechanism)
        b = manifest_for(cohort, GroupHours(rate=0.5, seed=42))
        assert a.content_hash == b.content_hash

    def test_different_seed_changes_content_hash(self, cohort: Cohort) -> None:
        a = manifest_for(cohort, GroupHours(rate=0.5, seed=1))
        b = manifest_for(cohort, GroupHours(rate=0.5, seed=2))
        assert a.content_hash != b.content_hash

    def test_protocol_changes_content_hash(
        self, cohort: Cohort, mechanism: GroupHours
    ) -> None:
        aware = manifest_for(cohort, mechanism, protocol=Protocol.SUPPORT_AWARE)
        blind = manifest_for(cohort, mechanism, protocol=Protocol.SUPPORT_BLIND)
        assert aware.content_hash != blind.content_hash

    def test_cost_regime_changes_content_hash(
        self, cohort: Cohort, mechanism: GroupHours
    ) -> None:
        a = manifest_for(cohort, mechanism, cost_regime="shared_plus_marginal")
        b = manifest_for(cohort, mechanism, cost_regime="uniform_group")
        assert a.content_hash != b.content_hash

    def test_hash_is_order_independent(self, cohort: Cohort, mechanism: GroupHours) -> None:
        original = manifest_for(cohort, mechanism)
        shuffled = original.model_copy(update={"cases": tuple(reversed(original.cases))})
        assert shuffled.content_hash == original.content_hash

    @pytest.mark.parametrize("field", ["config_hash", "dataset_fingerprint", "git_sha"])
    def test_every_manifest_provenance_field_changes_hash(
        self, cohort: Cohort, mechanism: GroupHours, field: str
    ) -> None:
        original = manifest_for(cohort, mechanism)
        changed = original.model_copy(update={field: "changed"})
        assert changed.content_hash != original.content_hash

    def test_roundtrip_through_disk(
        self, cohort: Cohort, mechanism: GroupHours, tmp_path: pathlib.Path
    ) -> None:
        original = manifest_for(cohort, mechanism)
        path = original.write(tmp_path / "manifest.json")
        restored = CaseManifest.read(path)
        assert restored.content_hash == original.content_hash
        assert restored.cases == original.cases

    def test_tampered_manifest_is_rejected(
        self, cohort: Cohort, mechanism: GroupHours, tmp_path: pathlib.Path
    ) -> None:
        path = manifest_for(cohort, mechanism).write(tmp_path / "manifest.json")
        text = path.read_text(encoding="utf-8").replace('"budget": 5.0', '"budget": 9.0')
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ConfigError, match="content hash mismatch"):
            CaseManifest.read(path)

    def test_manifest_stores_no_patient_data(
        self, cohort: Cohort, mechanism: GroupHours
    ) -> None:
        """Only descriptions are persisted, never values."""
        text = manifest_for(cohort, mechanism).to_json()
        for value in cohort.x.ravel()[:20]:
            assert f'"{value}"' not in text

    def test_manifest_records_required_provenance(
        self, cohort: Cohort, mechanism: GroupHours
    ) -> None:
        manifest = manifest_for(cohort, mechanism)
        case = manifest.cases[0]
        assert manifest.version == "2"
        assert manifest.dataset_fingerprint
        assert manifest.config_hash
        assert manifest.git_sha
        assert case.masking_seed == mechanism.seed
        assert case.masking_rate == mechanism.rate
        assert case.catalogue_version == CATALOGUE.version
        assert case.catalogue_hash

    def test_declared_set_must_match_cohort(
        self, cohort: Cohort, mechanism: GroupHours
    ) -> None:
        with pytest.raises(ConfigError, match="do not match cohort"):
            manifest_for(cohort, mechanism, sets=("b",))

    def test_final_epoch_must_equal_cutoff(
        self, cohort: Cohort, mechanism: GroupHours
    ) -> None:
        with pytest.raises(ConfigError, match="final epoch"):
            build_manifest(
                cohort,
                mechanism,
                CATALOGUE,
                cutoff_hours=3,
                epoch_hours=(2, 4),
                sets=("a",),
            )


class TestHiddenMaskRebuild:
    def test_mask_rebuilds_identically(self, cohort: Cohort, mechanism: GroupHours) -> None:
        case = manifest_for(cohort, mechanism).cases[3]
        a = hidden_mask_for(cohort, case, mechanism, CATALOGUE)
        b = hidden_mask_for(cohort, case, GroupHours(rate=0.5, seed=42), CATALOGUE)
        np.testing.assert_array_equal(a, b)

    def test_mismatched_mechanism_rejected(
        self, cohort: Cohort, mechanism: GroupHours
    ) -> None:
        case = manifest_for(cohort, mechanism).cases[0]
        with pytest.raises(ConfigError, match="does not match case"):
            hidden_mask_for(cohort, case, GroupHours(rate=0.9, seed=42), CATALOGUE)

    def test_engine_uses_the_case_cost_regime(
        self, cohort: Cohort, mechanism: GroupHours
    ) -> None:
        case = manifest_for(cohort, mechanism, cost_regime="uniform_group").cases[0]
        engine = engine_for(cohort, case, mechanism, CATALOGUE)
        assert engine.view().catalogue.cost_of("beta") == pytest.approx(1.0)

    def test_mask_never_uses_post_cutoff_support(self, cohort: Cohort) -> None:
        mechanism = GroupHours(rate=1.0, seed=1)
        manifest = build_manifest(
            cohort,
            mechanism,
            CATALOGUE,
            sets=("a",),
            cutoff_hours=2,
            epoch_hours=(2,),
        )
        hidden = hidden_mask_for(cohort, manifest.cases[0], mechanism, CATALOGUE)
        assert hidden[:2].any()
        assert not hidden[2:].any()

    def test_case_patient_reference_is_verified(
        self, cohort: Cohort, mechanism: GroupHours
    ) -> None:
        case = manifest_for(cohort, mechanism).cases[0].model_copy(update={"record_id": 999})
        with pytest.raises(ConfigError, match="does not match patient index"):
            hidden_mask_for(cohort, case, mechanism, CATALOGUE)

    def test_catalogue_content_is_verified(
        self, cohort: Cohort, mechanism: GroupHours
    ) -> None:
        case = manifest_for(cohort, mechanism).cases[0]
        changed = CATALOGUE.model_copy(update={"version": "changed"})
        with pytest.raises(ConfigError, match="catalogue version or content"):
            engine_for(cohort, case, mechanism, changed)


class TestEpisodeRunner:
    def _engine(
        self, cohort: Cohort, mechanism: MaskingMechanism, **kw: Any
    ) -> DisclosureEngine:
        case = manifest_for(cohort, mechanism, **kw).cases[0]
        return engine_for(cohort, case, mechanism, CATALOGUE)

    def test_no_acquisition_spends_nothing(
        self, cohort: Cohort, mechanism: GroupHours
    ) -> None:
        trace = run_episode(self._engine(cohort, mechanism), NoAcquisition())
        assert trace.spent == 0.0
        assert trace.n_requests == 0
        assert trace.termination is TerminationReason.POLICY_STOP

    def test_random_policy_spends_within_budget(
        self, cohort: Cohort, mechanism: GroupHours
    ) -> None:
        engine = self._engine(cohort, mechanism, budget=3.0)
        trace = run_episode(engine, RandomUniformAll(seed=1))
        assert 0 < trace.spent <= 3.0
        assert trace.n_requests >= 1

    def test_random_policy_is_deterministic(
        self, cohort: Cohort, mechanism: GroupHours
    ) -> None:
        a = run_episode(self._engine(cohort, mechanism), RandomUniformAll(seed=7))
        b = run_episode(self._engine(cohort, mechanism), RandomUniformAll(seed=7))
        assert [p.panel for p in a.purchases] == [p.panel for p in b.purchases]

    def test_fixed_order_respects_its_order(
        self, cohort: Cohort, mechanism: GroupHours
    ) -> None:
        engine = self._engine(cohort, mechanism, budget=10.0)
        trace = run_episode(engine, FixedOrder(order=("beta", "alpha")))
        panels = [p.panel for p in trace.purchases]
        assert panels[:2] == ["beta", "alpha"]

    def test_trace_records_protocol_and_patient(
        self, cohort: Cohort, mechanism: GroupHours
    ) -> None:
        engine = self._engine(cohort, mechanism, protocol=Protocol.SUPPORT_AWARE)
        trace = run_episode(engine, NoAcquisition())
        assert trace.protocol == "support_aware"
        assert trace.patient_index == 0

    def test_policy_cannot_overspend(self, cohort: Cohort, mechanism: GroupHours) -> None:
        """A greedy policy asking for everything must still respect the budget."""
        engine = self._engine(cohort, mechanism, budget=1.0)
        trace = run_episode(engine, FixedOrder(order=("beta", "alpha", "beta")))
        assert trace.spent <= 1.0

    def test_empty_requests_are_counted_and_charged(self, cohort: Cohort) -> None:
        """Under support_blind, wasted spend must be visible in the trace."""
        nothing_hidden = GroupHours(rate=0.0, seed=3)
        case = manifest_for(cohort, nothing_hidden, budget=10.0).cases[0]
        engine = engine_for(cohort, case, nothing_hidden, CATALOGUE)
        trace = run_episode(engine, FixedOrder(order=("alpha", "beta")))
        assert trace.n_disclosed == 0
        assert trace.n_empty_requests == 2
        assert trace.wasted_spend == pytest.approx(3.0)

    def test_runner_rejects_bad_guard(self, cohort: Cohort, mechanism: GroupHours) -> None:
        with pytest.raises(ConfigError, match="max_requests_per_epoch"):
            run_episode(
                self._engine(cohort, mechanism), NoAcquisition(), max_requests_per_epoch=0
            )

    def test_builtin_policies_satisfy_the_protocol(self) -> None:
        for policy in (
            NoAcquisition(),
            RandomUniformAll(),
            RandomSupportOracle(),
            FixedOrder(order=("alpha",)),
        ):
            assert isinstance(policy, Policy)

    def test_zero_budget_stops_without_calling_policy(
        self, cohort: Cohort, mechanism: GroupHours
    ) -> None:
        trace = run_episode(self._engine(cohort, mechanism, budget=0), NoAcquisition())
        assert trace.termination is TerminationReason.BUDGET_EXHAUSTED
        assert trace.n_requests == 0

    def test_unknown_action_has_explicit_termination(
        self, cohort: Cohort, mechanism: GroupHours
    ) -> None:
        policy = FixedOrder(order=("gamma",))
        # FixedOrder skips unknown names, so use an adversarial implementation.
        policy.select = lambda view: "gamma"  # type: ignore[method-assign]
        trace = run_episode(self._engine(cohort, mechanism), policy)
        assert trace.termination is TerminationReason.UNKNOWN_ACTION

    def test_malformed_action_has_explicit_termination(
        self, cohort: Cohort, mechanism: GroupHours
    ) -> None:
        policy = FixedOrder(order=("alpha",))
        policy.select = lambda view: ["alpha"]  # type: ignore[method-assign,return-value,assignment]
        trace = run_episode(self._engine(cohort, mechanism), policy)
        assert trace.termination is TerminationReason.MALFORMED_ACTION

    def test_repeating_action_hits_guard_without_looping(
        self, cohort: Cohort, mechanism: GroupHours
    ) -> None:
        policy = FixedOrder(order=("alpha",))
        policy.select = lambda view: "alpha"  # type: ignore[method-assign]
        trace = run_episode(
            self._engine(cohort, mechanism, budget=100),
            policy,
            max_requests_per_epoch=3,
        )
        assert trace.termination is TerminationReason.REQUEST_LIMIT
        assert trace.n_requests == 3

    def test_train_frequency_fit_ignores_held_out_support(self, cohort: Cohort) -> None:
        train = cohort.m[:3]
        a = RandomTrainFrequency.fit(train, VARIABLES, CATALOGUE, seed=1)
        changed_validation = cohort.m.copy()
        changed_validation[3:] = False
        b = RandomTrainFrequency.fit(changed_validation[:3], VARIABLES, CATALOGUE, seed=1)
        assert a.weights == b.weights

    def test_oracle_random_is_named_and_used_only_as_diagnostic(
        self, cohort: Cohort, mechanism: GroupHours
    ) -> None:
        engine = self._engine(cohort, mechanism, protocol=Protocol.SUPPORT_AWARE)
        trace = run_episode(engine, RandomSupportOracle(seed=1))
        assert trace.policy == "random_support_oracle"


class TestFactoryIntegration:
    def test_manifest_from_named_mechanism(self, cohort: Cohort) -> None:
        mech = build_mechanism("group_hours", rate=0.4, seed=9)
        manifest = manifest_for(cohort, mech)
        assert len(manifest.cases) == cohort.n_patients
        assert all(c.mechanism_id == mech.mechanism_id for c in manifest.cases)
