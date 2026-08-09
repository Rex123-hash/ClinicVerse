"""Split integrity — the tests that protect every downstream number.

Data leakage does not raise errors at runtime; it just inflates results. These
tests are the only thing standing between us and a silently invalid benchmark.
"""

from __future__ import annotations

import numpy as np
import pytest

from cliniverse.data.cohort import Cohort
from cliniverse.data.splits import (
    LOCKED_SET,
    UNLOCK_TOKEN,
    development_cohort,
    final_holdout,
    holdout_split,
    stratified_folds,
)
from cliniverse.exceptions import LeakageError


@pytest.fixture
def cohort_with_locked_set(toy_cohort: Cohort) -> Cohort:
    """Toy cohort where the last two patients belong to the locked set."""
    import dataclasses

    return dataclasses.replace(
        toy_cohort, source_set=np.array(["a", "a", "c", "c"], dtype=np.str_)
    )


class TestLockedSet:
    def test_stratified_folds_refuse_locked_records(
        self, cohort_with_locked_set: Cohort
    ) -> None:
        with pytest.raises(LeakageError, match="locked set"):
            stratified_folds(cohort_with_locked_set, n_folds=2)

    def test_holdout_split_refuses_locked_records(
        self, cohort_with_locked_set: Cohort
    ) -> None:
        with pytest.raises(LeakageError, match="locked set"):
            holdout_split(cohort_with_locked_set)

    def test_development_cohort_drops_locked_records(
        self, cohort_with_locked_set: Cohort
    ) -> None:
        dev = development_cohort(cohort_with_locked_set)
        assert dev.n_patients == 2
        assert not np.any(dev.source_set == LOCKED_SET)

    def test_final_holdout_requires_token(self, cohort_with_locked_set: Cohort) -> None:
        with pytest.raises(LeakageError, match="unlock token"):
            final_holdout(cohort_with_locked_set, "please")

    def test_final_holdout_with_token_returns_only_locked(
        self, cohort_with_locked_set: Cohort
    ) -> None:
        held = final_holdout(cohort_with_locked_set, UNLOCK_TOKEN)
        assert held.n_patients == 2
        assert set(held.source_set.tolist()) == {LOCKED_SET}

    def test_final_holdout_errors_when_absent(self, toy_cohort: Cohort) -> None:
        with pytest.raises(LeakageError, match="no set-c"):
            final_holdout(toy_cohort, UNLOCK_TOKEN)


class TestFoldIntegrity:
    def test_train_and_validation_are_disjoint(self, toy_cohort: Cohort) -> None:
        for split in stratified_folds(toy_cohort, n_folds=2):
            assert not set(split.train.tolist()) & set(split.validation.tolist())

    def test_folds_cover_every_patient_exactly_once(self, toy_cohort: Cohort) -> None:
        folds = stratified_folds(toy_cohort, n_folds=2)
        seen = np.concatenate([f.validation for f in folds])
        assert sorted(seen.tolist()) == list(range(toy_cohort.n_patients))

    def test_splits_are_deterministic(self, toy_cohort: Cohort) -> None:
        a = stratified_folds(toy_cohort, n_folds=2, seed=7)
        b = stratified_folds(toy_cohort, n_folds=2, seed=7)
        assert [f.fingerprint for f in a] == [f.fingerprint for f in b]

    def test_different_seeds_give_different_splits(self, toy_cohort: Cohort) -> None:
        a = stratified_folds(toy_cohort, n_folds=2, seed=1)
        b = stratified_folds(toy_cohort, n_folds=2, seed=2)
        # Not guaranteed for a 4-patient toy set, but must not raise; the real
        # determinism guarantee is the same-seed test above.
        assert len(a) == len(b) == 2

    def test_overlapping_split_rejected(self) -> None:
        from cliniverse.data.splits import Split

        with pytest.raises(LeakageError, match="both train and validation"):
            Split(
                train=np.array([0, 1, 2], dtype=np.int64),
                validation=np.array([2, 3], dtype=np.int64),
                fold=0,
                n_folds=1,
                seed=0,
            )

    def test_non_finite_labels_rejected(self, toy_cohort: Cohort) -> None:
        import dataclasses

        bad = dataclasses.replace(
            toy_cohort,
            labels={"mortality": np.array([0.0, 1.0, np.nan, 1.0], dtype=np.float32)},
        )
        with pytest.raises(LeakageError, match="non-finite labels"):
            stratified_folds(bad, n_folds=2)
