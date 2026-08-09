"""Patient-level splitting with a hard-locked final holdout.

Two rules are enforced mechanically rather than by convention:

1. **set-c is locked.** It may only be materialised through
   :func:`final_holdout`, which requires an explicit unlock token. Every other
   API path refuses to return it.
2. **Splits are by patient.** A patient never appears in two folds, so no
   summary statistic can bridge the boundary.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Final

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import StratifiedKFold, train_test_split

from cliniverse.data.cohort import Cohort
from cliniverse.exceptions import LeakageError

IntArray = npt.NDArray[np.int64]

#: Sets usable for development. set-c is deliberately absent.
DEVELOPMENT_SETS: Final = ("a", "b")
LOCKED_SET: Final = "c"

#: Required by :func:`final_holdout`. Not a secret — a deliberate speed bump, so
#: that touching the quarantined holdout is always an explicit, greppable act.
UNLOCK_TOKEN: Final = "I_HAVE_FINISHED_ALL_MODEL_SELECTION"  # noqa: S105


@dataclasses.dataclass(frozen=True, slots=True)
class Split:
    """Train/validation indices into a cohort, plus provenance."""

    train: IntArray
    validation: IntArray
    fold: int
    n_folds: int
    seed: int

    def __post_init__(self) -> None:
        overlap = np.intersect1d(self.train, self.validation)
        if overlap.size:
            raise LeakageError(
                f"fold {self.fold}: {overlap.size} indices in both train and validation"
            )

    @property
    def fingerprint(self) -> str:
        """Stable hash of the split membership, for reproducibility assertions."""
        h = hashlib.sha256()
        h.update(np.sort(self.train).tobytes())
        h.update(b"|")
        h.update(np.sort(self.validation).tobytes())
        return h.hexdigest()[:16]


def _assert_no_locked_set(cohort: Cohort) -> None:
    if np.any(cohort.source_set == LOCKED_SET):
        raise LeakageError(
            f"cohort contains locked set-{LOCKED_SET} records; "
            "load development data with sets=('a', 'b') or use final_holdout()"
        )


def development_cohort(cohort: Cohort) -> Cohort:
    """Validate and return a cohort containing development records only.

    Refusing rather than silently dropping set-c prevents a command that loaded
    the holdout by mistake from appearing to be a normal development run.
    """
    _assert_no_locked_set(cohort)
    return cohort


def stratified_folds(
    cohort: Cohort,
    *,
    task: str = "mortality",
    n_folds: int = 5,
    seed: int = 20260809,
) -> list[Split]:
    """Stratified k-fold over patients, stratified on the binary ``task`` label.

    Raises:
        LeakageError: if the cohort still contains locked-set records.
    """
    _assert_no_locked_set(cohort)
    y = cohort.labels[task]
    finite = np.isfinite(y)
    if not bool(finite.all()):
        raise LeakageError(
            f"task {task!r} has {int((~finite).sum())} non-finite labels; "
            "filter them before splitting so folds are well defined"
        )

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    indices = np.arange(cohort.n_patients, dtype=np.int64)
    return [
        Split(
            train=indices[tr],
            validation=indices[va],
            fold=i,
            n_folds=n_folds,
            seed=seed,
        )
        for i, (tr, va) in enumerate(skf.split(indices, y.astype(int)))
    ]


def holdout_split(
    cohort: Cohort,
    *,
    task: str = "mortality",
    validation_fraction: float = 0.2,
    seed: int = 20260809,
) -> Split:
    """A single stratified train/validation split, for quick iteration."""
    _assert_no_locked_set(cohort)
    y = cohort.labels[task]
    indices = np.arange(cohort.n_patients, dtype=np.int64)
    train, validation = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=seed,
        stratify=y.astype(int),
    )
    return Split(
        train=np.asarray(train, dtype=np.int64),
        validation=np.asarray(validation, dtype=np.int64),
        fold=0,
        n_folds=1,
        seed=seed,
    )


def final_holdout(cohort: Cohort, unlock_token: str) -> Cohort:
    """Materialise the locked set-c holdout.

    Call this **once**, after all model selection is complete. Every use must be
    recorded in ``docs/EXPERIMENTS.md``.

    Raises:
        LeakageError: if the unlock token is wrong.
    """
    if unlock_token != UNLOCK_TOKEN:
        raise LeakageError(
            "final_holdout() requires the explicit unlock token; "
            "set-c must not be touched during development"
        )
    keep = cohort.source_set == LOCKED_SET
    if not bool(keep.any()):
        raise LeakageError(f"cohort contains no set-{LOCKED_SET} records to evaluate")
    return cohort.select(np.flatnonzero(keep).astype(np.int64))
