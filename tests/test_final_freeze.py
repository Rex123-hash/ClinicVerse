"""M5-v2 final-freeze guards.

Two properties must hold for the frozen set-c test to mean anything, and both are
easy to break silently later:

**Train/calibration isolation.** If the calibrator ever sees its own training
rows, the frozen probabilities are self-calibrated and the one-shot holdout test
measures nothing. The partition must stay disjoint, correctly sized and
stratified.

**No set-c access during the freeze.** The freeze stage must be provably
incapable of materialising the holdout. These tests assert that at the source
level, so they fail even if the dataset is absent from the machine.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from cliniverse.data.physionet2012 import (
    DEVELOPMENT_RECORD_SETS,
    LOCKED_RECORD_SET,
    load_cohort,
)
from cliniverse.exceptions import DataError

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
FREEZE_SCRIPT = REPO_ROOT / "experiments" / "robustness" / "m5_v2_final_freeze.py"
FREEZE_RESULT = (
    REPO_ROOT
    / "experiments"
    / "robustness"
    / "results"
    / "m5v2_final_freeze"
    / "final_freeze.json"
)
FREEZE_PACKAGE = FREEZE_RESULT.parent
FREEZE_REPORT = REPO_ROOT / "docs" / "M5_V2_FINAL_FREEZE.md"

EXPECTED_CONTROL_POOL = [
    "ALP",
    "ALT",
    "AST",
    "Albumin",
    "BUN",
    "Bilirubin",
    "Cholesterol",
    "Creatinine",
    "Glucose",
    "HCO3",
    "HCT",
    "K",
    "Lactate",
    "Mg",
    "Na",
    "PaCO2",
    "PaO2",
    "Platelets",
    "SaO2",
    "TroponinI",
    "TroponinT",
    "WBC",
    "pH",
]

FREEZE_SEED = 20260809
N_FINAL_TRAIN = 6400
N_FINAL_CALIBRATION = 1600


class TestSetCIsUnreachableFromTheFreeze:
    def test_freeze_script_never_requests_the_holdout(self) -> None:
        """Source-level guard: the flag that materialises set-c is never passed.

        Asserted on the text rather than by running the script, so this fails on a
        machine with no dataset at all, and fails loudly if someone adds the flag
        later to "just check something".
        """
        source = FREEZE_SCRIPT.read_text(encoding="utf-8")
        offending = [
            line.strip()
            for line in source.splitlines()
            if "allow_final_holdout=True" in line.replace(" ", "") or "final_holdout(" in line
        ]
        assert not offending, f"freeze stage can reach set-c: {offending}"

    def test_freeze_script_does_not_name_the_locked_set_as_data(self) -> None:
        source = FREEZE_SCRIPT.read_text(encoding="utf-8")
        assert 'sets=("a", "b", "c")' not in source
        assert "RECORD_SETS" not in source

    def test_development_default_excludes_the_locked_set(self) -> None:
        assert LOCKED_RECORD_SET not in DEVELOPMENT_RECORD_SETS
        assert set(DEVELOPMENT_RECORD_SETS) == {"a", "b"}

    def test_loading_the_locked_set_without_the_flag_raises(self) -> None:
        """The lock is enforced by the loader, not merely by convention."""
        with pytest.raises(DataError, match="quarantined"):
            load_cohort(sets=("a", LOCKED_RECORD_SET), download=False)


class TestFinalPartitionContract:
    """The 6,400 / 1,600 split, reproduced independently of the freeze script."""

    @pytest.fixture
    def partition(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        y = (rng.random(N_FINAL_TRAIN + N_FINAL_CALIBRATION) < 0.14025).astype(np.float64)
        train_idx, calib_idx = train_test_split(
            np.arange(y.size),
            test_size=N_FINAL_CALIBRATION,
            random_state=FREEZE_SEED,
            stratify=y,
        )
        return np.sort(train_idx), np.sort(calib_idx), y

    def test_sizes_are_exactly_as_predeclared(
        self, partition: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        train_idx, calib_idx, _ = partition
        assert len(train_idx) == N_FINAL_TRAIN
        assert len(calib_idx) == N_FINAL_CALIBRATION

    def test_partitions_are_disjoint(
        self, partition: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        train_idx, calib_idx, _ = partition
        assert np.intersect1d(train_idx, calib_idx).size == 0

    def test_partitions_cover_every_patient_exactly_once(
        self, partition: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        train_idx, calib_idx, y = partition
        combined = np.concatenate([train_idx, calib_idx])
        assert np.array_equal(np.sort(combined), np.arange(y.size))

    def test_split_is_stratified(
        self, partition: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        train_idx, calib_idx, y = partition
        assert y[train_idx].mean() == pytest.approx(y[calib_idx].mean(), abs=0.005)

    def test_split_is_reproducible_from_the_seed(
        self, partition: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        train_idx, calib_idx, y = partition
        again_train, again_calib = train_test_split(
            np.arange(y.size),
            test_size=N_FINAL_CALIBRATION,
            random_state=FREEZE_SEED,
            stratify=y,
        )
        assert np.array_equal(train_idx, np.sort(again_train))
        assert np.array_equal(calib_idx, np.sort(again_calib))

    def test_a_different_seed_gives_a_different_partition(
        self, partition: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        train_idx, _, y = partition
        other, _ = train_test_split(
            np.arange(y.size),
            test_size=N_FINAL_CALIBRATION,
            random_state=FREEZE_SEED + 1,
            stratify=y,
        )
        assert not np.array_equal(train_idx, np.sort(other))


@pytest.mark.slow
class TestFrozenArtifact:
    """Checks on the committed freeze artifact, skipped when it is absent."""

    @pytest.fixture
    def freeze(self) -> dict[str, object]:
        if not FREEZE_RESULT.is_file():
            pytest.skip("final freeze artifact not generated")
        return json.loads(FREEZE_RESULT.read_text(encoding="utf-8"))

    def test_pattern_is_the_frozen_one(self, freeze: dict[str, object]) -> None:
        assert freeze["frozen_pattern"] == ["BUN", "Glucose", "Na"]

    def test_only_development_sets_were_used(self, freeze: dict[str, object]) -> None:
        access = freeze["set_c_access"]
        assert isinstance(access, dict)
        assert access["loaded_during_freeze"] is False
        assert access["scored_during_freeze"] is False
        assert access["allow_final_holdout_passed"] is False
        assert access["realised_source_sets"] == ["a", "b"]
        provenance = freeze["provenance"]
        assert isinstance(provenance, dict)
        assert provenance["sets"] == ["a", "b"]

    def test_partition_recorded_as_disjoint_and_sized(self, freeze: dict[str, object]) -> None:
        split = freeze["split"]
        assert isinstance(split, dict)
        assert split["n_final_train"] == N_FINAL_TRAIN
        assert split["n_final_calibration"] == N_FINAL_CALIBRATION
        assert split["disjoint"] is True
        assert split["seed"] == FREEZE_SEED

    def test_calibrator_isolation_is_recorded(self, freeze: dict[str, object]) -> None:
        calibrator = freeze["calibrator"]
        assert isinstance(calibrator, dict)
        assert calibrator["fitted_on"] == "the 1,600 clean final-calibration rows only"
        assert calibrator["never_refitted_under_withholding"] is True

    def test_model_is_not_a_constant(self, freeze: dict[str, object]) -> None:
        model = freeze["model"]
        assert isinstance(model, dict)
        assert int(model["n_features_used"]) > 0  # type: ignore[call-overload]

    def test_set_c_contract_is_frozen_and_unexecuted(self, freeze: dict[str, object]) -> None:
        contract = freeze["set_c_evaluation_contract"]
        assert isinstance(contract, dict)
        assert contract["n_bootstrap"] == 10000
        assert contract["bootstrap_seed"] == FREEZE_SEED
        assert contract["control_repeats"] == 5
        assert contract["primary_decision_rule"] == "PASS if and only if LB > 0"
        assert "NOT EXECUTED" in str(contract["status"])
        assert len(contract["frozen_control_seeds"]) == 5  # type: ignore[arg-type]
        assert "must be restated verbatim" in str(contract["monte_carlo_limitation"])

    def test_artifact_hashes_cover_the_package(self, freeze: dict[str, object]) -> None:
        hashes = freeze["artifact_hashes"]
        assert isinstance(hashes, dict)
        assert set(hashes) == {
            "final_model.json",
            "final_imputer.npz",
            "final_calibrator.json",
        }
        for digest in hashes.values():
            assert isinstance(digest, str)
            assert len(digest) == 64

    def test_artifact_hashes_match_package_bytes(self, freeze: dict[str, object]) -> None:
        hashes = freeze["artifact_hashes"]
        assert isinstance(hashes, dict)
        for name, expected in hashes.items():
            path = FREEZE_PACKAGE / str(name)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected

    def test_control_mechanism_and_pool_are_fully_frozen(
        self, freeze: dict[str, object]
    ) -> None:
        contract = freeze["set_c_evaluation_contract"]
        assert isinstance(contract, dict)
        assert contract["control_condition"] == "LossCondition.CELL_RANDOM"
        assert "exact realised per-patient" in str(contract["amount_matching"])
        assert contract["eligible_control_pool_n"] == 23
        assert contract["eligible_control_pool"] == EXPECTED_CONTROL_POOL
        assert contract["eligible_control_pool_includes_withheld_analytes"] is True
        assert "no refitting or substitution" in str(contract["fitted_objects"])

    def test_transparent_imputer_and_calibrator_state_is_complete(self) -> None:
        with np.load(FREEZE_PACKAGE / "final_imputer.npz", allow_pickle=False) as imputer:
            assert set(imputer.files) == {"medians", "jitter_scales", "seed", "strategy"}
            assert imputer["medians"].shape == (298,)
            assert imputer["jitter_scales"].shape == (298,)
            assert int(imputer["seed"]) == FREEZE_SEED
            assert str(imputer["strategy"]) == "median"
        calibrator = json.loads(
            (FREEZE_PACKAGE / "final_calibrator.json").read_text(encoding="utf-8")
        )
        assert calibrator["kind"] == "platt"
        assert calibrator["fitted"] is True
        assert calibrator["n_calibration"] == N_FINAL_CALIBRATION
        assert np.isfinite([calibrator["slope"], calibrator["intercept"]]).all()


def test_historical_exposure_wording_does_not_overclaim() -> None:
    report = FREEZE_REPORT.read_text(encoding="utf-8")
    assert "No set-c per-patient data" not in report
    assert "No Set-C patient-level information was retained or used" in report
