"""M5-v2 FINAL MODEL FREEZE — the contract in `docs/M5_V2_DESIGN.md` section 8.

This script fits ONE final pipeline on A+B and writes an auditable frozen package
plus the frozen set-c evaluation contract. It does not evaluate anything.

**It never loads, materialises or scores set-c.** `load_cohort()` is called with
its default development sets and the `allow_final_holdout` flag is never passed,
so a set-c read would raise rather than succeed silently. The realised source sets
are asserted to be exactly {a, b} and recorded in the artifact.

Training on all of A+B without isolation is not permitted: the calibrator must
never see its own training data. The 6,400 / 1,600 partition below is what keeps
that true, and `tests/test_final_freeze.py` pins it.

Usage:
    python experiments/robustness/m5_v2_final_freeze.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from cliniverse.acquisition import load_panel_catalogue
from cliniverse.data import load_cohort
from cliniverse.data.splits import development_cohort
from cliniverse.evaluation.artifacts import build_provenance, stable_hash
from cliniverse.evaluation.calibration import CalibratorKind, PlattCalibrator, build_calibrator
from cliniverse.evaluation.failure_search import analyte_columns, control_seed
from cliniverse.evaluation.representations import (
    FittedImputer,
    ImputationStrategy,
    Representation,
    build_representation,
)
from cliniverse.exceptions import ConfigError
from cliniverse.log import get_logger

log = get_logger(__name__)

# --------------------------------------------- frozen, unchanged from M2-M5 --
XGB_PARAMS: dict[str, Any] = {
    "max_depth": 5,
    "learning_rate": 0.05,
    "min_child_weight": 10,
    "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "eval_metric": "logloss",
    "n_jobs": 4,
}
REPRESENTATION = Representation.VALUES_MASK
CUTOFF = 24

# ------------------------------------- predeclared in M5_V2_DESIGN.md sec 8 --
FREEZE_SEED = 20260809
N_FINAL_TRAIN = 6400
N_FINAL_CALIBRATION = 1600
#: The M5-v2 frozen failure pattern. Not re-derived here; carried in as a constant
#: so this stage cannot silently select a different one.
FROZEN_PATTERN: tuple[str, ...] = ("BUN", "Glucose", "Na")

# ------------------------------------- predeclared in M5_V2_DESIGN.md sec 9 --
SET_C_CONTRACT: dict[str, Any] = {
    "status": "FROZEN — NOT EXECUTED. Requires separate explicit approval to run.",
    "pattern": list(FROZEN_PATTERN),
    "n_expected": 4000,
    "control_repeats": 5,
    "control_condition": "LossCondition.CELL_RANDOM",
    "amount_matching": (
        "For every patient and every control draw, match_counts is the frozen pattern's "
        "exact realised per-patient removed-cell count; abort on any mismatch."
    ),
    "control_seed_semantics": (
        "control_seed(20260809, ('BUN', 'Glucose', 'Na'), repetition) for "
        "repetition=0..4; no label or outcome input"
    ),
    "statistic": (
        "d_i = per-patient log loss under the withheld pattern minus the mean "
        "per-patient log loss over the R=5 amount-matched random controls; "
        "Delta_C = mean_i d_i"
    ),
    "interval": "paired patient-level percentile bootstrap on {d_i}",
    "n_bootstrap": 10000,
    "bootstrap_seed": FREEZE_SEED,
    "bound": "one-sided 95% LOWER confidence bound = 5th percentile of replicate means",
    "single_class_resamples": "skipped, matching project convention",
    "primary_decision_rule": "PASS if and only if LB > 0",
    "constraint": "AUROC(clean) - AUROC(pattern) <= 0.02 on set-c",
    "direction_fixed_by": "M5-v1 development finding; the hypothesis is strictly Delta_C > 0",
    "forbidden": [
        "any search or enumeration",
        "any tuning of model, calibrator, imputer or pattern",
        "any alternative or substitute pattern",
        "any second look, retest, or re-test under a different delta",
        "any refitting of the calibrator under withholding",
    ],
    "monte_carlo_limitation": (
        "The R=5 control draws are FIXED across all 10,000 bootstrap replicates. The "
        "interval propagates patient-sampling uncertainty but NOT control-draw "
        "Monte-Carlo uncertainty, and is mildly optimistic on that account. This "
        "sentence must be restated verbatim in any report of the set-c result."
    ),
}


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=FREEZE_SEED)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("experiments/robustness/results/m5v2_final_freeze"),
    )
    args = parser.parse_args(argv)

    catalogue = load_panel_catalogue()
    control_pool = sorted(catalogue.covered_variables)
    if len(control_pool) != 23:
        raise ConfigError(
            "final contract requires the inherited 23-lab control pool, "
            f"got {len(control_pool)}"
        )
    if not set(FROZEN_PATTERN).issubset(control_pool):
        raise ConfigError("frozen pattern is not contained in the eligible control pool")

    # `load_cohort()` defaults to the development sets. `allow_final_holdout` is
    # never passed anywhere in this file, so set-c cannot be materialised here.
    cohort = development_cohort(load_cohort())
    realised_sets = tuple(sorted(set(cohort.source_set.tolist())))
    if realised_sets != ("a", "b"):
        raise ConfigError(f"final freeze must use sets a+b only, got {realised_sets}")
    if cohort.n_patients != N_FINAL_TRAIN + N_FINAL_CALIBRATION:
        raise ConfigError(
            f"expected {N_FINAL_TRAIN + N_FINAL_CALIBRATION} development patients, "
            f"got {cohort.n_patients}"
        )

    y = cohort.labels["mortality"].astype(np.float64)
    truncated = cohort.truncate(CUTOFF)
    clean = build_representation(truncated, REPRESENTATION)
    log.info("cohort", n=cohort.n_patients, prevalence=float(y.mean()), sets=realised_sets)

    # ------------------------------------------------- the one partition ----
    train_idx, calib_idx = train_test_split(
        np.arange(cohort.n_patients),
        test_size=N_FINAL_CALIBRATION,
        random_state=args.seed,
        stratify=y,
    )
    train_idx = np.sort(train_idx)
    calib_idx = np.sort(calib_idx)
    if len(train_idx) != N_FINAL_TRAIN or len(calib_idx) != N_FINAL_CALIBRATION:
        raise ConfigError(
            f"partition is {len(train_idx)}/{len(calib_idx)}, expected "
            f"{N_FINAL_TRAIN}/{N_FINAL_CALIBRATION}"
        )
    if np.intersect1d(train_idx, calib_idx).size:
        raise ConfigError("final training and calibration partitions overlap")

    # ------------------------------------------------- fit, clean data only -
    imputer = FittedImputer.fit(
        clean.x[train_idx], strategy=ImputationStrategy.MEDIAN, seed=args.seed
    )
    model = XGBClassifier(random_state=args.seed, **XGB_PARAMS)
    model.fit(imputer.transform(clean.x[train_idx], draw_seed=args.seed), y[train_idx])

    trees = model.get_booster().trees_to_dataframe()
    n_features_used = len(set(trees["Feature"]) - {"Leaf"})
    if n_features_used == 0:
        raise ConfigError(
            "the final model splits on zero features, so it is a constant and no "
            "withheld information could change its prediction"
        )

    raw_calibration = model.predict_proba(
        imputer.transform(clean.x[calib_idx], draw_seed=100 + args.seed)
    )[:, 1]
    calibrator = build_calibrator(CalibratorKind.PLATT)
    calibrator.fit(np.asarray(raw_calibration, dtype=np.float64), y[calib_idx])
    if not isinstance(calibrator, PlattCalibrator):
        raise ConfigError(f"expected a Platt calibrator, got {type(calibrator).__name__}")
    log.info(
        "final model fitted",
        n_train=len(train_idx),
        n_calibration=len(calib_idx),
        n_features_used=n_features_used,
        slope=round(calibrator.slope, 6),
        intercept=round(calibrator.intercept, 6),
    )

    # ------------------------------------------------- write the package ----
    args.out.mkdir(parents=True, exist_ok=True)
    model_path = args.out / "final_model.json"
    imputer_path = args.out / "final_imputer.npz"
    calibrator_path = args.out / "final_calibrator.json"

    model.get_booster().save_model(str(model_path))
    np.savez_compressed(
        imputer_path,
        medians=imputer.medians,
        jitter_scales=imputer.jitter_scales,
        seed=np.asarray(imputer.seed),
        strategy=np.asarray(str(imputer.strategy)),
    )
    calibrator_path.write_text(
        json.dumps(
            {
                "kind": str(calibrator.kind),
                "slope": calibrator.slope,
                "intercept": calibrator.intercept,
                "fitted": calibrator.fitted,
                "n_calibration": len(calib_idx),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
        newline="\n",
    )

    pattern_columns = analyte_columns(truncated, FROZEN_PATTERN, catalogue).tolist()
    frozen_control_seeds = [
        control_seed(args.seed, FROZEN_PATTERN, r)
        for r in range(int(SET_C_CONTRACT["control_repeats"]))
    ]

    freeze: dict[str, Any] = {
        "schema": "cliniverse.m5v2.final_freeze/1",
        "design_document": "docs/M5_V2_DESIGN.md#8",
        "stage": "FINAL MODEL FREEZE — no evaluation performed",
        "frozen_pattern": list(FROZEN_PATTERN),
        "frozen_pattern_columns": pattern_columns,
        "frozen_pattern_provenance": (
            "M5-v2 A+B development, verdict v2-STABLE, selected in 11/20 resplits "
            "(exactly the majority threshold). Carried in as a constant; this stage "
            "performs no selection."
        ),
        "split": {
            "source_sets": list(realised_sets),
            "n_total": int(cohort.n_patients),
            "n_final_train": len(train_idx),
            "n_final_calibration": len(calib_idx),
            "seed": args.seed,
            "stratified_by": "mortality",
            "disjoint": True,
            "train_prevalence": float(y[train_idx].mean()),
            "calibration_prevalence": float(y[calib_idx].mean()),
            "train_index_hash": stable_hash(train_idx.tolist()),
            "calibration_index_hash": stable_hash(calib_idx.tolist()),
            "train_record_ids_hash": stable_hash(cohort.record_ids[train_idx].tolist()),
            "calibration_record_ids_hash": stable_hash(cohort.record_ids[calib_idx].tolist()),
        },
        "preprocessing": {
            "representation": str(REPRESENTATION),
            "cutoff_hours": CUTOFF,
            "n_features": int(clean.x.shape[1]),
            "imputation_strategy": str(ImputationStrategy.MEDIAN),
            "imputer_fitted_on": "the 6,400 clean final-training rows only",
            "imputer_seed": args.seed,
        },
        "model": {
            "estimator": "XGBClassifier",
            "hyperparameters": XGB_PARAMS,
            "random_state": args.seed,
            "fitted_on": "the 6,400 clean final-training rows only",
            "n_features_used": int(n_features_used),
        },
        "calibrator": {
            "kind": str(calibrator.kind),
            "slope": calibrator.slope,
            "intercept": calibrator.intercept,
            "fitted_on": "the 1,600 clean final-calibration rows only",
            "never_refitted_under_withholding": True,
        },
        "fitting_diagnostics_not_evaluation": {
            "note": (
                "In-sample for the calibrator by construction. Recorded to verify "
                "reproducibility of the freeze, NOT as evaluation evidence."
            ),
            "mean_raw_calibration_prediction": float(np.mean(raw_calibration)),
            "mean_calibrated_calibration_prediction": float(
                np.mean(calibrator.transform(np.asarray(raw_calibration, dtype=np.float64)))
            ),
        },
        "set_c_evaluation_contract": {
            **SET_C_CONTRACT,
            "frozen_control_seeds": frozen_control_seeds,
            "eligible_control_pool": control_pool,
            "eligible_control_pool_n": len(control_pool),
            "eligible_control_pool_includes_withheld_analytes": True,
            "eligible_control_pool_provenance": (
                "all variables covered by configs/panels.yaml catalogue version 2.0; "
                "identical to the M5-v2 development control pool"
            ),
            "fitted_objects": (
                "use exactly the three artifact-hashed fitted objects in this freeze; "
                "no refitting or substitution"
            ),
        },
        "set_c_access": {
            "loaded_during_freeze": False,
            "scored_during_freeze": False,
            "allow_final_holdout_passed": False,
            "realised_source_sets": list(realised_sets),
        },
    }
    freeze["provenance"] = build_provenance(
        cohort=truncated,
        splits=[],
        config_payload={
            "xgb": XGB_PARAMS,
            "cutoff": CUTOFF,
            "seed": args.seed,
            "n_final_train": N_FINAL_TRAIN,
            "n_final_calibration": N_FINAL_CALIBRATION,
            "frozen_pattern": list(FROZEN_PATTERN),
        },
        extra={"excluded_sets": ["c"], "catalogue_version": catalogue.version},
    )
    freeze["artifact_hashes"] = {
        model_path.name: sha256_file(model_path),
        imputer_path.name: sha256_file(imputer_path),
        calibrator_path.name: sha256_file(calibrator_path),
    }
    freeze_path = args.out / "final_freeze.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
        newline="\n",
    )

    _print_report(freeze, freeze_path)
    return 0


def _print_report(freeze: dict[str, Any], freeze_path: pathlib.Path) -> None:
    split = freeze["split"]
    print("\n" + "=" * 92)
    print("M5-v2 FINAL MODEL FREEZE — no evaluation performed")
    print("=" * 92)
    print(f"frozen pattern      : {' + '.join(freeze['frozen_pattern'])}")
    print(f"source sets         : {split['source_sets']}  (set-c never loaded)")
    print(
        f"partition           : {split['n_final_train']} train / "
        f"{split['n_final_calibration']} calibration, seed {split['seed']}, disjoint"
    )
    print(
        f"prevalence          : train {split['train_prevalence']:.5f}  "
        f"calibration {split['calibration_prevalence']:.5f}"
    )
    print(f"features            : {freeze['preprocessing']['n_features']}")
    print(f"model splits on     : {freeze['model']['n_features_used']} features")
    print(
        f"calibrator          : Platt slope {freeze['calibrator']['slope']:.6f} "
        f"intercept {freeze['calibrator']['intercept']:.6f}"
    )
    print(
        f"git sha             : {freeze['provenance']['git_sha'][:7]}  "
        f"dirty={freeze['provenance']['git_dirty']}"
    )
    print("\nartifact hashes (sha256):")
    for name, digest in sorted(freeze["artifact_hashes"].items()):
        print(f"  {name:<24} {digest}")
    print("\nset-c evaluation contract: FROZEN, NOT EXECUTED")
    print(
        f"  control seeds     : {freeze['set_c_evaluation_contract']['frozen_control_seeds']}"
    )
    print(f"\nwrote {freeze_path}")


if __name__ == "__main__":
    sys.exit(main())
