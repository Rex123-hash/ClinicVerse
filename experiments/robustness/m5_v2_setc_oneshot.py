"""M5-v2 ONE-SHOT SET-C CONFIRMATION — executes the frozen contract, once.

Frozen in `docs/M5_V2_DESIGN.md` section 9 and in the machine-readable
`set_c_evaluation_contract` block of the final freeze artifact. This script
executes that contract and nothing else.

**No search. No tuning. No refitting.** The model, imputer and calibrator are
deserialised from the artifact-hashed freeze package and used as-is. The pattern
is read from the frozen contract. The five control seeds are read from the frozen
contract. Every one of those is verified against the recorded hashes and values
before set-c is unlocked, and the run aborts on any mismatch.

Set-c is unlocked exactly once, through `final_holdout()` with its explicit token,
after all verification has passed.

Usage:
    python experiments/robustness/m5_v2_setc_oneshot.py --i-have-read-the-contract
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import sys
from typing import Any

import numpy as np
import xgboost as xgb

from cliniverse.acquisition import load_panel_catalogue
from cliniverse.data import load_cohort
from cliniverse.data.splits import UNLOCK_TOKEN, final_holdout
from cliniverse.evaluation.artifacts import build_provenance, stable_hash
from cliniverse.evaluation.calibration import CalibratorKind, PlattCalibrator
from cliniverse.evaluation.failure_search import (
    apply_analyte_subset_loss,
    matched_random_control,
)
from cliniverse.evaluation.metrics import (
    auprc,
    auroc,
    brier_score,
    calibration_intercept,
    calibration_slope,
    negative_log_likelihood,
    paired_mean_difference_bootstrap,
    per_patient_log_loss,
)
from cliniverse.evaluation.representations import (
    FittedImputer,
    ImputationStrategy,
    Representation,
    build_representation,
)
from cliniverse.exceptions import ConfigError
from cliniverse.log import get_logger

log = get_logger(__name__)

FREEZE_DIR = pathlib.Path("experiments/robustness/results/m5v2_final_freeze")
REPRESENTATION = Representation.VALUES_MASK
CUTOFF = 24
AUROC_DELTA = 0.02


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen() -> tuple[xgb.Booster, FittedImputer, PlattCalibrator, dict[str, Any]]:
    """Deserialise the frozen package and verify it against its recorded hashes."""
    freeze = json.loads((FREEZE_DIR / "final_freeze.json").read_text(encoding="utf-8"))
    recorded: dict[str, str] = freeze["artifact_hashes"]
    for name, expected in sorted(recorded.items()):
        actual = sha256_file(FREEZE_DIR / name)
        if actual != expected:
            raise ConfigError(
                f"frozen artifact {name} does not match its recorded hash: "
                f"expected {expected}, found {actual}"
            )
        log.info("artifact verified", name=name, sha256=actual)

    booster = xgb.Booster()
    booster.load_model(str(FREEZE_DIR / "final_model.json"))

    with np.load(FREEZE_DIR / "final_imputer.npz", allow_pickle=False) as archive:
        strategy = ImputationStrategy(str(archive["strategy"]))
        if strategy is not ImputationStrategy.MEDIAN:
            raise ConfigError(f"frozen imputer is {strategy}, expected median")
        imputer = FittedImputer(
            strategy=strategy,
            medians=np.asarray(archive["medians"], dtype=np.float64),
            # Unused by the median strategy; the frozen package stores no pools.
            pools=(),
            jitter_scales=np.asarray(archive["jitter_scales"], dtype=np.float64),
            seed=int(archive["seed"]),
        )

    stored = json.loads((FREEZE_DIR / "final_calibrator.json").read_text(encoding="utf-8"))
    if stored["kind"] != str(CalibratorKind.PLATT) or not stored["fitted"]:
        raise ConfigError(f"frozen calibrator is not a fitted Platt calibrator: {stored}")
    calibrator = PlattCalibrator(
        kind=CalibratorKind.PLATT,
        slope=float(stored["slope"]),
        intercept=float(stored["intercept"]),
        fitted=True,
    )
    if calibrator.slope != freeze["calibrator"]["slope"]:
        raise ConfigError("calibrator slope disagrees with the freeze record")
    return booster, imputer, calibrator, freeze


def predict(
    features: np.ndarray,
    booster: xgb.Booster,
    imputer: FittedImputer,
    calibrator: PlattCalibrator,
    *,
    draw_seed: int,
) -> np.ndarray:
    """The frozen pipeline, applied exactly as fitted. Nothing here is refitted."""
    x = imputer.transform(features, draw_seed=draw_seed)
    raw = np.asarray(booster.predict(xgb.DMatrix(x)), dtype=np.float64)
    return np.asarray(calibrator.transform(raw), dtype=np.float64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--i-have-read-the-contract",
        action="store_true",
        help="required acknowledgement that this unlocks the holdout, once",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("experiments/robustness/results/m5v2_setc"),
    )
    args = parser.parse_args(argv)
    if not args.i_have_read_the_contract:
        raise SystemExit(
            "refusing to run: this unlocks the set-c holdout for a single "
            "pre-registered test. Pass --i-have-read-the-contract."
        )

    started_at = dt.datetime.now(dt.UTC).isoformat()
    booster, imputer, calibrator, freeze = load_frozen()
    contract = freeze["set_c_evaluation_contract"]
    pattern: tuple[str, ...] = tuple(contract["pattern"])
    control_seeds: list[int] = list(contract["frozen_control_seeds"])
    n_bootstrap = int(contract["n_bootstrap"])
    bootstrap_seed = int(contract["bootstrap_seed"])
    repeats = int(contract["control_repeats"])
    if pattern != ("BUN", "Glucose", "Na"):
        raise ConfigError(f"frozen pattern changed: {pattern}")
    if len(control_seeds) != repeats:
        raise ConfigError("frozen control seed count does not match control_repeats")
    log.info(
        "frozen contract",
        pattern=list(pattern),
        control_seeds=control_seeds,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
    )

    catalogue = load_panel_catalogue()

    # ---------------------------------------------------------------------
    # Pre-flight integrity check, on DEVELOPMENT data only. The deserialised
    # pipeline must reproduce the mean raw prediction the freeze recorded on its
    # 1,600 calibration rows. If the reload were subtly wrong, this fails here
    # rather than silently corrupting the one-shot holdout result.
    # ---------------------------------------------------------------------
    from sklearn.model_selection import train_test_split

    from cliniverse.data.splits import development_cohort

    dev = development_cohort(load_cohort())
    y_dev = dev.labels["mortality"].astype(np.float64)
    dev_clean = build_representation(dev.truncate(CUTOFF), REPRESENTATION).x
    _, calib_idx = train_test_split(
        np.arange(dev.n_patients),
        test_size=freeze["split"]["n_final_calibration"],
        random_state=freeze["split"]["seed"],
        stratify=y_dev,
    )
    calib_idx = np.sort(calib_idx)
    raw_calibration = booster.predict(
        xgb.DMatrix(
            imputer.transform(dev_clean[calib_idx], draw_seed=100 + freeze["split"]["seed"])
        )
    )
    recorded_mean = float(
        freeze["fitting_diagnostics_not_evaluation"]["mean_raw_calibration_prediction"]
    )
    observed_mean = float(np.mean(raw_calibration))
    if abs(observed_mean - recorded_mean) > 1e-9:
        raise ConfigError(
            "reloaded frozen pipeline does not reproduce the freeze diagnostic: "
            f"recorded {recorded_mean!r}, observed {observed_mean!r}"
        )
    log.info("preflight ok", recorded=recorded_mean, observed=observed_mean)
    del dev, dev_clean, y_dev

    # ---------------------------------------------------------------------
    # THE SINGLE UNLOCK. Everything above ran on development data.
    # ---------------------------------------------------------------------
    log.info("unlocking set-c", note="single pre-registered use")
    holdout = final_holdout(load_cohort(sets=("c",), allow_final_holdout=True), UNLOCK_TOKEN)
    realised = tuple(sorted(set(holdout.source_set.tolist())))
    if realised != ("c",):
        raise ConfigError(f"holdout must be set-c only, got {realised}")
    y = holdout.labels["mortality"].astype(np.float64)
    truncated = holdout.truncate(CUTOFF)
    log.info("set-c loaded", n=holdout.n_patients, prevalence=float(y.mean()))

    # ------------------------------------------------- clean and withheld --
    clean_features = build_representation(truncated, REPRESENTATION).x
    p_clean = predict(clean_features, booster, imputer, calibrator, draw_seed=0)

    loss = apply_analyte_subset_loss(truncated, pattern, catalogue)
    p_withheld = predict(
        build_representation(loss.cohort, REPRESENTATION).x,
        booster,
        imputer,
        calibrator,
        draw_seed=0,
    )

    # ------------------------------------------------- amount-matched R=5 --
    control_predictions: list[np.ndarray] = []
    control_losses = np.zeros((repeats, holdout.n_patients), dtype=np.float64)
    for r, seed in enumerate(control_seeds):
        control = matched_random_control(truncated, loss.removed_cells, catalogue, seed=seed)
        removed = (truncated.m & ~control.m).sum(axis=(1, 2))
        if not np.array_equal(removed, loss.removed_cells):
            raise ConfigError(f"amount matching failed for control draw {r} (seed {seed})")
        p_control = predict(
            build_representation(control, REPRESENTATION).x,
            booster,
            imputer,
            calibrator,
            draw_seed=0,
        )
        control_predictions.append(p_control)
        control_losses[r] = per_patient_log_loss(y, p_control)

    # ------------------------------------------------- the primary test ----
    d = per_patient_log_loss(y, p_withheld) - control_losses.mean(axis=0)
    delta_c = float(d.mean())
    # percentiles=(5, 95) -> `.low` is the 5th percentile, i.e. the one-sided
    # 95% LOWER bound the contract specifies.
    interval = paired_mean_difference_bootstrap(
        y,
        d,
        metric_name="delta_c_excess_nll",
        name_a="amount_matched_random",
        name_b="+".join(pattern),
        n_boot=n_bootstrap,
        seed=bootstrap_seed,
        percentiles=(5.0, 95.0),
    )
    lower_bound = float(interval.low)

    clean_auroc = float(auroc(y, p_clean))
    withheld_auroc = float(auroc(y, p_withheld))
    auroc_drop = clean_auroc - withheld_auroc

    primary_pass = bool(lower_bound > 0.0)
    silent_pass = bool(auroc_drop <= AUROC_DELTA)
    confirmed = bool(primary_pass and silent_pass)

    # ------------------------------------------------- artifacts ----------
    args.out.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "record_ids": holdout.record_ids,
        "labels": y,
        "d_i": d,
        "p_clean": p_clean,
        "p_withheld": p_withheld,
        "removed_cells": loss.removed_cells,
        "control_log_losses": control_losses,
        "control_seeds": np.asarray(control_seeds, dtype=np.int64),
    }
    for r, p_control in enumerate(control_predictions):
        arrays[f"p_control_{r}"] = p_control
    np.savez_compressed(args.out / "setc_oneshot_predictions.npz", **arrays)  # type: ignore[arg-type]

    def descriptive(p: np.ndarray) -> dict[str, float]:
        return {
            "nll": float(negative_log_likelihood(y, p)),
            "brier": float(brier_score(y, p)),
            "auroc": float(auroc(y, p)),
            "auprc": float(auprc(y, p)),
            "calibration_intercept": float(calibration_intercept(y, p)),
            "calibration_slope": float(calibration_slope(y, p)),
            "mean_predicted_risk": float(p.mean()),
        }

    payload: dict[str, Any] = {
        "schema": "cliniverse.m5v2.setc_oneshot/1",
        "design_document": "docs/M5_V2_DESIGN.md#9",
        "stage": "ONE-SHOT SET-C CONFIRMATION — executed once, per the frozen contract",
        "started_at_utc": started_at,
        "finished_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "frozen_contract_as_executed": contract,
        "frozen_artifact_hashes": freeze["artifact_hashes"],
        "freeze_source_git_sha": freeze["provenance"]["git_sha"],
        "cohort": {
            "sets": list(realised),
            "n_patients": int(holdout.n_patients),
            "prevalence": float(y.mean()),
            "n_deaths": int(y.sum()),
        },
        "primary": {
            "pattern": list(pattern),
            "delta_c": delta_c,
            "one_sided_95_lower_bound": lower_bound,
            "n_bootstrap": n_bootstrap,
            "n_valid_resamples": interval.n_boot,
            "bootstrap_seed": bootstrap_seed,
            "decision_rule": "PASS if and only if LB > 0",
            "passes": primary_pass,
        },
        "discrimination_silent": {
            "clean_auroc": clean_auroc,
            "withheld_auroc": withheld_auroc,
            "auroc_drop": auroc_drop,
            "delta": AUROC_DELTA,
            "passes": silent_pass,
        },
        "confirmation": {
            "passes": confirmed,
            "rule": "primary LB > 0 AND AUROC drop <= 0.02",
        },
        "secondary_descriptive": {
            "clean": descriptive(p_clean),
            "withheld": descriptive(p_withheld),
            "mean_removed_cells_per_patient": float(loss.removed_cells.mean()),
            "mean_realized_severity": float(loss.realized_severity.mean()),
            "note": "Descriptive only. Not part of the frozen decision rule.",
        },
        "monte_carlo_limitation": contract["monte_carlo_limitation"],
        "historical_disclosure": (
            "No Set-C patient-level information was retained or used for model "
            "fitting, model selection, failure-pattern selection, or any M5-v2 "
            "statistic after the aggregate audit."
        ),
        "alternative_analyses_run": {
            "count": 0,
            "note": (
                "Exactly one pattern, one control configuration, one bootstrap and "
                "one decision were executed. No alternative pattern, R, control "
                "pool, delta, calibration or bootstrap was run."
            ),
        },
        "record_ids_hash": stable_hash(holdout.record_ids.tolist()),
    }
    payload["provenance"] = build_provenance(
        cohort=truncated,
        splits=[],
        config_payload={
            "frozen_contract": contract,
            "freeze_hashes": freeze["artifact_hashes"],
        },
        extra={"evaluated_sets": ["c"], "catalogue_version": catalogue.version},
    )
    payload["predictions_file"] = {
        "name": "setc_oneshot_predictions.npz",
        "sha256": sha256_file(args.out / "setc_oneshot_predictions.npz"),
    }
    (args.out / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
        newline="\n",
    )
    _print_report(payload)
    print(f"\nwrote {args.out / 'results.json'}")
    return 0


def _print_report(payload: dict[str, Any]) -> None:
    primary = payload["primary"]
    silent = payload["discrimination_silent"]
    clean = payload["secondary_descriptive"]["clean"]
    withheld = payload["secondary_descriptive"]["withheld"]

    def verdict(passed: bool) -> str:
        return "PASS" if passed else "FAIL"

    print("\n" + "=" * 92)
    print("M5-v2 ONE-SHOT SET-C CONFIRMATION")
    print("=" * 92)
    print(
        f"cohort            : set-{payload['cohort']['sets'][0]}, "
        f"n={payload['cohort']['n_patients']}, "
        f"deaths={payload['cohort']['n_deaths']}, "
        f"prevalence={payload['cohort']['prevalence']:.5f}"
    )
    print(f"pattern           : {' + '.join(primary['pattern'])}")
    print("\nPRIMARY (frozen):")
    print(f"  Delta_C                    {primary['delta_c']:+.6f}")
    print(f"  one-sided 95% LOWER bound  {primary['one_sided_95_lower_bound']:+.6f}")
    print(
        f"  bootstrap                  {primary['n_bootstrap']} resamples, "
        f"seed {primary['bootstrap_seed']}, {primary['n_valid_resamples']} valid"
    )
    print(f"  LB > 0                     {verdict(primary['passes'])}")
    print("\nDISCRIMINATION-SILENT (frozen):")
    print(f"  clean AUROC                {silent['clean_auroc']:.6f}")
    print(f"  withheld AUROC             {silent['withheld_auroc']:.6f}")
    print(
        f"  drop                       {silent['auroc_drop']:+.6f}  (delta {silent['delta']})"
    )
    print(f"  drop <= delta              {verdict(silent['passes'])}")
    print(f"\nCONFIRMATION: {verdict(payload['confirmation']['passes'])}")
    print("\nsecondary descriptive (not part of the decision):")
    print(f"  {'':<22}{'clean':>12}{'withheld':>12}")
    for key in (
        "nll",
        "brier",
        "calibration_intercept",
        "calibration_slope",
        "mean_predicted_risk",
    ):
        print(f"  {key:<22}{clean[key]:>12.5f}{withheld[key]:>12.5f}")


if __name__ == "__main__":
    sys.exit(main())
