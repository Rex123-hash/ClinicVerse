"""M3: does confidence degrade appropriately when coherent information disappears?

Design is predeclared in `docs/M3_DESIGN.md` and was committed before this ran.

Per outer fold (5-fold stratified, sets a+b, n=8,000), development patients are
split three ways:

    model-train    fits the imputer, scaler and model
    calibration    fits the calibrator only, on CLEAN data
    outer test     fits nothing; evaluated under every loss condition

The imputer is fitted once per fold on clean model-train data and is **never**
refitted under stress, so the pipeline cannot adapt to the stress distribution.
The calibrator is fitted on clean calibration data, because the question is what
happens when a normally-calibrated model meets information loss.

Loss is applied to the cohort before feature construction, and group loss is
severity-matched to cell loss per patient.

Usage:
    python experiments/robustness/m3_calibration_under_loss.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from cliniverse.acquisition import load_panel_catalogue
from cliniverse.data import load_cohort
from cliniverse.data.cohort import Cohort
from cliniverse.data.splits import development_cohort, stratified_folds
from cliniverse.evaluation.artifacts import build_provenance, stable_hash
from cliniverse.evaluation.calibration import CalibratorKind, build_calibrator
from cliniverse.evaluation.information_loss import (
    LossCondition,
    LossOutcome,
    apply_information_loss,
    matched_pair,
)
from cliniverse.evaluation.metrics import (
    auprc,
    auroc,
    brier_score,
    calibration_intercept,
    calibration_slope,
    negative_log_likelihood,
    paired_bootstrap_difference,
    reliability_curve,
)
from cliniverse.evaluation.representations import (
    FittedImputer,
    ImputationStrategy,
    Representation,
    build_representation,
)
from cliniverse.evaluation.selective import (
    aurc,
    downsample_curve,
    mean_predicted_probability,
    mean_predictive_entropy,
    risk_coverage_curve,
)
from cliniverse.log import get_logger

log = get_logger(__name__)

# ------------------------------------------------------- frozen from M2 -----
# Modal M2 selection for values_mask::xgboost. Not re-searched. n_estimators is
# fixed rather than early-stopped: a stopping signal whose composition changes
# with the stress condition would make the model itself condition-dependent.
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
LR_C = 0.01

PRIMARY_REPRESENTATION = Representation.VALUES_MASK
SECONDARY_REPRESENTATIONS = (Representation.VALUES_ONLY,)
SEVERITIES: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75)
CALIBRATION_FRACTION = 0.25

METRICS = {
    "auroc": auroc,
    "auprc": auprc,
    "brier": brier_score,
    "nll": negative_log_likelihood,
    "aurc": aurc,
    "calibration_slope": calibration_slope,
    "calibration_intercept": calibration_intercept,
    "mean_predicted_probability": mean_predicted_probability,
    "mean_predictive_entropy": mean_predictive_entropy,
}
#: Predeclared primary contrast metrics (docs/M3_DESIGN.md section 10).
PRIMARY_CONTRAST = ("nll", "brier", "aurc")


def _make_model(kind: str, seed: int) -> Any:
    if kind == "xgboost":
        return XGBClassifier(random_state=seed, **XGB_PARAMS)
    return LogisticRegression(C=LR_C, max_iter=5000, random_state=seed)


def _fit_fold(
    x_train: np.ndarray, y_train: np.ndarray, kind: str, seed: int
) -> tuple[Any, StandardScaler | None]:
    model = _make_model(kind, seed)
    if kind == "logreg":
        scaler = StandardScaler().fit(x_train)
        model.fit(scaler.transform(x_train), y_train)
        return model, scaler
    model.fit(x_train, y_train)
    return model, None


def _predict(model: Any, scaler: StandardScaler | None, x: np.ndarray) -> np.ndarray:
    if scaler is not None:
        x = scaler.transform(x)
    return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)


def _build_loss_variants(
    cohort: Cohort, catalogue: Any, severity: float, seed: int
) -> dict[str, LossOutcome]:
    """Loss conditions for one severity, with group and cell matched per patient."""
    if severity == 0.0:
        return {
            "none": apply_information_loss(
                cohort, LossCondition.NONE, 0.0, catalogue, seed=seed
            )
        }
    group, cell = matched_pair(cohort, severity, catalogue, seed=seed)
    return {"group_structured": group, "cell_random": cell}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=int, default=24)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("experiments/robustness/results/m3"),
    )
    args = parser.parse_args(argv)

    catalogue = load_panel_catalogue()
    cohort = development_cohort(load_cohort())  # defaults to sets a+b; set-c never loaded
    y = cohort.labels["mortality"].astype(np.float64)
    splits = stratified_folds(cohort, n_folds=args.folds, seed=args.seed)
    log.info("cohort", n=cohort.n_patients, prevalence=float(y.mean()))

    representations = (PRIMARY_REPRESENTATION, *SECONDARY_REPRESENTATIONS)
    models = ("xgboost", "logreg")
    calibrators = (CalibratorKind.IDENTITY, CalibratorKind.PLATT, CalibratorKind.ISOTONIC)

    # ---- build every stressed cohort once, before any model is fitted -------
    truncated = cohort.truncate(args.cutoff)
    variants: dict[tuple[float, str], LossOutcome] = {}
    for severity in SEVERITIES:
        for name, outcome in _build_loss_variants(
            truncated, catalogue, severity, args.seed
        ).items():
            variants[(severity, name)] = outcome
    severity_report = {
        f"{sev}|{name}": outcome.summary() for (sev, name), outcome in variants.items()
    }
    for key, summary in severity_report.items():
        log.info("loss variant", variant=key, **summary)

    # Features are deterministic per patient, so build them once per variant.
    feature_cache: dict[tuple[float, str, str], np.ndarray] = {}
    for (sev, name), outcome in variants.items():
        for rep in representations:
            feature_cache[(sev, name, str(rep))] = build_representation(outcome.cohort, rep).x

    # ---- fit per fold, evaluate every condition on the outer test ----------
    keys: list[tuple[str, str, str, float, str]] = []
    predictions: dict[tuple[str, str, str, float, str], np.ndarray] = {}
    fold_report: list[dict[str, Any]] = []

    for split in splits:
        # model-train / calibration split, carved from outer-train only.
        train_idx, calib_idx = train_test_split(
            split.train,
            test_size=CALIBRATION_FRACTION,
            random_state=args.seed + split.fold,
            stratify=y[split.train],
        )
        test_idx = split.validation
        fold_report.append(
            {
                "fold": split.fold,
                "n_model_train": len(train_idx),
                "n_calibration": len(calib_idx),
                "n_outer_test": len(test_idx),
            }
        )

        for rep in representations:
            clean = feature_cache[(0.0, "none", str(rep))]
            # Imputer fitted on CLEAN model-train only; never refitted under stress.
            imputer = FittedImputer.fit(
                clean[train_idx],
                strategy=ImputationStrategy.MEDIAN,
                seed=args.seed + split.fold,
            )
            x_train = imputer.transform(clean[train_idx], draw_seed=split.fold)
            x_calib = imputer.transform(clean[calib_idx], draw_seed=100 + split.fold)

            for kind in models:
                t0 = time.perf_counter()
                model, scaler = _fit_fold(x_train, y[train_idx], kind, args.seed + split.fold)
                # Calibrators fit on CLEAN calibration predictions.
                p_calib = _predict(model, scaler, x_calib)
                fitted: dict[CalibratorKind, Any] = {}
                for ck in calibrators:
                    cal = build_calibrator(ck)
                    cal.fit(p_calib, y[calib_idx])
                    fitted[ck] = cal

                for (sev, cond), _ in variants.items():
                    x_test = imputer.transform(
                        feature_cache[(sev, cond, str(rep))][test_idx],
                        draw_seed=200 + split.fold,
                    )
                    p_raw = _predict(model, scaler, x_test)
                    for ck, cal in fitted.items():
                        key = (str(rep), kind, str(ck), sev, cond)
                        if key not in predictions:
                            predictions[key] = np.full(cohort.n_patients, np.nan)
                            keys.append(key)
                        predictions[key][test_idx] = cal.transform(p_raw)
                log.info(
                    "fold done",
                    fold=split.fold,
                    representation=str(rep),
                    model=kind,
                    seconds=round(time.perf_counter() - t0, 1),
                )

    for key, arr in predictions.items():
        if not np.isfinite(arr).all():
            raise RuntimeError(f"incomplete out-of-fold predictions for {key}")

    # ---- metrics -----------------------------------------------------------
    rows: list[dict[str, Any]] = []
    for key in keys:
        rep, kind, ck, sev, cond = key
        p = predictions[key]
        row: dict[str, Any] = {
            "representation": rep,
            "model": kind,
            "calibrator": ck,
            "severity": sev,
            "condition": cond,
            "metrics": {name: float(fn(y, p)) for name, fn in METRICS.items()},
            "reliability": reliability_curve(y, p),
            "risk_coverage": downsample_curve(risk_coverage_curve(y, p)),
        }
        rows.append(row)

    # ---- predeclared primary contrast: group - matched cell ---------------
    contrasts: list[dict[str, Any]] = []
    for rep in representations:
        for kind in models:
            for ck in calibrators:
                for sev in SEVERITIES:
                    if sev == 0.0:
                        continue
                    a = (str(rep), kind, str(ck), sev, "cell_random")
                    b = (str(rep), kind, str(ck), sev, "group_structured")
                    if a not in predictions or b not in predictions:
                        continue
                    for metric_name in PRIMARY_CONTRAST:
                        diff = paired_bootstrap_difference(
                            y,
                            predictions[a],
                            predictions[b],
                            METRICS[metric_name],
                            metric_name=metric_name,
                            name_a=f"cell_random@{sev}",
                            name_b=f"group_structured@{sev}",
                            n_boot=args.n_boot,
                            seed=args.seed,
                        )
                        contrasts.append(
                            {
                                "representation": str(rep),
                                "model": kind,
                                "calibrator": str(ck),
                                "severity": sev,
                                **diff.as_dict(),
                            }
                        )

    provenance = build_provenance(
        cohort=truncated,
        splits=splits,
        config_payload={
            "xgb_params": XGB_PARAMS,
            "lr_C": LR_C,
            "severities": SEVERITIES,
            "calibration_fraction": CALIBRATION_FRACTION,
            "primary_representation": str(PRIMARY_REPRESENTATION),
            "imputation": str(ImputationStrategy.MEDIAN),
            "cutoff": args.cutoff,
            "seed": args.seed,
        },
        extra={
            "task": "T1_in_hospital_mortality",
            "excluded_sets": ["c"],
            "calibration_split_hash": stable_hash(fold_report),
            "fold_partitions": fold_report,
            "design_document": "docs/M3_DESIGN.md",
        },
    )

    args.out.mkdir(parents=True, exist_ok=True)
    import json

    (args.out / "results.json").write_text(
        json.dumps(
            {
                "schema": "cliniverse.m3/1",
                "provenance": provenance,
                "severity_report": severity_report,
                "rows": rows,
                "contrasts": contrasts,
            },
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
        newline="\n",
    )
    np.savez_compressed(
        args.out / "predictions.npz",
        labels=y,
        record_ids=cohort.record_ids,
        **{"|".join(str(k) for k in key): predictions[key] for key in keys},  # type: ignore[arg-type]
    )

    _report(rows, contrasts, severity_report, y)
    print(f"\nwrote {args.out / 'results.json'}")
    print(f"wrote {args.out / 'predictions.npz'}")
    return 0


def _report(
    rows: list[dict[str, Any]],
    contrasts: list[dict[str, Any]],
    severity_report: dict[str, dict[str, Any]],
    y: np.ndarray,
) -> None:
    print("\n" + "=" * 118)
    print("M3 — REALIZED SEVERITY (group loss is indivisible, so it overshoots the request)")
    print("=" * 118)
    print(
        f"{'variant':<34}{'requested':>11}{'realized mean':>15}"
        f"{'median':>9}{'p10':>8}{'p90':>8}{'cells removed':>16}"
    )
    print("-" * 118)
    for key in sorted(severity_report):
        s = severity_report[key]
        print(
            f"{key:<34}{s['requested_severity']:>11.2f}{s['realized_severity_mean']:>15.3f}"
            f"{s['realized_severity_median']:>9.3f}{s['realized_severity_p10']:>8.3f}"
            f"{s['realized_severity_p90']:>8.3f}{s['total_removed_cells']:>16,}"
        )

    for rep in sorted({r["representation"] for r in rows}):
        for kind in sorted({r["model"] for r in rows}):
            print("\n" + "=" * 118)
            print(f"M3 — {rep} / {kind}  (n={len(y):,}, prevalence={y.mean():.2%})")
            print("=" * 118)
            print(
                f"{'calibrator':<14}{'sev':>6}{'condition':<18}{'AUROC':>8}{'AP':>8}"
                f"{'Brier':>9}{'NLL':>9}{'AURC':>9}{'slope':>8}"
                f"{'intcpt':>8}{'mean p':>9}{'entropy':>9}"
            )
            print("-" * 118)
            for r in rows:
                if r["representation"] != rep or r["model"] != kind:
                    continue
                m = r["metrics"]
                print(
                    f"{r['calibrator']:<14}{r['severity']:>6.2f}{r['condition']:<18}"
                    f"{m['auroc']:>8.4f}{m['auprc']:>8.4f}{m['brier']:>9.4f}{m['nll']:>9.4f}"
                    f"{m['aurc']:>9.4f}{m['calibration_slope']:>8.3f}"
                    f"{m['calibration_intercept']:>8.3f}"
                    f"{m['mean_predicted_probability']:>9.4f}"
                    f"{m['mean_predictive_entropy']:>9.4f}"
                )

    print("\n" + "=" * 118)
    print(
        "M3 PRIMARY CONTRAST - GROUP_STRUCTURED minus MATCHED CELL_RANDOM "
        "(paired, same patients)"
    )
    print("Positive = group loss is worse. * = 95% interval excludes zero.")
    print("=" * 118)
    print(
        f"{'representation':<14}{'model':<9}{'calibrator':<14}{'sev':>6}{'metric':<8}{'difference':>28}"
    )
    print("-" * 118)
    for c in contrasts:
        flag = "*" if c["excludes_zero"] else " "
        text = f"{c['difference']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]"
        print(
            f"{flag}{c['representation']:<13}{c['model']:<9}{c['calibrator']:<14}"
            f"{c['severity']:>6.2f}{c['metric']:<8}{text:>28}"
        )


if __name__ == "__main__":
    sys.exit(main())
