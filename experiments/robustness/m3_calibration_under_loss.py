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
import hashlib
import json
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
    eligible_columns,
    matched_trio,
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
    """Loss conditions with amount- and variable-matched controls."""
    if severity == 0.0:
        return {
            "none": apply_information_loss(
                cohort, LossCondition.NONE, 0.0, catalogue, seed=seed
            )
        }
    group, variable, cell = matched_trio(cohort, severity, catalogue, seed=seed)
    return {
        "group_structured": group,
        "variable_matched_scattered": variable,
        "cell_random": cell,
    }


def _loss_audit(
    cohort: Cohort,
    variants: dict[tuple[float, str], LossOutcome],
    catalogue: Any,
) -> dict[str, Any]:
    """Quantify variable identity, group selection, and mask matching."""
    columns, groups = eligible_columns(cohort, catalogue)
    variable_names = list(cohort.variable_names)
    report: dict[str, Any] = {
        "eligible_variable_names": [variable_names[int(col)] for col in columns],
        "semantics": (
            "A selected co-measurement group removes every naturally observed cell of each "
            "member variable across the full truncated patient window. It does not remove "
            "individual orders or patient-hour group instances."
        ),
        "variable_matched_fallback": (
            "Per-variable requests are deterministically clipped to naturally observed "
            "availability and the shortfall is recorded. No fallback was needed in the "
            "matched M3 trios because reference counts came from the same patient mask."
        ),
        "variants": {},
        "groups": {},
    }

    for (severity, condition), outcome in variants.items():
        key = f"{severity}|{condition}"
        group = variants.get((severity, "group_structured"))
        variables: list[dict[str, Any]] = []
        removed_totals = outcome.removed_by_variable.sum(axis=0)
        for col in columns:
            column = int(col)
            baseline = cohort.m[:, :, column].sum(axis=1, dtype=np.int64)
            removed = outcome.removed_by_variable[:, column]
            total_observed = int(baseline.sum())
            variables.append(
                {
                    "variable": variable_names[column],
                    "total_observed_cells": total_observed,
                    "total_removed_cells": int(removed.sum()),
                    "removed_proportion": (
                        float(removed.sum() / total_observed) if total_observed else 0.0
                    ),
                    "n_patients_with_removed_cells": int((removed > 0).sum()),
                    "mean_removed_cells_per_patient": float(removed.mean()),
                    "median_removed_cells_per_patient": float(np.median(removed)),
                }
            )

        total_removed = int(removed_totals.sum())
        variant_report: dict[str, Any] = {
            "mask_sha256": hashlib.sha256(outcome.cohort.m.tobytes()).hexdigest(),
            "total_removed_cells": total_removed,
            "variables": variables,
            "n_patients_with_match_mismatch": int((outcome.match_mismatch_cells > 0).sum()),
            "total_match_mismatch_cells": int(outcome.match_mismatch_cells.sum()),
        }
        if group is not None:
            group_totals = group.removed_by_variable.sum(axis=0).astype(np.float64)
            candidate_totals = removed_totals.astype(np.float64)
            if group_totals.sum() and candidate_totals.sum():
                tv = (
                    0.5
                    * np.abs(
                        group_totals / group_totals.sum()
                        - candidate_totals / candidate_totals.sum()
                    ).sum()
                )
            else:
                tv = 0.0
            patient_diff = np.any(outcome.cohort.m != group.cohort.m, axis=(1, 2))
            per_variable_diff = np.any(
                outcome.removed_by_variable != group.removed_by_variable, axis=1
            )
            variant_report.update(
                {
                    "removed_variable_distribution_tv_from_group": float(tv),
                    "n_patients_mask_differs_from_group": int(patient_diff.sum()),
                    "n_patients_variable_counts_differ_from_group": int(
                        per_variable_diff.sum()
                    ),
                }
            )
        report["variants"][key] = variant_report

    for severity in SEVERITIES:
        if severity == 0.0:
            continue
        group_outcome = variants[(severity, "group_structured")]
        group_rows: list[dict[str, Any]] = []
        for name, cols in groups.items():
            baseline = cohort.m[:, :, cols].sum(axis=(1, 2), dtype=np.int64)
            removed = group_outcome.removed_by_variable[:, cols].sum(axis=1, dtype=np.int64)
            selected = (baseline > 0) & (removed == baseline)
            group_rows.append(
                {
                    "group": name,
                    "variables": [variable_names[int(col)] for col in cols],
                    "n_patients_present": int((baseline > 0).sum()),
                    "n_patients_removed": int(selected.sum()),
                    "total_removed_cells": int(removed.sum()),
                }
            )
        report["groups"][str(severity)] = group_rows
    return report


def _baseline_variable_importance(
    importances: list[np.ndarray], feature_names: tuple[str, ...], eligible: list[str]
) -> dict[str, Any]:
    """Aggregate XGBoost gain importance by source variable, descriptively."""
    rows: list[dict[str, Any]] = []
    for variable in eligible:
        indices = [
            i
            for i, name in enumerate(feature_names)
            if name.split("::", maxsplit=1)[1] == variable
        ]
        values = np.array([float(importance[indices].sum()) for importance in importances])
        rows.append(
            {
                "variable": variable,
                "mean_normalized_gain": float(values.mean()),
                "min_fold": float(values.min()),
                "max_fold": float(values.max()),
            }
        )
    return {
        "description": (
            "Descriptive mean of XGBoost normalized gain importance across the five clean "
            "model-training folds, aggregated over all values_mask features for each analyte. "
            "Not a causal importance measure."
        ),
        "n_folds": len(importances),
        "variables": rows,
    }


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
        built = _build_loss_variants(truncated, catalogue, severity, args.seed)
        repeated = _build_loss_variants(truncated, catalogue, severity, args.seed)
        for name, outcome in built.items():
            if not np.array_equal(outcome.cohort.m, repeated[name].cohort.m):
                raise RuntimeError(
                    f"information loss is not deterministic for {severity}|{name}"
                )
            variants[(severity, name)] = outcome
    severity_report = {
        f"{sev}|{name}": outcome.summary() for (sev, name), outcome in variants.items()
    }
    for key, summary in severity_report.items():
        log.info("loss variant", variant=key, **summary)
    loss_audit = _loss_audit(truncated, variants, catalogue)

    # Features are deterministic per patient, so build them once per variant.
    feature_cache: dict[tuple[float, str, str], np.ndarray] = {}
    feature_names: dict[str, tuple[str, ...]] = {}
    for (sev, name), outcome in variants.items():
        for rep in representations:
            view = build_representation(outcome.cohort, rep)
            feature_cache[(sev, name, str(rep))] = view.x
            if str(rep) in feature_names and feature_names[str(rep)] != view.names:
                raise RuntimeError(f"feature names changed under stress for {rep}")
            feature_names[str(rep)] = view.names

    # ---- fit per fold, evaluate every condition on the outer test ----------
    keys: list[tuple[str, str, str, float, str]] = []
    predictions: dict[tuple[str, str, str, float, str], np.ndarray] = {}
    fold_report: list[dict[str, Any]] = []
    fold_id = np.full(cohort.n_patients, -1, dtype=np.int64)
    xgb_importances: list[np.ndarray] = []
    calibration_fit_report: list[dict[str, Any]] = []
    calibration_application_report: list[dict[str, Any]] = []

    for split in splits:
        # model-train / calibration split, carved from outer-train only.
        train_idx, calib_idx = train_test_split(
            split.train,
            test_size=CALIBRATION_FRACTION,
            random_state=args.seed + split.fold,
            stratify=y[split.train],
        )
        test_idx = split.validation
        fold_id[test_idx] = split.fold
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
                if rep is PRIMARY_REPRESENTATION and kind == "xgboost":
                    xgb_importances.append(
                        np.asarray(model.feature_importances_, dtype=np.float64)
                    )
                # Calibrators fit on CLEAN calibration predictions.
                p_calib = _predict(model, scaler, x_calib)
                fitted: dict[CalibratorKind, Any] = {}
                for ck in calibrators:
                    cal = build_calibrator(ck)
                    cal.fit(p_calib, y[calib_idx])
                    fitted[ck] = cal
                calibration_fit_report.append(
                    {
                        "fold": split.fold,
                        "representation": str(rep),
                        "model": kind,
                        "n_calibration": len(calib_idx),
                        "n_events": int(y[calib_idx].sum()),
                        "raw_prediction_min": float(p_calib.min()),
                        "raw_prediction_max": float(p_calib.max()),
                        "calibrators": {str(ck): cal.config() for ck, cal in fitted.items()},
                    }
                )

                for (sev, cond), _ in variants.items():
                    x_test = imputer.transform(
                        feature_cache[(sev, cond, str(rep))][test_idx],
                        draw_seed=200 + split.fold,
                    )
                    p_raw = _predict(model, scaler, x_test)
                    isotonic_config = fitted[CalibratorKind.ISOTONIC].config()
                    support_min = float(isotonic_config["calibration_min"])
                    support_max = float(isotonic_config["calibration_max"])
                    calibration_application_report.append(
                        {
                            "fold": split.fold,
                            "representation": str(rep),
                            "model": kind,
                            "severity": sev,
                            "condition": cond,
                            "n_outer_test": len(test_idx),
                            "n_below_clean_isotonic_support": int((p_raw < support_min).sum()),
                            "n_above_clean_isotonic_support": int((p_raw > support_max).sum()),
                        }
                    )
                    for ck, cal in fitted.items():
                        prediction_key = (str(rep), kind, str(ck), sev, cond)
                        if prediction_key not in predictions:
                            predictions[prediction_key] = np.full(cohort.n_patients, np.nan)
                            keys.append(prediction_key)
                        predictions[prediction_key][test_idx] = cal.transform(p_raw)
                log.info(
                    "fold done",
                    fold=split.fold,
                    representation=str(rep),
                    model=kind,
                    seconds=round(time.perf_counter() - t0, 1),
                )

    for prediction_key, arr in predictions.items():
        if not np.isfinite(arr).all():
            raise RuntimeError(f"incomplete out-of-fold predictions for {prediction_key}")
    if bool((fold_id < 0).any()):
        raise RuntimeError("fold identity is incomplete")

    eligible_names = list(loss_audit["eligible_variable_names"])
    baseline_importance = _baseline_variable_importance(
        xgb_importances, feature_names[str(PRIMARY_REPRESENTATION)], eligible_names
    )

    # ---- metrics -----------------------------------------------------------
    rows: list[dict[str, Any]] = []
    for prediction_key in keys:
        row_rep, row_kind, row_calibrator, row_severity, row_condition = prediction_key
        p = predictions[prediction_key]
        row: dict[str, Any] = {
            "representation": row_rep,
            "model": row_kind,
            "calibrator": row_calibrator,
            "severity": row_severity,
            "condition": row_condition,
            "metrics": {name: float(fn(y, p)) for name, fn in METRICS.items()},
            "reliability": reliability_curve(y, p),
            "risk_coverage": downsample_curve(risk_coverage_curve(y, p)),
        }
        rows.append(row)

    # ---- original and Review #3 controls: group - matched controls --------
    contrasts: list[dict[str, Any]] = []
    for rep in representations:
        for kind in models:
            for ck in calibrators:
                for sev in SEVERITIES:
                    if sev == 0.0:
                        continue
                    for control in ("cell_random", "variable_matched_scattered"):
                        a = (str(rep), kind, str(ck), sev, control)
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
                                name_a=f"{control}@{sev}",
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
                                    "inferential_status": (
                                        "original_predeclared"
                                        if control == "cell_random"
                                        else "review_3_post_hoc_control"
                                    ),
                                    **diff.as_dict(),
                                }
                            )

    calibration_contrasts: list[dict[str, Any]] = []
    for sev in SEVERITIES:
        condition = "none" if sev == 0.0 else "group_structured"
        raw_key = (
            str(PRIMARY_REPRESENTATION),
            "xgboost",
            str(CalibratorKind.IDENTITY),
            sev,
            condition,
        )
        platt_key = (
            str(PRIMARY_REPRESENTATION),
            "xgboost",
            str(CalibratorKind.PLATT),
            sev,
            condition,
        )
        for metric_name in ("nll", "brier"):
            diff = paired_bootstrap_difference(
                y,
                predictions[raw_key],
                predictions[platt_key],
                METRICS[metric_name],
                metric_name=metric_name,
                name_a=f"raw@{sev}|{condition}",
                name_b=f"platt@{sev}|{condition}",
                n_boot=args.n_boot,
                seed=args.seed,
            )
            calibration_contrasts.append(
                {
                    "representation": str(PRIMARY_REPRESENTATION),
                    "model": "xgboost",
                    "severity": sev,
                    "condition": condition,
                    "inferential_status": "exploratory_calibrator_comparison",
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
            "review_3_control": str(LossCondition.VARIABLE_MATCHED_SCATTERED),
        },
        extra={
            "task": "T1_in_hospital_mortality",
            "excluded_sets": ["c"],
            "calibration_split_hash": stable_hash(fold_report),
            "fold_partitions": fold_report,
            "design_document": "docs/M3_DESIGN.md",
            "review_status": (
                "Review #3 post-hoc falsification control; not part of the original "
                "predeclared M3 contrasts"
            ),
        },
    )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps(
            {
                "schema": "cliniverse.m3/2",
                "provenance": provenance,
                "severity_report": severity_report,
                "information_loss_audit": loss_audit,
                "baseline_variable_importance": baseline_importance,
                "calibration_fit_report": calibration_fit_report,
                "calibration_application_report": calibration_application_report,
                "rows": rows,
                "contrasts": contrasts,
                "calibration_contrasts": calibration_contrasts,
            },
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
        newline="\n",
    )
    array_payload = {
        "labels": y,
        "record_ids": cohort.record_ids,
        "source_set": cohort.source_set,
        "fold_id": fold_id,
        "loss_variable_names": np.asarray(cohort.variable_names, dtype=np.str_),
        **{
            f"loss|{sev}|{name}|removed_cells": outcome.removed_cells
            for (sev, name), outcome in variants.items()
        },
        **{
            f"loss|{sev}|{name}|eligible_cells": outcome.eligible_cells
            for (sev, name), outcome in variants.items()
        },
        **{
            f"loss|{sev}|{name}|removed_by_variable": outcome.removed_by_variable
            for (sev, name), outcome in variants.items()
        },
        **{
            "|".join(str(part) for part in prediction_key): predictions[prediction_key]
            for prediction_key in keys
        },
    }
    np.savez_compressed(args.out / "predictions.npz", **array_payload)  # type: ignore[arg-type]

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
    print("M3 CONTRASTS - GROUP_STRUCTURED minus MATCHED CONTROLS (paired, same patients)")
    print("Positive = group loss is worse. * = 95% interval excludes zero.")
    print("=" * 118)
    print(
        f"{'representation':<14}{'model':<9}{'calibrator':<14}{'sev':>6}"
        f"{'control':<36}{'metric':<8}{'difference':>28}"
    )
    print("-" * 118)
    for c in contrasts:
        flag = "*" if c["excludes_zero"] else " "
        text = f"{c['difference']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]"
        print(
            f"{flag}{c['representation']:<13}{c['model']:<9}{c['calibrator']:<14}"
            f"{c['severity']:>6.2f}{c['name_a']:<36}{c['metric']:<8}{text:>28}"
        )


if __name__ == "__main__":
    sys.exit(main())
