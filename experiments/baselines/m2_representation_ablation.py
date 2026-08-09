"""M2: how much mortality signal is in measurement presence, and how much in values?

Compares three representations on identical patients, identical folds and an
identical 24h information boundary:

    mask_only     measurement-presence patterns only, no clinical value
    values_only   clinical values only, no explicit presence features
    values_mask   both

plus a prevalence floor and two supplementary views (statics).

Protocol
--------
Outer 5-fold stratified CV over development patients (sets a+b) produces
out-of-fold predictions for every patient exactly once. Hyperparameters are
selected *inside* each outer training fold on a held-out inner validation split,
from a compact predeclared grid. Imputation and scaling are fitted on inner or
outer training rows only and never see validation rows. set-c is never loaded.

Comparisons use paired patient-level bootstrap on identical out-of-fold
predictions, not overlapping standalone intervals.

Usage:
    python experiments/baselines/m2_representation_ablation.py [--folds 5] [--n-boot 2000]
"""

from __future__ import annotations

import argparse
import itertools
import pathlib
import sys
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from cliniverse.data import load_cohort
from cliniverse.data.splits import development_cohort, stratified_folds
from cliniverse.evaluation.artifacts import RunArtifact, build_provenance, write_run
from cliniverse.evaluation.metrics import (
    METRIC_FUNCTIONS,
    bootstrap_metric,
    classification_metrics,
    paired_bootstrap_difference,
    reliability_curve,
)
from cliniverse.evaluation.representations import (
    CORE_REPRESENTATIONS,
    FittedImputer,
    ImputationStrategy,
    Representation,
    build_representation,
)
from cliniverse.log import get_logger

log = get_logger(__name__)

# --------------------------------------------------------- predeclared grids --
# Compact and fixed before any result was seen. Selection is by inner-validation
# AUROC. Small grids are a deliberate guard against leaderboard overfitting.
LR_GRID: dict[str, list[Any]] = {"C": [0.01, 0.1, 1.0, 10.0]}

XGB_GRID: dict[str, list[Any]] = {
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "min_child_weight": [1, 10],
}
XGB_FIXED: dict[str, Any] = {
    "n_estimators": 600,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "eval_metric": "logloss",
    "early_stopping_rounds": 50,
    "tree_method": "hist",
    "n_jobs": 4,
}

SELECTION_METRIC = "inner_validation_auroc"
INNER_VALIDATION_FRACTION = 0.2


def _grid(space: dict[str, list[Any]]) -> Iterator[dict[str, Any]]:
    keys = sorted(space)
    for values in itertools.product(*(space[k] for k in keys)):
        yield dict(zip(keys, values, strict=True))


def _fit_logreg(
    x_tr: np.ndarray, y_tr: np.ndarray, params: dict[str, Any], seed: int
) -> tuple[LogisticRegression, StandardScaler]:
    scaler = StandardScaler().fit(x_tr)
    model = LogisticRegression(max_iter=5000, random_state=seed, **params)
    model.fit(scaler.transform(x_tr), y_tr)
    return model, scaler


def _evaluate_logreg(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    params: dict[str, Any],
    seed: int,
) -> np.ndarray:
    model, scaler = _fit_logreg(x_tr, y_tr, params, seed)
    return np.asarray(model.predict_proba(scaler.transform(x_va))[:, 1])


def _fit_xgb(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    params: dict[str, Any],
    seed: int,
) -> XGBClassifier:
    model = XGBClassifier(random_state=seed, **XGB_FIXED, **params)
    model.fit(x_tr, y_tr, eval_set=[(x_val, y_val)], verbose=False)
    return model


def _select_and_predict(
    model_kind: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any], float]:
    """Select hyperparameters inside the training fold, then predict.

    The inner split is carved from ``x_train`` only. Validation rows never
    influence selection, fitting, imputation or scaling.
    """
    idx_tr, idx_in = train_test_split(
        np.arange(len(y_train)),
        test_size=INNER_VALIDATION_FRACTION,
        random_state=seed,
        stratify=y_train,
    )
    x_inner_tr, y_inner_tr = x_train[idx_tr], y_train[idx_tr]
    x_inner_va, y_inner_va = x_train[idx_in], y_train[idx_in]

    best_params: dict[str, Any] = {}
    best_score = -np.inf
    space = LR_GRID if model_kind == "logreg" else XGB_GRID

    for params in _grid(space):
        if model_kind == "logreg":
            p_in = _evaluate_logreg(x_inner_tr, y_inner_tr, x_inner_va, params, seed)
        else:
            model = _fit_xgb(x_inner_tr, y_inner_tr, x_inner_va, y_inner_va, params, seed)
            p_in = np.asarray(model.predict_proba(x_inner_va)[:, 1])
        score = float(METRIC_FUNCTIONS["auroc"](y_inner_va.astype(float), p_in))
        if score > best_score:
            best_score, best_params = score, params

    if model_kind == "logreg":
        preds = _evaluate_logreg(x_train, y_train, x_valid, best_params, seed)
        chosen = dict(best_params)
    else:
        # Refit on the full outer-training fold. Early stopping still needs a
        # validation signal, so the inner split is reused for that purpose only.
        model = _fit_xgb(
            x_train[idx_tr], y_train[idx_tr], x_inner_va, y_inner_va, best_params, seed
        )
        preds = np.asarray(model.predict_proba(x_valid)[:, 1])
        chosen = {**best_params, "best_iteration": int(model.best_iteration or 0)}
    return preds, chosen, best_score


def run_one(
    representation: Representation,
    model_kind: str,
    x_full: np.ndarray,
    y: np.ndarray,
    splits: list,
    *,
    seed: int,
    imputation: ImputationStrategy,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Produce out-of-fold predictions for one (representation, model) pair."""
    oof = np.full(len(y), np.nan)
    per_fold: list[dict[str, Any]] = []

    for split in splits:
        tr, va = split.train, split.validation
        imputer = FittedImputer.fit(x_full[tr], strategy=imputation, seed=seed + split.fold)
        x_tr = imputer.transform(x_full[tr], draw_seed=split.fold)
        x_va = imputer.transform(x_full[va], draw_seed=1000 + split.fold)

        preds, params, inner_score = _select_and_predict(
            model_kind, x_tr, y[tr], x_va, seed + split.fold
        )
        oof[va] = preds
        per_fold.append(
            {
                "fold": split.fold,
                "n_train": len(tr),
                "n_validation": len(va),
                "selected": params,
                SELECTION_METRIC: inner_score,
            }
        )

    if not np.isfinite(oof).all():
        raise RuntimeError("some patients received no out-of-fold prediction")
    return oof, per_fold


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=int, default=24)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("experiments/baselines/results/m2"),
    )
    args = parser.parse_args(argv)

    # sets defaults to ('a','b'); set-c requires allow_final_holdout and is never
    # requested here.
    cohort = development_cohort(load_cohort())
    truncated = cohort.truncate(args.cutoff)
    y = cohort.labels["mortality"].astype(np.float64)
    splits = stratified_folds(cohort, n_folds=args.folds, seed=args.seed)

    log.info("cohort", cutoff=args.cutoff, n=cohort.n_patients, prevalence=float(y.mean()))

    config_payload = {
        "cutoff": args.cutoff,
        "folds": args.folds,
        "seed": args.seed,
        "lr_grid": LR_GRID,
        "xgb_grid": XGB_GRID,
        "xgb_fixed": XGB_FIXED,
        "inner_validation_fraction": INNER_VALIDATION_FRACTION,
    }
    provenance = build_provenance(
        cohort=truncated,
        splits=splits,
        config_payload=config_payload,
        extra={"task": "T1_in_hospital_mortality", "excluded_sets": ["c"]},
    )

    plan: list[tuple[Representation, str, ImputationStrategy]] = []
    for rep in (
        *CORE_REPRESENTATIONS,
        Representation.STATICS_ONLY,
        Representation.VALUES_MASK_STATICS,
    ):
        for model_kind in ("logreg", "xgboost"):
            plan.append((rep, model_kind, ImputationStrategy.MEDIAN))
    # Residual-missingness sensitivity for values_only.
    plan.append((Representation.VALUES_ONLY, "logreg", ImputationStrategy.STOCHASTIC))
    plan.append((Representation.VALUES_ONLY, "xgboost", ImputationStrategy.STOCHASTIC))

    artifacts: list[RunArtifact] = []

    # Prevalence floor: the training-fold positive rate, predicted for everyone.
    prevalence_oof = np.full(len(y), np.nan)
    for split in splits:
        prevalence_oof[split.validation] = float(y[split.train].mean())
    artifacts.append(
        _artifact(
            "prevalence",
            "prevalence",
            "constant",
            prevalence_oof,
            y,
            cohort,
            args,
            provenance,
            n_features=0,
            names=[],
            hyperparameters={},
            search_space={},
            per_fold=[],
        )
    )
    log.info("done", model="prevalence", auroc=0.5)

    for rep, model_kind, imputation in plan:
        view = build_representation(truncated, rep)
        if rep in (Representation.VALUES_ONLY,) and view.contains_presence_features():
            raise RuntimeError(
                "values_only representation contains explicit presence features"
            )
        suffix = "" if imputation is ImputationStrategy.MEDIAN else "_stochastic"
        run_id = f"{rep}{suffix}::{model_kind}"

        t0 = time.perf_counter()
        oof, per_fold = run_one(
            rep, model_kind, view.x, y, splits, seed=args.seed, imputation=imputation
        )
        elapsed = time.perf_counter() - t0

        artifacts.append(
            _artifact(
                run_id,
                f"{rep}{suffix}",
                model_kind,
                oof,
                y,
                cohort,
                args,
                provenance,
                n_features=view.n_features,
                names=list(view.names),
                hyperparameters={"per_fold": per_fold},
                search_space=LR_GRID if model_kind == "logreg" else XGB_GRID,
                per_fold=per_fold,
                extra_provenance={"imputation": str(imputation), "fit_seconds": elapsed},
            )
        )
        log.info(
            "done",
            run=run_id,
            auroc=round(artifacts[-1].metrics["auroc"], 4),
            n_features=view.n_features,
            seconds=round(elapsed, 1),
        )

    manifest_path, predictions_path = write_run(artifacts, args.out)
    _report(artifacts, y, args)
    print(f"\nwrote {manifest_path}\nwrote {predictions_path}")
    return 0


def _artifact(
    run_id: str,
    representation: str,
    model: str,
    oof: np.ndarray,
    y: np.ndarray,
    cohort: Any,
    args: argparse.Namespace,
    provenance: dict[str, Any],
    *,
    n_features: int,
    names: list[str],
    hyperparameters: dict[str, Any],
    search_space: dict[str, Any],
    per_fold: list[dict[str, Any]],
    extra_provenance: dict[str, Any] | None = None,
) -> RunArtifact:
    metrics = classification_metrics(y, oof)
    intervals = {
        name: bootstrap_metric(y, oof, fn, n_boot=args.n_boot, seed=args.seed).as_dict()
        for name, fn in METRIC_FUNCTIONS.items()
    }
    return RunArtifact(
        run_id=run_id,
        representation=representation,
        model=model,
        cutoff_hours=args.cutoff,
        seed=args.seed,
        n_features=n_features,
        feature_names=names,
        hyperparameters=hyperparameters,
        search_space=search_space,
        selection_metric=SELECTION_METRIC,
        per_fold_selection=per_fold,
        metrics=metrics.as_dict(),
        intervals=intervals,
        reliability=reliability_curve(y, oof),
        predictions=oof,
        labels=y,
        record_ids=cohort.record_ids,
        provenance={**provenance, **(extra_provenance or {})},
    )


def _report(artifacts: list[RunArtifact], y: np.ndarray, args: argparse.Namespace) -> None:
    by_id = {a.run_id: a for a in artifacts}
    print("\n" + "=" * 104)
    print(
        f"M2 — T1 IN-HOSPITAL MORTALITY AT {args.cutoff}h | n={len(y):,} | "
        f"prevalence={y.mean():.2%} | {args.folds}-fold CV | sets a+b (set-c locked)"
    )
    print("=" * 104)
    header = (
        f"{'run':<40}{'#feat':>7}{'AUROC':>9}{'95% CI':>20}{'AUPRC':>9}{'Brier':>9}{'NLL':>9}"
    )
    print(header)
    print("-" * 104)
    for a in artifacts:
        ci = a.intervals["auroc"]
        ci_text = f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"
        print(
            f"{a.run_id:<40}{a.n_features:>7}{a.metrics['auroc']:>9.4f}{ci_text:>20}"
            f"{a.metrics['auprc']:>9.4f}{a.metrics['brier']:>9.4f}{a.metrics['nll']:>9.4f}"
        )

    print("\n" + "=" * 104)
    print("CALIBRATION (slope 1.0 and intercept 0.0 are perfect)")
    print("=" * 104)
    print(f"{'run':<40}{'slope':>10}{'intercept':>12}")
    print("-" * 104)
    for a in artifacts:
        if a.model == "constant":
            continue
        print(
            f"{a.run_id:<40}{a.metrics['calibration_slope']:>10.3f}"
            f"{a.metrics['calibration_intercept']:>12.3f}"
        )

    print("\n" + "=" * 104)
    print("PAIRED DIFFERENCES on identical out-of-fold predictions (patient bootstrap)")
    print("=" * 104)
    pairs = [
        ("values_mask", "values_only", "VALUES+MASK - VALUES ONLY"),
        ("values_only", "mask_only", "VALUES ONLY - MASK ONLY"),
        ("values_mask", "mask_only", "VALUES+MASK - MASK ONLY"),
    ]
    for model_kind in ("logreg", "xgboost"):
        print(f"\n[{model_kind}]")
        for b_rep, a_rep, label in pairs:
            a_id, b_id = f"{a_rep}::{model_kind}", f"{b_rep}::{model_kind}"
            if a_id not in by_id or b_id not in by_id:
                continue
            for metric_name in ("auroc", "auprc"):
                diff = paired_bootstrap_difference(
                    y,
                    by_id[a_id].predictions,
                    by_id[b_id].predictions,
                    METRIC_FUNCTIONS[metric_name],
                    metric_name=metric_name,
                    name_a=a_id,
                    name_b=b_id,
                    n_boot=args.n_boot,
                    seed=args.seed,
                )
                flag = "*" if diff.excludes_zero else " "
                print(
                    f"  {flag} {label:<28} {metric_name:<6} "
                    f"{diff.difference:+.4f} [{diff.low:+.4f}, {diff.high:+.4f}]"
                )

    print("\n" + "=" * 104)
    print("RESIDUAL-MISSINGNESS CONTROL — values_only(median) vs values_only(stochastic)")
    print("=" * 104)
    print(
        "Median imputation writes the training median into every unmeasured cell, so a model\n"
        "can detect 'exactly the median' and recover the missingness indicator. Stochastic\n"
        "imputation samples from the training marginal, removing that signature. The gap is\n"
        "an estimate of how much values-only performance is recoverable missingness\n"
        "information rather than physiology."
    )
    for model_kind in ("logreg", "xgboost"):
        a_id = f"values_only_stochastic::{model_kind}"
        b_id = f"values_only::{model_kind}"
        if a_id not in by_id or b_id not in by_id:
            continue
        for metric_name in ("auroc", "auprc"):
            diff = paired_bootstrap_difference(
                y,
                by_id[a_id].predictions,
                by_id[b_id].predictions,
                METRIC_FUNCTIONS[metric_name],
                metric_name=metric_name,
                name_a=a_id,
                name_b=b_id,
                n_boot=args.n_boot,
                seed=args.seed,
            )
            flag = "*" if diff.excludes_zero else " "
            print(
                f"  {flag} [{model_kind}] median - stochastic  {metric_name:<6} "
                f"{diff.difference:+.4f} [{diff.low:+.4f}, {diff.high:+.4f}]"
            )

    print("\n  * = 95% paired interval excludes zero.")


if __name__ == "__main__":
    sys.exit(main())
