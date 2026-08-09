"""Diagnose how recoverable original value-summary missingness is after imputation.

This is a representation diagnostic, not a mortality model and not a claim of
novel imputation leakage. For each source variable, a small gradient-boosted
classifier predicts whether its five value summaries were originally absent,
using only those summaries after fold-honest imputation. Outer folds are grouped
by patient; no evaluation row enters an imputer or reconstructibility classifier.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from cliniverse.data import load_cohort
from cliniverse.data.splits import development_cohort, stratified_folds
from cliniverse.evaluation.artifacts import build_provenance
from cliniverse.evaluation.representations import (
    FittedImputer,
    ImputationStrategy,
    Representation,
    build_representation,
)

STATISTICS = ("last", "mean", "min", "max", "slope")
STRATEGIES = (
    ImputationStrategy.MEDIAN,
    ImputationStrategy.MEDIAN_JITTER,
    ImputationStrategy.EMPIRICAL_MARGINAL,
)


def _indices(names: tuple[str, ...]) -> tuple[tuple[str, ...], dict[str, list[int]]]:
    by_stat: dict[str, dict[str, int]] = {stat: {} for stat in STATISTICS}
    for i, name in enumerate(names):
        statistic, variable = name.split("::", maxsplit=1)
        if statistic in by_stat:
            by_stat[statistic][variable] = i
    variables = tuple(sorted(by_stat["last"]))
    return variables, {
        variable: [by_stat[stat][variable] for stat in STATISTICS] for variable in variables
    }


def _coherence_violation(x: np.ndarray, missing: np.ndarray, groups: list[list[int]]) -> float:
    violations: list[np.ndarray] = []
    for j, cols in enumerate(groups):
        last, mean, minimum, maximum, _ = (x[:, col] for col in cols)
        invalid = (minimum > mean) | (mean > maximum) | (last < minimum) | (last > maximum)
        violations.append(invalid[missing[:, j]])
    pooled = np.concatenate(violations)
    return float(pooled.mean()) if pooled.size else float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=int, default=24)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("experiments/baselines/results/m2/imputation_diagnostics.json"),
    )
    args = parser.parse_args(argv)

    cohort = development_cohort(load_cohort()).truncate(args.cutoff)
    splits = stratified_folds(cohort, n_folds=args.folds, seed=args.seed)
    view = build_representation(cohort, Representation.VALUES_ONLY)
    variables, by_variable = _indices(view.names)
    groups = [by_variable[variable] for variable in variables]
    originally_missing = ~np.isfinite(view.x[:, [cols[0] for cols in groups]])

    results: dict[str, dict[str, Any]] = {}
    for strategy in STRATEGIES:
        oof = np.full(originally_missing.shape, np.nan, dtype=np.float64)
        coherence: list[float] = []
        for split in splits:
            imputer = FittedImputer.fit(
                view.x[split.train], strategy=strategy, seed=args.seed + split.fold
            )
            x_train = imputer.transform(view.x[split.train], draw_seed=30)
            x_test = imputer.transform(view.x[split.validation], draw_seed=31)
            coherence.append(
                _coherence_violation(
                    x_test,
                    originally_missing[split.validation],
                    groups,
                )
            )
            for j, cols in enumerate(groups):
                target = originally_missing[split.train, j].astype(int)
                if np.unique(target).size < 2:
                    oof[split.validation, j] = float(target[0])
                    continue
                model = HistGradientBoostingClassifier(
                    max_iter=40,
                    max_leaf_nodes=15,
                    min_samples_leaf=30,
                    l2_regularization=1.0,
                    random_state=args.seed + split.fold,
                )
                model.fit(x_train[:, cols], target)
                oof[split.validation, j] = model.predict_proba(x_test[:, cols])[:, 1]

        target_flat = originally_missing.ravel().astype(int)
        prediction_flat = oof.ravel()
        if not np.isfinite(prediction_flat).all():
            raise RuntimeError(f"{strategy}: missing out-of-fold diagnostic predictions")
        per_variable_auroc = [
            float(roc_auc_score(originally_missing[:, j], oof[:, j]))
            for j in range(len(variables))
            if np.unique(originally_missing[:, j]).size == 2
        ]
        results[str(strategy)] = {
            "micro_auroc": float(roc_auc_score(target_flat, prediction_flat)),
            "micro_average_precision": float(
                average_precision_score(target_flat, prediction_flat)
            ),
            "median_variable_auroc": float(np.median(per_variable_auroc)),
            "min_variable_auroc": float(np.min(per_variable_auroc)),
            "max_variable_auroc": float(np.max(per_variable_auroc)),
            "coherence_violation_rate_on_imputed_groups": float(np.mean(coherence)),
            "n_patient_variable_examples": int(target_flat.size),
            "missing_fraction": float(target_flat.mean()),
        }

    provenance = build_provenance(
        cohort=cohort,
        splits=splits,
        config_payload={
            "cutoff": args.cutoff,
            "folds": args.folds,
            "seed": args.seed,
            "classifier": "HistGradientBoostingClassifier(max_iter=40,max_leaf_nodes=15)",
            "strategies": [str(strategy) for strategy in STRATEGIES],
        },
        extra={"diagnostic": "value_summary_missingness_reconstructibility"},
    )
    payload = {
        "schema": "cliniverse.m2.imputation_diagnostics/1",
        "provenance": provenance,
        "statistics": list(STATISTICS),
        "variables": list(variables),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
