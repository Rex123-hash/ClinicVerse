"""Compact decomposition of M2 mask-only discrimination.

The diagnostic separates ever-measured flags, measurement counts/frequency,
and recency/time-since-last while retaining the corrected M2 nested logistic
regression protocol. It is supplementary and does not expand the M2 headline
model family.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any

import numpy as np

from cliniverse.data import load_cohort
from cliniverse.data.splits import development_cohort, stratified_folds
from cliniverse.evaluation.artifacts import build_provenance
from cliniverse.evaluation.metrics import (
    METRIC_FUNCTIONS,
    bootstrap_metric,
    classification_metrics,
)
from cliniverse.evaluation.representations import (
    FittedImputer,
    ImputationStrategy,
    Representation,
    build_representation,
)
from experiments.baselines.m2_representation_ablation import LR_GRID, run_one

SUBSETS: dict[str, tuple[str, ...]] = {
    "ever_measured": ("ever::",),
    "counts_frequency": ("n_obs::", "n_distinct_vars::"),
    "recency_time_since_last": ("recency::",),
    "ever_plus_counts": ("ever::", "n_obs::", "n_distinct_vars::"),
    "full_mask": ("ever::", "n_obs::", "n_distinct_vars::", "recency::"),
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
        default=pathlib.Path("experiments/baselines/results/m2/mask_decomposition.json"),
    )
    args = parser.parse_args(argv)

    cohort = development_cohort(load_cohort()).truncate(args.cutoff)
    y = cohort.labels["mortality"].astype(np.float64)
    splits = stratified_folds(cohort, n_folds=args.folds, seed=args.seed)
    view = build_representation(cohort, Representation.MASK_ONLY)

    arrays: dict[str, np.ndarray] = {"labels": y, "record_ids": cohort.record_ids}
    results: dict[str, dict[str, Any]] = {}
    for subset, prefixes in SUBSETS.items():
        selected = [i for i, name in enumerate(view.names) if name.startswith(prefixes)]
        names = [view.names[i] for i in selected]
        oof, per_fold = run_one(
            Representation.MASK_ONLY,
            "logreg",
            view.x[:, selected],
            y,
            splits,
            seed=args.seed,
            imputation=ImputationStrategy.MEDIAN,
        )
        metrics = classification_metrics(y, oof).as_dict()
        intervals = {
            name: bootstrap_metric(y, oof, fn, n_boot=args.n_boot, seed=args.seed).as_dict()
            for name, fn in METRIC_FUNCTIONS.items()
        }
        arrays[f"pred__{subset}"] = oof
        results[subset] = {
            "n_features": len(selected),
            "feature_names": names,
            "metrics": metrics,
            "intervals": intervals,
            "per_fold_selection": per_fold,
        }

    predictions_path = args.out.with_name("mask_decomposition_predictions.npz")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    provenance = build_provenance(
        cohort=cohort,
        splits=splits,
        config_payload={
            "cutoff": args.cutoff,
            "folds": args.folds,
            "seed": args.seed,
            "n_boot": args.n_boot,
            "lr_grid": LR_GRID,
            "subsets": SUBSETS,
        },
        extra={
            "diagnostic": "mask_only_component_decomposition",
            "imputation": str(ImputationStrategy.MEDIAN),
            "preprocessing": FittedImputer.__name__,
        },
    )
    np.savez_compressed(predictions_path, **arrays)  # type: ignore[arg-type]
    payload = {
        "schema": "cliniverse.m2.mask_decomposition/1",
        "provenance": provenance,
        "predictions_file": {
            "name": predictions_path.name,
            "sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        },
        "results": results,
    }
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(json.dumps({k: v["metrics"] for k, v in results.items()}, indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {predictions_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
