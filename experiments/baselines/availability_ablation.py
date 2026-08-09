"""How much of ICU mortality prediction is carried by measurement-presence
patterns rather than measured values?

At a 24h decision point we fit the same models on three disjoint feature views:

  AVAILABILITY  measurement-presence patterns only: per-variable counts,
                ever-measured flags and recency. No measured value is included.
  VALUES        measured values only (last/mean/min/max/slope), imputed
                within-fold, with no presence or mask information.
  STATICS       admission descriptors.

Measurement presence reflects the historical care process — which observations
were recorded, how often, how recently — rather than physiology alone. This is
an associational statement about what a model can predict from presence
patterns; it asserts nothing causal, and nothing about clinician intent.

It matters for acquisition evaluation because the acquirable set in a
retrospective replay is derived from those same historical records.

Everything is fold-honest: imputation and scaling are fit on training rows only.

Usage:
    python experiments/baselines/availability_ablation.py [--cutoff 24] [--folds 5]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections.abc import Callable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cliniverse.data import load_cohort
from cliniverse.data.splits import development_cohort, stratified_folds
from cliniverse.encoders import FeatureBlock, build_features
from cliniverse.log import get_logger

log = get_logger(__name__)

VIEWS: dict[str, tuple[FeatureBlock, ...]] = {
    "availability_only": (FeatureBlock.AVAILABILITY,),
    "values_only": (FeatureBlock.VALUES,),
    "statics_only": (FeatureBlock.STATICS,),
    "availability+statics": (FeatureBlock.AVAILABILITY, FeatureBlock.STATICS),
    "values+statics": (FeatureBlock.VALUES, FeatureBlock.STATICS),
    "all": (FeatureBlock.AVAILABILITY, FeatureBlock.VALUES, FeatureBlock.STATICS),
}


def _model(kind: str, seed: int) -> Pipeline:
    if kind == "logreg":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=seed)),
            ]
        )
    # HistGradientBoosting handles NaN natively, but we impute anyway so that
    # every view sees an identical preprocessing path.
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=0.06,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=seed,
                ),
            ),
        ]
    )


def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    point = float(fn(y, p))
    stats = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        stats.append(fn(y[idx], p[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5]) if stats else (np.nan, np.nan)
    return point, float(lo), float(hi)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=int, default=24, help="decision point, hours")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--sets", nargs="+", default=["a", "b"])
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("experiments/baselines/results/availability_ablation.json"),
    )
    args = parser.parse_args(argv)

    cohort = development_cohort(load_cohort(sets=tuple(args.sets)))
    truncated = cohort.truncate(args.cutoff)
    log.info("cohort", cutoff=args.cutoff, **truncated.describe())

    features = build_features(truncated)
    y = cohort.labels["mortality"].astype(int)
    log.info("features", n_features=features.n_features, prevalence=float(y.mean()))

    folds = stratified_folds(cohort, n_folds=args.folds, seed=args.seed)
    results: dict[str, dict[str, object]] = {}

    for view_name, blocks in VIEWS.items():
        view = features.subset(*blocks)
        for model_kind in ("logreg", "gbdt"):
            key = f"{view_name}::{model_kind}"
            oof = np.full(len(y), np.nan)
            t0 = time.perf_counter()
            for split in folds:
                model = _model(model_kind, args.seed + split.fold)
                model.fit(view.x[split.train], y[split.train])
                oof[split.validation] = model.predict_proba(view.x[split.validation])[:, 1]
            elapsed = time.perf_counter() - t0

            assert np.isfinite(oof).all(), "some patients received no out-of-fold prediction"
            auroc, auroc_lo, auroc_hi = bootstrap_ci(
                y, oof, roc_auc_score, args.n_boot, args.seed
            )
            auprc, auprc_lo, auprc_hi = bootstrap_ci(
                y, oof, average_precision_score, args.n_boot, args.seed
            )
            brier = float(brier_score_loss(y, oof))

            results[key] = {
                "view": view_name,
                "model": model_kind,
                "n_features": view.n_features,
                "auroc": auroc,
                "auroc_ci": [auroc_lo, auroc_hi],
                "auprc": auprc,
                "auprc_ci": [auprc_lo, auprc_hi],
                "brier": brier,
                "fit_seconds": round(elapsed, 1),
                "oof": oof.tolist(),
            }
            log.info(
                "done",
                view=view_name,
                model=model_kind,
                auroc=round(auroc, 4),
                auprc=round(auprc, 4),
                n_features=view.n_features,
            )

    print("\n" + "=" * 96)
    print(
        f"IN-HOSPITAL MORTALITY AT {args.cutoff}h  |  n={len(y):,}  "
        f"prevalence={y.mean():.2%}  |  {args.folds}-fold CV, sets {args.sets}"
    )
    print("=" * 96)
    print(
        f"{'feature view':<24}{'model':<9}{'#feat':>7}{'AUROC':>9}{'95% CI':>18}"
        f"{'AUPRC':>9}{'Brier':>9}"
    )
    print("-" * 96)
    for r in results.values():
        ci = f"[{r['auroc_ci'][0]:.3f},{r['auroc_ci'][1]:.3f}]"  # type: ignore[index]
        print(
            f"{r['view']:<24}{r['model']:<9}{r['n_features']:>7}"
            f"{r['auroc']:>9.4f}{ci:>18}{r['auprc']:>9.4f}{r['brier']:>9.4f}"
        )

    # Raw comparison only. We deliberately do NOT print a "share of full-model
    # skill" ratio: it depends on an arbitrary chance-normalisation and on an
    # untuned full model, and the comparable claim needs a properly tuned
    # full-value baseline on the identical cohort/split/protocol (M2).
    for model_kind in ("logreg", "gbdt"):
        a = results[f"availability_only::{model_kind}"]
        v = results[f"values_only::{model_kind}"]
        gap = float(v["auroc"]) - float(a["auroc"])  # type: ignore[arg-type]
        print(
            f"\n[{model_kind}] availability-only AUROC = {float(a['auroc']):.4f} "  # type: ignore[arg-type]
            f"(measurement-presence patterns only, no measured values); "
            f"values-only = {float(v['auroc']):.4f}; difference {gap:+.4f}."  # type: ignore[arg-type]
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cutoff_hours": args.cutoff,
        "sets": args.sets,
        "n_patients": len(y),
        "prevalence": float(y.mean()),
        "n_folds": args.folds,
        "seed": args.seed,
        "results": {
            k: {kk: vv for kk, vv in r.items() if kk != "oof"} for k, r in results.items()
        },
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.out.with_suffix(".oof.npz"),
        y=y,
        **{k.replace("::", "__"): np.asarray(r["oof"]) for k, r in results.items()},
    )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
