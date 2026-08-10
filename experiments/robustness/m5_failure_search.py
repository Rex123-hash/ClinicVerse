"""M5: search for withheld-information configurations that break probability
reliability while discrimination stays close to its clean value.

Design predeclared in `docs/M5_DESIGN.md` and committed before this ran.

The model, imputer and calibrator are frozen from M2/M3/M4 and fitted only on
clean training partitions; nothing is refitted under any configuration. Loss is
applied to the truncated cohort before feature construction. set-c is never
loaded.

Usage:
    python experiments/robustness/m5_failure_search.py
    python experiments/robustness/m5_failure_search.py --limit 40   # smoke run
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import time
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from cliniverse.acquisition import load_panel_catalogue
from cliniverse.data import load_cohort
from cliniverse.data.cohort import Cohort
from cliniverse.data.splits import development_cohort, stratified_folds
from cliniverse.evaluation.artifacts import build_provenance
from cliniverse.evaluation.calibration import CalibratorKind, build_calibrator
from cliniverse.evaluation.failure_search import (
    ConfigurationScore,
    GroupSubset,
    apply_group_subset_loss,
    control_seed,
    enumerate_group_subsets,
    holm_bonferroni,
    matched_random_control,
    select_top_k,
    spearman_permutation_test,
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
    per_patient_squared_error,
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

# ------------------------------------------- frozen from M2 / M3 / M4 -------
# Deliberately a copy of the frozen pipeline rather than an import across
# experiment scripts. `tests/test_failure_search.py` pins these constants and the
# resulting per-fold split-feature counts to the values M4 recorded, so a silent
# divergence from the frozen contract fails a test rather than a review.
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
CALIBRATION_FRACTION = 0.25
CUTOFF = 24

# ------------------------------------------- predeclared in M5_DESIGN.md ----
DISCOVERY_FOLDS: tuple[int, ...] = (0, 1, 2)
CONFIRMATION_FOLDS: tuple[int, ...] = (3, 4)
AUROC_DELTA = 0.02
TOP_K = 5
CONTROL_REPEATS = 3


@dataclasses.dataclass(slots=True)
class FoldModel:
    """Frozen per-fold pipeline: imputer, model, clean-fitted calibrator."""

    fold: int
    test_index: np.ndarray
    imputer: FittedImputer
    model: XGBClassifier
    calibrator: Any

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = self.imputer.transform(features, draw_seed=self.fold)
        raw = np.asarray(self.model.predict_proba(x)[:, 1], dtype=np.float64)
        return np.asarray(self.calibrator.transform(raw), dtype=np.float64)

    @property
    def n_features_used(self) -> int:
        """How many features the fitted booster actually splits on.

        Zero means every tree is a stump and the model is a constant, which would
        make every configuration score identically and look exactly like a
        propagation bug. The guard raises instead.
        """
        trees = self.model.get_booster().trees_to_dataframe()
        return len(set(trees["Feature"]) - {"Leaf"})


def fit_folds(cohort: Cohort, y: np.ndarray, seed: int, folds: int) -> list[FoldModel]:
    """Fit the frozen pipeline per fold on CLEAN data only."""
    truncated = cohort.truncate(CUTOFF)
    clean = build_representation(truncated, REPRESENTATION).x
    out: list[FoldModel] = []
    for split in stratified_folds(cohort, n_folds=folds, seed=seed):
        train_idx, calib_idx = train_test_split(
            split.train,
            test_size=CALIBRATION_FRACTION,
            random_state=seed + split.fold,
            stratify=y[split.train],
        )
        imputer = FittedImputer.fit(
            clean[train_idx], strategy=ImputationStrategy.MEDIAN, seed=seed + split.fold
        )
        model = XGBClassifier(random_state=seed + split.fold, **XGB_PARAMS)
        model.fit(imputer.transform(clean[train_idx], draw_seed=split.fold), y[train_idx])

        raw_calib = model.predict_proba(
            imputer.transform(clean[calib_idx], draw_seed=100 + split.fold)
        )[:, 1]
        calibrator = build_calibrator(CalibratorKind.PLATT)
        calibrator.fit(np.asarray(raw_calib, dtype=np.float64), y[calib_idx])

        out.append(
            FoldModel(
                fold=split.fold,
                test_index=split.validation,
                imputer=imputer,
                model=model,
                calibrator=calibrator,
            )
        )
        used = out[-1].n_features_used
        if used == 0:
            raise ConfigError(
                f"fold {split.fold}: the fitted model splits on zero features, so it "
                "is a constant and no withheld information can change its prediction. "
                "Increase the cohort rather than changing the frozen hyperparameters."
            )
        log.info(
            "fold fitted",
            fold=split.fold,
            n_model_train=len(train_idx),
            n_calibration=len(calib_idx),
            n_test=len(split.validation),
            n_features_used=used,
        )
    return out


def predict_all(features: np.ndarray, fold_models: list[FoldModel]) -> np.ndarray:
    """Score every patient with its own fold's frozen pipeline."""
    out = np.full(features.shape[0], np.nan, dtype=np.float64)
    for fm in fold_models:
        out[fm.test_index] = fm.predict(features[fm.test_index])
    if not np.isfinite(out).all():
        raise RuntimeError("some patient was not scored by any fold")
    return out


def patient_set(fold_models: list[FoldModel], folds: tuple[int, ...]) -> np.ndarray:
    """Outer-test patient indices for the named folds."""
    selected = [fm.test_index for fm in fold_models if fm.fold in folds]
    if len(selected) != len(folds):
        raise ConfigError(f"expected folds {folds}, found {[f.fold for f in fold_models]}")
    return np.sort(np.concatenate(selected)).astype(np.int64)


def _scores_on(
    index: int,
    subset: GroupSubset,
    where: np.ndarray,
    y: np.ndarray,
    p_subset: np.ndarray,
    p_control: np.ndarray,
    control_loss_nll: np.ndarray,
    control_loss_brier: np.ndarray,
    removed_cells: np.ndarray,
    severity: np.ndarray,
) -> ConfigurationScore:
    """Aggregate one configuration on one patient set."""
    y_w = y[where]
    p_w = p_subset[where]
    return ConfigurationScore(
        index=index,
        subset=subset,
        n_groups=len(subset),
        mean_removed_cells=float(removed_cells[where].mean()),
        mean_realized_severity=float(severity[where].mean()),
        delta_nll_excess=float(
            per_patient_log_loss(y_w, p_w).mean() - control_loss_nll[where].mean()
        ),
        delta_brier_excess=float(
            per_patient_squared_error(y_w, p_w).mean() - control_loss_brier[where].mean()
        ),
        nll=float(negative_log_likelihood(y_w, p_w)),
        nll_control=float(control_loss_nll[where].mean()),
        brier=float(brier_score(y_w, p_w)),
        brier_control=float(control_loss_brier[where].mean()),
        auroc=float(auroc(y_w, p_w)),
        auroc_control_last_draw=float(auroc(y_w, p_control[where])),
        auprc=float(auprc(y_w, p_w)),
        calibration_intercept=float(calibration_intercept(y_w, p_w)),
        calibration_slope=float(calibration_slope(y_w, p_w)),
        mean_predicted_risk=float(p_w.mean()),
    )


def evaluate_configuration(
    index: int,
    subset: GroupSubset,
    truncated: Cohort,
    y: np.ndarray,
    fold_models: list[FoldModel],
    catalogue: Any,
    discovery: np.ndarray,
    confirmation: np.ndarray,
    *,
    seed: int,
    repeats: int,
) -> tuple[ConfigurationScore, ConfigurationScore, dict[str, np.ndarray]]:
    """Score one configuration and its amount-matched control on both sets."""
    loss = apply_group_subset_loss(truncated, subset, catalogue)
    p_subset = predict_all(build_representation(loss.cohort, REPRESENTATION).x, fold_models)

    nll_draws = np.zeros((repeats, len(y)), dtype=np.float64)
    brier_draws = np.zeros((repeats, len(y)), dtype=np.float64)
    p_control_last = np.zeros(len(y), dtype=np.float64)
    for repetition in range(repeats):
        control = matched_random_control(
            truncated,
            loss.removed_cells,
            catalogue,
            seed=control_seed(seed, subset, repetition),
        )
        p_control = predict_all(build_representation(control, REPRESENTATION).x, fold_models)
        nll_draws[repetition] = per_patient_log_loss(y, p_control)
        brier_draws[repetition] = per_patient_squared_error(y, p_control)
        p_control_last = p_control

    control_nll = nll_draws.mean(axis=0)
    control_brier = brier_draws.mean(axis=0)
    severity = loss.realized_severity

    common = (
        y,
        p_subset,
        p_control_last,
        control_nll,
        control_brier,
        loss.removed_cells,
        severity,
    )
    return (
        _scores_on(index, subset, discovery, *common),
        _scores_on(index, subset, confirmation, *common),
        {
            "p_subset": p_subset,
            "per_patient_nll_excess": per_patient_log_loss(y, p_subset) - control_nll,
            "per_patient_brier_excess": per_patient_squared_error(y, p_subset) - control_brier,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--control-repeats", type=int, default=CONTROL_REPEATS)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--delta", type=float, default=AUROC_DELTA)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="evaluate only the first N configurations (smoke runs only; a "
        "limited run is not a predeclared result and is flagged as such)",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("experiments/robustness/results/m5"),
    )
    args = parser.parse_args(argv)

    catalogue = load_panel_catalogue()
    cohort = development_cohort(load_cohort())
    y = cohort.labels["mortality"].astype(np.float64)
    truncated = cohort.truncate(CUTOFF)
    log.info("cohort", n=cohort.n_patients, prevalence=float(y.mean()))

    fold_models = fit_folds(cohort, y, args.seed, args.folds)
    discovery = patient_set(fold_models, DISCOVERY_FOLDS)
    confirmation = patient_set(fold_models, CONFIRMATION_FOLDS)
    if np.intersect1d(discovery, confirmation).size:
        raise ConfigError("discovery and confirmation patient sets overlap")
    log.info("patient sets", n_discovery=len(discovery), n_confirmation=len(confirmation))

    # ------------------------------------------------- clean reference ------
    p_clean = predict_all(build_representation(truncated, REPRESENTATION).x, fold_models)
    clean = {
        name: {
            "n_patients": len(where),
            "prevalence": float(y[where].mean()),
            "nll": float(negative_log_likelihood(y[where], p_clean[where])),
            "brier": float(brier_score(y[where], p_clean[where])),
            "auroc": float(auroc(y[where], p_clean[where])),
            "auprc": float(auprc(y[where], p_clean[where])),
            "calibration_intercept": float(calibration_intercept(y[where], p_clean[where])),
            "calibration_slope": float(calibration_slope(y[where], p_clean[where])),
            "mean_predicted_risk": float(p_clean[where].mean()),
        }
        for name, where in (("discovery", discovery), ("confirmation", confirmation))
    }
    log.info(
        "clean reference",
        discovery_auroc=round(clean["discovery"]["auroc"], 4),
        confirmation_auroc=round(clean["confirmation"]["auroc"], 4),
    )

    # ------------------------------------------------- pass 1: enumerate ----
    subsets = enumerate_group_subsets(catalogue.panel_names)
    if args.limit:
        subsets = subsets[: args.limit]
    log.info("enumerating configurations", n=len(subsets))

    discovery_scores: list[ConfigurationScore] = []
    confirmation_scores: list[ConfigurationScore] = []
    t0 = time.perf_counter()
    for index, subset in enumerate(subsets):
        d_score, c_score, _ = evaluate_configuration(
            index,
            subset,
            truncated,
            y,
            fold_models,
            catalogue,
            discovery,
            confirmation,
            seed=args.seed,
            repeats=args.control_repeats,
        )
        discovery_scores.append(d_score)
        confirmation_scores.append(c_score)
        if (index + 1) % 50 == 0 or index + 1 == len(subsets):
            elapsed = time.perf_counter() - t0
            log.info(
                "enumeration progress",
                done=index + 1,
                total=len(subsets),
                seconds=round(elapsed, 1),
                eta_seconds=round(elapsed / (index + 1) * (len(subsets) - index - 1), 1),
            )

    # ------------------------------------------------- the lock -------------
    # select_top_k accepts discovery scores only; it cannot see a confirmation
    # number even by accident. The locked list is written to the artifact before
    # any confirmatory statistic below is computed.
    locked = select_top_k(
        discovery_scores,
        k=args.top_k,
        clean_auroc=clean["discovery"]["auroc"],
        delta=args.delta,
    )
    log.info("locked selection", top_k=[list(s) for s in locked])

    args.out.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": "cliniverse.m5/1",
        "design_document": "docs/M5_DESIGN.md",
        "predeclared": {
            "discovery_folds": list(DISCOVERY_FOLDS),
            "confirmation_folds": list(CONFIRMATION_FOLDS),
            "auroc_delta": args.delta,
            "top_k": args.top_k,
            "control_repeats": args.control_repeats,
            "n_boot": args.n_boot,
            "n_permutations": args.n_permutations,
            "seed": args.seed,
            "primary_endpoint": "delta_nll_excess (subset minus amount-matched random)",
            "co_primary_endpoint": "delta_brier_excess",
        },
        "complete_enumeration": not args.limit,
        "n_configurations": len(subsets),
        "clean_reference": clean,
        "locked_selection": [list(s) for s in locked],
        "discovery": [s.as_dict() for s in discovery_scores],
        "confirmation": [s.as_dict() for s in confirmation_scores],
    }
    (args.out / "locked_selection.json").write_text(
        json.dumps(
            {
                "locked_selection": [list(s) for s in locked],
                "selected_on": "discovery folds 0-2 only",
                "auroc_delta": args.delta,
                "clean_discovery_auroc": clean["discovery"]["auroc"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
        newline="\n",
    )

    # ------------------------------------------------- pass 2: confirm ------
    by_subset = {s.subset: s for s in confirmation_scores}
    pass1_discovery = {s.subset: s for s in discovery_scores}
    confirmatory: list[dict[str, Any]] = []
    stored: dict[str, np.ndarray] = {"labels": y, "record_ids": cohort.record_ids}
    for rank, subset in enumerate(locked, start=1):
        index = pass1_discovery[subset].index
        _, c_score, arrays = evaluate_configuration(
            index,
            subset,
            truncated,
            y,
            fold_models,
            catalogue,
            discovery,
            confirmation,
            seed=args.seed,
            repeats=args.control_repeats,
        )
        # Determinism self-check: pass 2 must reproduce pass 1 exactly.
        if not np.isclose(
            c_score.delta_nll_excess, by_subset[subset].delta_nll_excess, rtol=0, atol=0
        ):
            raise RuntimeError(
                f"pass 2 did not reproduce pass 1 for {subset}: "
                f"{c_score.delta_nll_excess} != {by_subset[subset].delta_nll_excess}"
            )
        # Primary interval at 95%; supporting ranks at Bonferroni-adjusted 99%.
        percentiles = (2.5, 97.5) if rank == 1 else (0.5, 99.5)
        nll_interval = paired_mean_difference_bootstrap(
            y[confirmation],
            arrays["per_patient_nll_excess"][confirmation],
            metric_name="delta_nll_excess",
            name_a="amount_matched_random",
            name_b="+".join(subset),
            n_boot=args.n_boot,
            seed=args.seed,
            percentiles=percentiles,
        )
        brier_interval = paired_mean_difference_bootstrap(
            y[confirmation],
            arrays["per_patient_brier_excess"][confirmation],
            metric_name="delta_brier_excess",
            name_a="amount_matched_random",
            name_b="+".join(subset),
            n_boot=args.n_boot,
            seed=args.seed,
            percentiles=percentiles,
        )
        confirmatory.append(
            {
                "rank": rank,
                "subset": list(subset),
                "interval_level": "95%" if rank == 1 else "99% (Bonferroni 0.05/5)",
                "discovery_delta_nll_excess": pass1_discovery[subset].delta_nll_excess,
                "confirmation_delta_nll_excess": nll_interval.as_dict(),
                "confirmation_delta_brier_excess": brier_interval.as_dict(),
                "confirmation_auroc": c_score.auroc,
                "confirmation_auroc_drop_vs_clean": clean["confirmation"]["auroc"]
                - c_score.auroc,
                "auroc_constraint_holds": bool(
                    clean["confirmation"]["auroc"] - c_score.auroc <= args.delta
                ),
                "confirmation_calibration_intercept": c_score.calibration_intercept,
                "confirmation_calibration_slope": c_score.calibration_slope,
                "confirmation_mean_predicted_risk": c_score.mean_predicted_risk,
                "confirmation_mean_realized_severity": c_score.mean_realized_severity,
                "confirmation_mean_removed_cells": c_score.mean_removed_cells,
            }
        )
        stored[f"p_subset|{'+'.join(subset)}"] = arrays["p_subset"]
        stored[f"nll_excess|{'+'.join(subset)}"] = arrays["per_patient_nll_excess"]
        stored[f"brier_excess|{'+'.join(subset)}"] = arrays["per_patient_brier_excess"]

    stored["p_clean"] = p_clean
    stored["discovery_index"] = discovery
    stored["confirmation_index"] = confirmation

    # ------------------------------------------------- T4 generalization ----
    generalization = spearman_permutation_test(
        np.array([s.delta_nll_excess for s in discovery_scores]),
        np.array([s.delta_nll_excess for s in confirmation_scores]),
        n_permutations=args.n_permutations,
        seed=args.seed,
    )

    primary = confirmatory[0]
    t1_pass = bool(primary["confirmation_delta_nll_excess"]["excludes_zero"]) and (
        primary["confirmation_delta_nll_excess"]["difference"] > 0
    )
    t2_p = float(primary["confirmation_delta_brier_excess"]["p_value"])
    t4_p = float(generalization["p_value_one_sided"])
    secondary_rejected = holm_bonferroni([t2_p, t4_p], alpha=0.05)
    payload["tests"] = {
        "T1_primary_confirmation_delta_nll_excess": {
            "passes": t1_pass,
            **primary["confirmation_delta_nll_excess"],
        },
        "T2_confirmation_delta_brier_excess": {
            "p_value": t2_p,
            "passes_after_holm": bool(secondary_rejected[0])
            and primary["confirmation_delta_brier_excess"]["difference"] > 0,
        },
        "T3_auroc_constraint": {
            "clean_confirmation_auroc": clean["confirmation"]["auroc"],
            "configuration_auroc": primary["confirmation_auroc"],
            "drop": primary["confirmation_auroc_drop_vs_clean"],
            "delta": args.delta,
            "passes": bool(primary["auroc_constraint_holds"]),
        },
        "T4_generalization_spearman": {
            **generalization,
            "passes_after_holm": bool(secondary_rejected[1])
            and generalization["spearman_rho"] > 0,
        },
        "holm_family": ["T2", "T4"],
    }
    payload["confirmatory"] = confirmatory
    payload["provenance"] = build_provenance(
        cohort=truncated,
        splits=stratified_folds(cohort, n_folds=args.folds, seed=args.seed),
        config_payload={
            "xgb": XGB_PARAMS,
            "cutoff": CUTOFF,
            "seed": args.seed,
            "discovery_folds": DISCOVERY_FOLDS,
            "confirmation_folds": CONFIRMATION_FOLDS,
            "auroc_delta": args.delta,
            "top_k": args.top_k,
            "control_repeats": args.control_repeats,
        },
        extra={"excluded_sets": ["c"], "catalogue_version": catalogue.version},
    )

    # Keyword expansion is required: it is what names each array in the archive.
    # numpy's stub types **kwargs as `bool` (for `allow_pickle`), so the correct
    # call does not type-check; see the same note in evaluation/artifacts.py.
    np.savez_compressed(args.out / "m5_predictions.npz", **stored)  # type: ignore[arg-type]
    (args.out / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
        newline="\n",
    )
    _print_report(payload)
    print(f"\nwrote {args.out / 'results.json'}")
    return 0


def _print_report(payload: dict[str, Any]) -> None:
    clean = payload["clean_reference"]
    print("\n" + "=" * 100)
    print("M5 — DISCRIMINATION-SILENT RELIABILITY FAILURE SEARCH")
    print("=" * 100)
    print(
        f"configurations enumerated: {payload['n_configurations']} "
        f"(complete={payload['complete_enumeration']})"
    )
    print(
        f"clean discovery   AUROC {clean['discovery']['auroc']:.4f}  "
        f"NLL {clean['discovery']['nll']:.4f}  n={clean['discovery']['n_patients']}"
    )
    print(
        f"clean confirmation AUROC {clean['confirmation']['auroc']:.4f}  "
        f"NLL {clean['confirmation']['nll']:.4f}  n={clean['confirmation']['n_patients']}"
    )
    print("\nLOCKED SELECTION (chosen on discovery folds 0-2 only):")
    for row in payload["confirmatory"]:
        nll = row["confirmation_delta_nll_excess"]
        print(
            f"  #{row['rank']} {'+'.join(row['subset'])}\n"
            f"      discovery excess NLL {row['discovery_delta_nll_excess']:+.5f}\n"
            f"      confirmation excess NLL {nll['difference']:+.5f} "
            f"[{nll['ci_low']:+.5f}, {nll['ci_high']:+.5f}] ({row['interval_level']})\n"
            f"      confirmation AUROC {row['confirmation_auroc']:.4f} "
            f"(drop {row['confirmation_auroc_drop_vs_clean']:+.4f}, "
            f"constraint {'OK' if row['auroc_constraint_holds'] else 'VIOLATED'})\n"
            f"      calibration intercept {row['confirmation_calibration_intercept']:+.3f}  "
            f"slope {row['confirmation_calibration_slope']:.3f}  "
            f"mean risk {row['confirmation_mean_predicted_risk']:.4f}  "
            f"severity {row['confirmation_mean_realized_severity']:.3f}"
        )
    tests = payload["tests"]
    t1 = tests["T1_primary_confirmation_delta_nll_excess"]
    t2 = tests["T2_confirmation_delta_brier_excess"]
    t3 = tests["T3_auroc_constraint"]
    t4 = tests["T4_generalization_spearman"]

    def verdict(passed: bool) -> str:
        return "PASS" if passed else "FAIL"

    print("\nPREDECLARED TESTS:")
    print(f"  T1 primary  : {verdict(t1['passes'])}")
    print(f"  T2 Brier    : {verdict(t2['passes_after_holm'])}  (p={t2['p_value']:.5f})")
    print(f"  T3 AUROC    : {verdict(t3['passes'])}  (drop {t3['drop']:+.4f})")
    print(
        f"  T4 transfer : {verdict(t4['passes_after_holm'])}"
        f"  (rho={t4['spearman_rho']:+.4f}, p={t4['p_value_one_sided']:.5f})"
    )


if __name__ == "__main__":
    sys.exit(main())
