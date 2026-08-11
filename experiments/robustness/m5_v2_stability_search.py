"""M5-v2: stability-aware search for a minimal discrimination-silent failure pattern.

Design predeclared in `docs/M5_V2_DESIGN.md` and committed at 5562120, before this
ran. M5-v1 is untouched and remains M5-C.

All 8,000 A+B patients are DEVELOPMENT. The 20 resplits reuse them and are not
independent, so nothing computed here is confirmatory: no interval, no p-value, no
effect claim. The run emits a frozen pattern, a frozen statistic, a decision rule
and a detectability verdict. set-c is never loaded.

Usage:
    python experiments/robustness/m5_v2_stability_search.py
    python experiments/robustness/m5_v2_stability_search.py --repeats 2 --limit 20
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
    AnalyteSubset,
    apply_analyte_subset_loss,
    control_seed,
    enumerate_group_subsets,
    fold_dispersion,
    matched_random_control,
    minimum_detectable_effect,
    select_one_se_parsimonious,
    selection_frequency,
)
from cliniverse.evaluation.metrics import (
    auroc,
    brier_score,
    calibration_intercept,
    calibration_slope,
    negative_log_likelihood,
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

# ------------------------------------------- frozen from M2 / M3 / M4 -------
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

# ---------------------------------------- predeclared in M5_V2_DESIGN.md ----
TARGET_GROUP = "BMP_like"
NULL_CONTROL_GROUPS: tuple[str, ...] = ("CBC_like", "ABG_like")
AUROC_DELTA = 0.02
CONTROL_REPEATS = 5
N_RESPLITS = 20
REFERENCE_RESPLIT = 0
MAJORITY = 11
SET_C_N = 4000


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
        """Features the fitted booster splits on. Zero means a constant model."""
        trees = self.model.get_booster().trees_to_dataframe()
        return len(set(trees["Feature"]) - {"Leaf"})


def fit_resplit(cohort: Cohort, y: np.ndarray, seed: int, folds: int) -> list[FoldModel]:
    """Fit the frozen pipeline specification for one resplit, on CLEAN data only.

    "Frozen" here means the *specification* — hyperparameters, representation and
    the three-way isolation — not the fitted objects, which must be refit per
    resplit for the split to be honest. `M5_V2_DESIGN.md` §10 records that.
    """
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
        fold_model = FoldModel(
            fold=split.fold,
            test_index=split.validation,
            imputer=imputer,
            model=model,
            calibrator=calibrator,
        )
        if fold_model.n_features_used == 0:
            raise ConfigError(
                f"resplit seed {seed} fold {split.fold}: the fitted model splits on zero "
                "features, so it is a constant and no withheld information can change "
                "its prediction. Increase the cohort rather than changing the frozen "
                "hyperparameters."
            )
        out.append(fold_model)
    return out


def predict_all(features: np.ndarray, fold_models: list[FoldModel]) -> np.ndarray:
    """Out-of-fold prediction for every patient, each by its own fold's pipeline."""
    out = np.full(features.shape[0], np.nan, dtype=np.float64)
    for fm in fold_models:
        out[fm.test_index] = fm.predict(features[fm.test_index])
    if not np.isfinite(out).all():
        raise RuntimeError("some patient was not scored by any fold")
    return out


def build_candidates(catalogue: Any) -> list[tuple[AnalyteSubset, str]]:
    """The 141 predeclared candidates, target and null-control regions in one pool."""
    candidates: list[tuple[AnalyteSubset, str]] = []
    for region in (TARGET_GROUP, *NULL_CONTROL_GROUPS):
        members = catalogue.panels[region].members
        tag = "target" if region == TARGET_GROUP else "null_control"
        candidates.extend((subset, tag) for subset in enumerate_group_subsets(members))
    return candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--repeats", type=int, default=N_RESPLITS)
    parser.add_argument("--control-repeats", type=int, default=CONTROL_REPEATS)
    parser.add_argument("--delta", type=float, default=AUROC_DELTA)
    parser.add_argument(
        "--limit", type=int, default=0, help="smoke runs only; flagged in the artifact"
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("experiments/robustness/results/m5v2"),
    )
    args = parser.parse_args(argv)

    catalogue = load_panel_catalogue()
    cohort = development_cohort(load_cohort())
    y = cohort.labels["mortality"].astype(np.float64)
    truncated = cohort.truncate(CUTOFF)
    log.info("cohort", n=cohort.n_patients, prevalence=float(y.mean()))

    # ------------------------------------------- resplit pipelines ---------
    t0 = time.perf_counter()
    resplits = [fit_resplit(cohort, y, args.seed + b, args.folds) for b in range(args.repeats)]
    log.info("resplits fitted", n=len(resplits), seconds=round(time.perf_counter() - t0, 1))

    clean_features = build_representation(truncated, REPRESENTATION).x
    clean_predictions = [predict_all(clean_features, fm) for fm in resplits]
    clean_auroc = [float(auroc(y, p)) for p in clean_predictions]
    log.info(
        "clean reference",
        reference_auroc=round(clean_auroc[REFERENCE_RESPLIT], 4),
        reference_nll=round(float(negative_log_likelihood(y, clean_predictions[0])), 4),
    )

    candidates = build_candidates(catalogue)
    if args.limit:
        candidates = candidates[: args.limit]
    log.info("candidates", n=len(candidates))

    # ------------------------------------------- score every candidate -----
    # Withholding and control draws are deterministic and cohort-fixed, so each
    # candidate's six feature matrices are built once and reused across all
    # resplits; only prediction varies.
    n_patients = cohort.n_patients
    deltas = np.zeros((len(candidates), args.repeats, args.folds), dtype=np.float64)
    cand_auroc = np.zeros((len(candidates), args.repeats), dtype=np.float64)
    reference_d: dict[AnalyteSubset, np.ndarray] = {}
    reference_rows: list[dict[str, Any]] = []

    t0 = time.perf_counter()
    for ci, (pattern, region) in enumerate(candidates):
        loss = apply_analyte_subset_loss(truncated, pattern, catalogue)
        subset_features = build_representation(loss.cohort, REPRESENTATION).x
        control_features = [
            build_representation(
                matched_random_control(
                    truncated,
                    loss.removed_cells,
                    catalogue,
                    seed=control_seed(args.seed, pattern, r),
                ),
                REPRESENTATION,
            ).x
            for r in range(args.control_repeats)
        ]

        for b, fold_models in enumerate(resplits):
            p_subset = predict_all(subset_features, fold_models)
            control_loss = np.zeros(n_patients, dtype=np.float64)
            for features in control_features:
                control_loss += per_patient_log_loss(y, predict_all(features, fold_models))
            control_loss /= args.control_repeats

            d = per_patient_log_loss(y, p_subset) - control_loss
            for k, fm in enumerate(fold_models):
                deltas[ci, b, k] = float(d[fm.test_index].mean())
            cand_auroc[ci, b] = float(auroc(y, p_subset))

            if b == REFERENCE_RESPLIT:
                reference_d[pattern] = d
                reference_rows.append(
                    {
                        "pattern": list(pattern),
                        "region": region,
                        "n_analytes": len(pattern),
                        "mean_delta": float(deltas[ci, b].mean()),
                        "fold_dispersion": fold_dispersion(deltas[ci, b]),
                        "auroc": cand_auroc[ci, b],
                        "auroc_drop": clean_auroc[b] - cand_auroc[ci, b],
                        "nll": float(negative_log_likelihood(y, p_subset)),
                        "brier": float(brier_score(y, p_subset)),
                        "calibration_intercept": float(calibration_intercept(y, p_subset)),
                        "calibration_slope": float(calibration_slope(y, p_subset)),
                        "mean_predicted_risk": float(p_subset.mean()),
                        "mean_realized_severity": float(loss.realized_severity.mean()),
                        "mean_removed_cells": float(loss.removed_cells.mean()),
                    }
                )
        if (ci + 1) % 10 == 0 or ci + 1 == len(candidates):
            elapsed = time.perf_counter() - t0
            log.info(
                "scoring progress",
                done=ci + 1,
                total=len(candidates),
                seconds=round(elapsed, 1),
                eta_seconds=round(elapsed / (ci + 1) * (len(candidates) - ci - 1), 1),
            )

    patterns = [pattern for pattern, _ in candidates]
    region_of = dict(candidates)

    def select(resplit: int, folds: tuple[int, ...]) -> AnalyteSubset:
        """Run the predeclared 1-SE parsimony rule over the named folds."""
        means = {
            patterns[ci]: float(deltas[ci, resplit, list(folds)].mean())
            for ci in range(len(patterns))
        }
        dispersions = {
            patterns[ci]: fold_dispersion(deltas[ci, resplit, list(folds)])
            for ci in range(len(patterns))
        }
        eligible = {
            patterns[ci]
            for ci in range(len(patterns))
            if clean_auroc[resplit] - cand_auroc[ci, resplit] <= args.delta
        }
        return select_one_se_parsimonious(means, dispersions, eligible)

    all_folds = tuple(range(args.folds))
    per_resplit = [select(b, all_folds) for b in range(args.repeats)]
    frequency = selection_frequency(per_resplit)
    frozen = min(frequency, key=lambda c: (-frequency[c], len(c), c))
    frozen_index = patterns.index(frozen)
    log.info("frozen pattern", pattern=list(frozen), pi=frequency[frozen])

    # ------------------------------- nested out-of-selection estimate ------
    # Re-select on four folds, read the effect off the held-out fifth. Pure
    # arithmetic on the table above: no extra model or prediction work.
    oos_components: list[dict[str, Any]] = []
    for b in range(args.repeats):
        for k in range(args.folds):
            others = tuple(f for f in all_folds if f != k)
            chosen = select(b, others)
            oos_components.append(
                {
                    "resplit": b,
                    "held_out_fold": k,
                    "selected": list(chosen),
                    "delta_on_held_out_fold": float(deltas[patterns.index(chosen), b, k]),
                }
            )
    delta_oos = float(np.mean([c["delta_on_held_out_fold"] for c in oos_components]))
    naive_delta = float(deltas[frozen_index].mean())

    # ------------------------------------------- detectability -------------
    sigma_delta = float(np.std(reference_d[frozen], ddof=1))
    mde = minimum_detectable_effect(sigma_delta, SET_C_N)

    # ------------------------------------------- gates ---------------------
    reference_auroc_drop = float(
        clean_auroc[REFERENCE_RESPLIT] - cand_auroc[frozen_index, REFERENCE_RESPLIT]
    )
    g1 = region_of[frozen] == "target"
    g2 = round(frequency[frozen] * args.repeats) >= MAJORITY
    g3 = reference_auroc_drop <= args.delta
    g4 = mde <= delta_oos
    if not g1:
        verdict = "v2-SANITY-FAILURE"
    elif not g2:
        verdict = "v2-DIFFUSE"
    elif not g3:
        # M5_V2_DESIGN.md section 12 enumerates outcomes for G1, G2 and G4 but not
        # for G3 failing alone. Recorded as a named gate failure rather than
        # forced into an outcome it does not fit, exactly as the M5-v1 taxonomy
        # gap was recorded rather than repaired retroactively.
        verdict = "v2-GATE-FAILURE-G3 (outcome not enumerated in design section 12)"
    elif not g4:
        verdict = "v2-UNDERPOWERED"
    else:
        verdict = "v2-STABLE"

    args.out.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": "cliniverse.m5v2/1",
        "design_document": "docs/M5_V2_DESIGN.md",
        "complete_enumeration": not args.limit,
        "predeclared": {
            "target_group": TARGET_GROUP,
            "null_control_groups": list(NULL_CONTROL_GROUPS),
            "n_resplits": args.repeats,
            "reference_resplit": REFERENCE_RESPLIT,
            "reference_seed": args.seed,
            "control_repeats": args.control_repeats,
            "auroc_delta": args.delta,
            "majority": MAJORITY,
            "set_c_n": SET_C_N,
            "selection_rule": "1-SE parsimony on mean fold excess NLL",
        },
        "statistical_status": (
            "All A+B quantities here are DEVELOPMENT estimates. The 20 resplits reuse "
            "the same 8,000 patients and are not independent. No interval, p-value or "
            "confirmatory claim may be derived from them."
        ),
        "n_candidates": len(candidates),
        "clean_auroc_by_resplit": clean_auroc,
        "reference_run": reference_rows,
        "selection_by_resplit": [list(c) for c in per_resplit],
        "selection_frequency": sorted(
            (
                {"pattern": list(k), "pi": v, "region": region_of[k]}
                for k, v in frequency.items()
            ),
            key=lambda r: -float(r["pi"]),
        ),
        "frozen_pattern": list(frozen),
        "frozen_pattern_region": region_of[frozen],
        "development_estimates": {
            "naive_mean_delta": naive_delta,
            "out_of_selection_delta": delta_oos,
            "shrinkage_fraction": (
                float(1.0 - delta_oos / naive_delta) if naive_delta > 0 else float("nan")
            ),
            "sigma_delta_reference_run": sigma_delta,
            "reference_auroc_drop": reference_auroc_drop,
        },
        "detectability": {
            "sigma_delta": sigma_delta,
            "set_c_n": SET_C_N,
            "alpha_one_sided": 0.05,
            "power": 0.80,
            "minimum_detectable_effect": mde,
            "out_of_selection_delta": delta_oos,
            "passes": g4,
        },
        "out_of_selection_components": oos_components,
        "gates": {
            "G1_null_control_sanity": {
                "passes": g1,
                "frozen_region": region_of[frozen],
                "note": "necessary, not sufficient to validate the method",
            },
            "G2_majority_stability": {
                "passes": g2,
                "pi": frequency[frozen],
                "count": round(frequency[frozen] * args.repeats),
                "required": MAJORITY,
                "of": args.repeats,
            },
            "G3_discrimination_silent": {
                "passes": g3,
                "auroc_drop_reference_run": reference_auroc_drop,
                "delta": args.delta,
            },
            "G4_detectability": {"passes": g4, "mde": mde, "delta_oos": delta_oos},
        },
        "verdict": verdict,
        "set_c": "LOCKED — not loaded, not scored, not referenced by any statistic here",
    }
    payload["provenance"] = build_provenance(
        cohort=truncated,
        splits=stratified_folds(cohort, n_folds=args.folds, seed=args.seed),
        config_payload={
            "xgb": XGB_PARAMS,
            "cutoff": CUTOFF,
            "seed": args.seed,
            "resplits": args.repeats,
            "control_repeats": args.control_repeats,
            "auroc_delta": args.delta,
        },
        extra={"excluded_sets": ["c"], "catalogue_version": catalogue.version},
    )

    np.savez_compressed(
        args.out / "m5v2_tables.npz",
        labels=y,
        record_ids=cohort.record_ids,
        deltas=deltas,
        candidate_auroc=cand_auroc,
        clean_auroc=np.asarray(clean_auroc),
        reference_d_frozen=reference_d[frozen],
        reference_clean_predictions=clean_predictions[REFERENCE_RESPLIT],
    )
    (args.out / "frozen_pattern.json").write_text(
        json.dumps(
            {
                "frozen_pattern": list(frozen),
                "region": region_of[frozen],
                "selected_on": "A+B development only; set-c never loaded",
                "selection_frequency": frequency[frozen],
                "verdict": verdict,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
        newline="\n",
    )
    (args.out / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
        newline="\n",
    )
    _print_report(payload)
    print(f"\nwrote {args.out / 'results.json'}")
    return 0


def _print_report(payload: dict[str, Any]) -> None:
    gates = payload["gates"]
    dev = payload["development_estimates"]
    det = payload["detectability"]

    def verdict(passed: bool) -> str:
        return "PASS" if passed else "FAIL"

    print("\n" + "=" * 96)
    print("M5-v2 — STABILITY-AWARE ADVERSARIAL FAILURE SEARCH (A+B DEVELOPMENT ONLY)")
    print("=" * 96)
    print(f"candidates: {payload['n_candidates']}  complete={payload['complete_enumeration']}")
    print(
        f"frozen pattern: {'+'.join(payload['frozen_pattern'])}  "
        f"[{payload['frozen_pattern_region']}]"
    )
    print("\nSELECTION FREQUENCY (top 8):")
    for row in payload["selection_frequency"][:8]:
        print(f"  pi={row['pi']:.2f}  {'+'.join(row['pattern']):<44} [{row['region']}]")
    print("\nDEVELOPMENT ESTIMATES (not confirmatory, no intervals):")
    print(f"  naive mean excess NLL          {dev['naive_mean_delta']:+.5f}")
    print(f"  out-of-selection excess NLL    {dev['out_of_selection_delta']:+.5f}")
    print(f"  shrinkage                      {dev['shrinkage_fraction'] * 100:.1f}%")
    print(f"  reference-run AUROC drop       {dev['reference_auroc_drop']:+.5f}")
    print("\nSET-C DETECTABILITY (set-c NOT loaded):")
    print(f"  sigma_delta (reference run)    {det['sigma_delta']:.5f}")
    print(f"  MDE at n=4000, 1-sided 5%, 80% {det['minimum_detectable_effect']:+.5f}")
    print(f"  out-of-selection effect        {det['out_of_selection_delta']:+.5f}")
    print("\nGATES:")
    print(f"  G1 null-control sanity : {verdict(gates['G1_null_control_sanity']['passes'])}")
    print(
        f"  G2 majority stability  : {verdict(gates['G2_majority_stability']['passes'])}"
        f"  ({gates['G2_majority_stability']['count']}/{gates['G2_majority_stability']['of']},"
        f" need {gates['G2_majority_stability']['required']})"
    )
    print(
        f"  G3 discrimination-silent: {verdict(gates['G3_discrimination_silent']['passes'])}"
    )
    print(f"  G4 detectability       : {verdict(gates['G4_detectability']['passes'])}")
    print(f"\nVERDICT: {payload['verdict']}")
    print(f"set-c: {payload['set_c']}")


if __name__ == "__main__":
    sys.exit(main())
