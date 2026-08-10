"""M4: does the acquisition policy judged best stay best when assumptions change?

Design predeclared in `docs/M4_DESIGN.md` and committed before this ran.

Every state transition goes through the tested TwinBench DisclosureEngine. The
model, imputer and calibrator are frozen from M2/M3 and fitted only on clean
training partitions; the calibrator is never refitted under any acquisition
state. set-c is never loaded.

Usage:
    python experiments/acquisition/m4_ranking_stability.py --scope primary
    python experiments/acquisition/m4_ranking_stability.py --scope grid
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import pathlib
import sys
import time
from typing import Any

import numpy as np
from scipy.stats import kendalltau
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from cliniverse.acquisition import load_panel_catalogue
from cliniverse.acquisition.evaluator import run_adaptive
from cliniverse.acquisition.policies import (
    FixedOrderBatch,
    GreedyEIGBatch,
    NoAcquisitionBatch,
    RandomTrainFrequencyBatch,
    RandomUniformBatch,
)
from cliniverse.acquisition.simulation import build_training_quantiles, make_simulator
from cliniverse.data import load_cohort
from cliniverse.data.cohort import Cohort
from cliniverse.data.splits import development_cohort, stratified_folds
from cliniverse.evaluation.artifacts import build_provenance, stable_hash
from cliniverse.evaluation.calibration import CalibratorKind, build_calibrator
from cliniverse.evaluation.information_loss import eligible_columns
from cliniverse.evaluation.metrics import (
    auprc,
    auroc,
    brier_score,
    calibration_intercept,
    calibration_slope,
    negative_log_likelihood,
)
from cliniverse.evaluation.representations import (
    FittedImputer,
    ImputationStrategy,
    Representation,
    build_representation,
)
from cliniverse.exceptions import ConfigError
from cliniverse.log import get_logger
from twinbench.cases import build_manifest, engine_for
from twinbench.disclosure import Protocol
from twinbench.masking import GroupHours

log = get_logger(__name__)

# ------------------------------------------------- frozen from M2 / M3 ------
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
EPOCH_HOURS = (12, 18, 24)

# ------------------------------------------- predeclared in M4_DESIGN.md ----
BUDGET_FRACTIONS: tuple[float, ...] = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0)
COST_REGIMES: tuple[str, ...] = (
    "shared_plus_marginal",
    "uniform_group",
    "ordinal_tier",
    "per_analyte",
)
MASK_RATES: tuple[float, ...] = (0.3, 0.6)
PROTOCOLS = (Protocol.SUPPORT_BLIND, Protocol.SUPPORT_AWARE)
FIXED_ORDER = (
    "BMP_like",
    "CBC_like",
    "ABG_like",
    "Lactate",
    "hepatic_like",
    "SaO2",
    "Albumin",
    "TroponinT",
    "TroponinI",
    "Cholesterol",
)
#: Policies that enter ranking tables. The support oracle and the
#: full-information ceiling are diagnostics and are excluded (M4_DESIGN 11).
RANKED_POLICIES = (
    "no_acquisition",
    "random_uniform_all",
    "random_train_frequency",
    "fixed_domain_order",
    "greedy_eig",
    "greedy_eig_per_cost",
)
PRIMARY_CONDITION = ("support_blind", "shared_plus_marginal", 0.6)
GRID_SUBSAMPLE = 2000

METRICS = {
    "nll": negative_log_likelihood,
    "brier": brier_score,
    "auroc": auroc,
    "auprc": auprc,
    "calibration_intercept": calibration_intercept,
    "calibration_slope": calibration_slope,
}


def integrate(values: list[float], grid: tuple[float, ...] = BUDGET_FRACTIONS) -> float:
    """Trapezoidal integral over the normalized budget grid, divided by width."""
    x = np.asarray(grid, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


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

        Zero means every tree is a stump and the model is a constant. That is a
        silent disaster for an acquisition experiment: no disclosure can change a
        constant prediction, so every policy scores identically and the run looks
        like a propagation bug. It is what happens when the training partition is
        too small for the frozen `min_child_weight`.
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
        fold_model = out[-1]
        used = fold_model.n_features_used
        if used == 0:
            raise ConfigError(
                f"fold {split.fold}: the fitted model splits on zero features, so it "
                f"is a constant and no acquisition can change its prediction. "
                f"n_model_train={len(train_idx)} with min_child_weight="
                f"{XGB_PARAMS['min_child_weight']} is too small. Increase the cohort "
                f"rather than changing the frozen hyperparameters."
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


def make_policies(
    training_mask: np.ndarray,
    variable_names: tuple[str, ...],
    groups: dict[str, list[int]],
    feature_names: tuple[str, ...],
    group_members: dict[str, tuple[str, ...]],
    quantiles: dict[str, tuple[float, ...]],
    costs: dict[str, float],
    predict: Any,
    seed: int,
) -> dict[str, Any]:
    """Instantiate every policy. Learned quantities come from training data only."""
    simulate = make_simulator(feature_names, group_members, quantiles)
    freq = RandomTrainFrequencyBatch.fit(training_mask, variable_names, groups, seed=seed)
    return {
        "no_acquisition": NoAcquisitionBatch(),
        "random_uniform_all": RandomUniformBatch(seed=seed),
        "random_train_frequency": freq,
        "fixed_domain_order": FixedOrderBatch(order=FIXED_ORDER),
        "greedy_eig": GreedyEIGBatch(
            predict=predict, simulate=simulate, costs=costs, per_cost=False
        ),
        "greedy_eig_per_cost": GreedyEIGBatch(
            predict=predict,
            simulate=simulate,
            costs=costs,
            per_cost=True,
            name="greedy_eig_per_cost",
        ),
    }


def run_condition(
    cohort: Cohort,
    y: np.ndarray,
    fold_models: list[FoldModel],
    catalogue: Any,
    protocol: Protocol,
    cost_regime: str,
    mask_rate: float,
    seed: int,
    *,
    collect_trace: bool = False,
) -> tuple[dict[str, dict[float, np.ndarray]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate every policy across the budget grid for one condition."""
    priced = (
        catalogue
        if cost_regime == catalogue.schedule_name
        else catalogue.with_schedule(cost_regime)
    )
    total_cost = sum(priced.cost_of(a) for a in priced.panel_names)
    costs = {a: priced.cost_of(a) for a in priced.panel_names}

    truncated = cohort.truncate(CUTOFF)
    mechanism = GroupHours(rate=mask_rate, seed=seed)
    _, group_cols = eligible_columns(truncated, priced)
    groups = {k: v.tolist() for k, v in group_cols.items()}
    group_members = {a: priced.panels[a].members for a in priced.panel_names}

    feature_names = build_representation(truncated, REPRESENTATION).names

    predictions: dict[str, dict[float, np.ndarray]] = {}
    spend_rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []

    for fm in fold_models:
        test_idx = fm.test_index
        train_idx = np.setdiff1d(np.arange(cohort.n_patients), test_idx)
        sub = truncated.select(test_idx)

        quantiles = build_training_quantiles(
            truncated.x[train_idx].astype(np.float64),
            truncated.m[train_idx],
            truncated.variable_names,
        )
        policies = make_policies(
            truncated.m[train_idx],
            truncated.variable_names,
            groups,
            feature_names,
            group_members,
            quantiles,
            costs,
            fm.predict,
            seed + fm.fold,
        )

        manifest = build_manifest(
            sub,
            mechanism,
            priced,
            sets=tuple(sorted(set(sub.source_set.tolist()))),
            cutoff_hours=CUTOFF,
            protocol=protocol,
            cost_regime=cost_regime,
            budget=total_cost,
            epoch_hours=EPOCH_HOURS,
        )

        for policy_name, policy in policies.items():
            predictions.setdefault(policy_name, {})
            for beta in BUDGET_FRACTIONS:
                budget = beta * total_cost
                # CaseSpec is a frozen pydantic model; model_copy re-specifies the
                # budget without touching the seeded mask or catalogue binding.
                engines = [
                    engine_for(
                        sub, case.model_copy(update={"budget": budget}), mechanism, priced
                    )
                    for case in manifest.cases
                ]
                final, trace = run_adaptive(
                    engines,
                    policy,
                    sub,
                    lambda c: build_representation(c, REPRESENTATION).x,
                    collect_trace=collect_trace and beta == 0.5,
                )
                features = build_representation(final, REPRESENTATION).x
                preds = fm.predict(features)

                arr = predictions[policy_name].setdefault(
                    beta, np.full(cohort.n_patients, np.nan)
                )
                arr[test_idx] = preds

                spent = np.array([e.spent for e in engines], dtype=np.float64)
                disclosed = np.array(
                    [sum(p.n_disclosed for p in e.purchases) for e in engines],
                    dtype=np.float64,
                )
                requests = np.array([len(e.purchases) for e in engines])
                failed = np.array(
                    [sum(1 for p in e.purchases if p.was_empty) for e in engines]
                )
                spend_rows.append(
                    {
                        "policy": policy_name,
                        "fold": fm.fold,
                        "budget_fraction": beta,
                        "n_patients": len(engines),
                        "total_realized_cost": float(spent.sum()),
                        "total_disclosed_cells": int(disclosed.sum()),
                        "total_requests": int(requests.sum()),
                        "total_failed_requests": int(failed.sum()),
                        "mean_realized_cost": float(spent.mean()),
                        "mean_disclosed_cells": float(disclosed.mean()),
                        "mean_requests": float(requests.mean()),
                        "mean_failed_requests": float(failed.mean()),
                        "failure_rate": float(failed.sum() / max(requests.sum(), 1)),
                    }
                )
                if trace:
                    traces.extend(
                        {
                            **r.as_row(),
                            "record_id": int(sub.record_ids[r.patient_index]),
                            "policy": policy_name,
                            "fold": fm.fold,
                            "budget_fraction": beta,
                        }
                        for r in trace
                        if r.patient_index < 2
                    )
        del sub
    return predictions, spend_rows, traces


def score_condition(
    y: np.ndarray, predictions: dict[str, dict[float, np.ndarray]]
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Metric-vs-budget rows and the integrated primary score per policy."""
    rows: list[dict[str, Any]] = []
    integrated: dict[str, float] = {}
    for policy, by_budget in predictions.items():
        nll_curve: list[float] = []
        for beta in BUDGET_FRACTIONS:
            p = by_budget[beta]
            if not np.isfinite(p).all():
                raise RuntimeError(f"incomplete predictions for {policy} at beta={beta}")
            row = {"policy": policy, "budget_fraction": beta}
            row.update({name: float(fn(y, p)) for name, fn in METRICS.items()})
            row["mean_predicted_risk"] = float(p.mean())
            rows.append(row)
            nll_curve.append(row["nll"])
        integrated[policy] = integrate(nll_curve)
    return rows, integrated


def paired_integrated_bootstrap(
    y: np.ndarray,
    a: dict[float, np.ndarray],
    b: dict[float, np.ndarray],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    """Paired bootstrap of delta AUNLLC (b minus a).

    Curves are rebuilt inside every replicate, as the design requires; already
    aggregated budget points are never resampled independently.
    """
    rng = np.random.default_rng(seed)
    n = len(y)

    # NLL is a patient mean and trapezoidal integration is linear. Therefore
    # integrating each patient's eight point-losses once and averaging those
    # integrated losses on a resample is algebraically identical to rebuilding
    # eight mean-NLL points and integrating the curve inside every replicate.
    # This removes ~100,000 repeated sklearn metric calls without changing a
    # patient index, seed, budget weight, or bootstrap definition.
    weights = np.zeros(len(BUDGET_FRACTIONS), dtype=np.float64)
    for i, (left, right) in enumerate(itertools.pairwise(BUDGET_FRACTIONS)):
        half_width = (right - left) / 2.0
        weights[i] += half_width
        weights[i + 1] += half_width

    def patient_integrated_loss(predictions: dict[float, np.ndarray]) -> np.ndarray:
        out = np.zeros(n, dtype=np.float64)
        for weight, beta in zip(weights, BUDGET_FRACTIONS, strict=True):
            probability = np.clip(predictions[beta], 1e-15, 1.0 - 1e-15)
            point_loss = -(y * np.log(probability) + (1.0 - y) * np.log1p(-probability))
            out += weight * point_loss
        return out / (BUDGET_FRACTIONS[-1] - BUDGET_FRACTIONS[0])

    loss_a = patient_integrated_loss(a)
    loss_b = patient_integrated_loss(b)
    patient_differences = loss_b - loss_a
    point = float(patient_differences.mean())
    diffs: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(float(patient_differences[idx].mean()))
    lo, hi = (float(v) for v in np.percentile(diffs, [2.5, 97.5]))
    return {
        "difference": point,
        "low": lo,
        "high": hi,
        "n_boot": len(diffs),
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
    }


def classify_reversal_support(
    support: dict[str, dict[str, float | bool]],
) -> tuple[str, bool, bool]:
    """Classify two-condition evidence without overstating a one-sided result.

    The design's binding rule flagged evidence when either condition excluded
    zero. Review #4 retains that flag for transparency, but reserves "supported
    reversal" for the case in which both relevant paired comparisons resolve.
    """
    excludes = [bool(interval["excludes_zero"]) for interval in support.values()]
    one_condition_evidence = any(excludes)
    resolved_in_both = all(excludes)
    if resolved_in_both:
        classification = "SUPPORTED REVERSAL"
    elif one_condition_evidence:
        classification = "ONE-CONDITION EVIDENCE / REVERSAL UNRESOLVED"
    else:
        classification = "UNRESOLVED / EFFECTIVELY TIED"
    return classification, one_condition_evidence, resolved_in_both


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("primary", "grid", "both"), default="both")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("experiments/acquisition/results/m4"),
    )
    args = parser.parse_args(argv)

    catalogue = load_panel_catalogue()
    cohort = development_cohort(load_cohort())
    y_full = cohort.labels["mortality"].astype(np.float64)
    log.info("cohort", n=cohort.n_patients, prevalence=float(y_full.mean()))

    args.out.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": "cliniverse.m4/1",
        "design_document": "docs/M4_DESIGN.md",
        "budget_grid": list(BUDGET_FRACTIONS),
        "ranked_policies": list(RANKED_POLICIES),
        "conditions": [],
    }

    # ---------------- primary condition, all 8,000 patients ----------------
    fold_models = fit_folds(cohort, y_full, args.seed, args.folds)
    if args.scope in ("primary", "both"):
        protocol, regime, rate = PRIMARY_CONDITION
        t0 = time.perf_counter()
        preds, spend, traces = run_condition(
            cohort,
            y_full,
            fold_models,
            catalogue,
            Protocol(protocol),
            regime,
            rate,
            args.seed,
            collect_trace=True,
        )
        rows, integrated = score_condition(y_full, preds)
        payload["primary"] = {
            "condition": {
                "protocol": protocol,
                "cost_regime": regime,
                "mask_rate": rate,
                "n_patients": cohort.n_patients,
            },
            "rows": rows,
            "integrated_aunllc": integrated,
            "spend": spend,
            # Complete action histories for two deterministic patients per fold
            # and policy, rather than a prefix that contained only step-zero
            # records from the first policy.
            "trace_sample": traces,
            "seconds": round(time.perf_counter() - t0, 1),
        }
        np.savez_compressed(
            args.out / "primary_predictions.npz",
            labels=y_full,
            record_ids=cohort.record_ids,
            **{
                f"{pol}|{beta}": arr
                for pol, by_b in preds.items()
                for beta, arr in by_b.items()
            },
        )
        _print_primary(rows, integrated, spend, y_full)

    # ---------------- stability grid, fixed 2,000-patient subsample --------
    if args.scope in ("grid", "both"):
        rng = np.random.default_rng(args.seed)
        subsample = np.sort(
            rng.choice(cohort.n_patients, size=GRID_SUBSAMPLE, replace=False)
        ).astype(np.int64)
        sub_cohort = cohort.select(subsample)
        y_sub = y_full[subsample]
        sub_models = [
            dataclasses.replace(
                fm,
                test_index=np.flatnonzero(np.isin(subsample, fm.test_index)).astype(np.int64),
            )
            for fm in fold_models
        ]
        log.info("grid subsample", n=len(subsample), prevalence=float(y_sub.mean()))

        grid_results: list[dict[str, Any]] = []
        grid_preds: dict[str, dict[str, dict[float, np.ndarray]]] = {}
        for protocol, regime, rate in itertools.product(PROTOCOLS, COST_REGIMES, MASK_RATES):
            key = f"{protocol}|{regime}|{rate}"
            t0 = time.perf_counter()
            preds, spend, _ = run_condition(
                sub_cohort, y_sub, sub_models, catalogue, protocol, regime, rate, args.seed
            )
            rows, integrated = score_condition(y_sub, preds)
            ranked = {k: v for k, v in integrated.items() if k in RANKED_POLICIES}
            order = sorted(ranked, key=lambda k: ranked[k])
            grid_results.append(
                {
                    "condition": key,
                    "protocol": str(protocol),
                    "cost_regime": regime,
                    "mask_rate": rate,
                    "rows": rows,
                    "integrated_aunllc": integrated,
                    "ranking": order,
                    "winner": order[0],
                    "spend": spend,
                }
            )
            grid_preds[key] = preds
            log.info(
                "condition done",
                condition=key,
                winner=order[0],
                seconds=round(time.perf_counter() - t0, 1),
            )

        payload["grid"] = {
            "n_patients": len(subsample),
            "subsample_hash": stable_hash(subsample.tolist()),
            "results": grid_results,
        }
        payload["conditions"] = [result["condition"] for result in grid_results]
        payload["stability"] = _stability(grid_results, y_sub, grid_preds, args)
        np.savez_compressed(
            args.out / "grid_predictions.npz",
            labels=y_sub,
            record_ids=cohort.record_ids[subsample],
            **{
                f"{cond}|{pol}|{beta}": arr
                for cond, by_pol in grid_preds.items()
                for pol, by_b in by_pol.items()
                for beta, arr in by_b.items()
            },
        )
        _print_grid(payload["grid"], payload["stability"])

    payload["provenance"] = build_provenance(
        cohort=cohort.truncate(CUTOFF),
        splits=stratified_folds(cohort, n_folds=args.folds, seed=args.seed),
        config_payload={
            "xgb": XGB_PARAMS,
            "budgets": BUDGET_FRACTIONS,
            "regimes": COST_REGIMES,
            "mask_rates": MASK_RATES,
            "seed": args.seed,
            "cutoff": CUTOFF,
        },
        extra={"excluded_sets": ["c"], "catalogue_version": catalogue.version},
    )
    (args.out / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
        newline="\n",
    )
    print(f"\nwrote {args.out / 'results.json'}")
    return 0


def _stability(
    grid: list[dict[str, Any]],
    y: np.ndarray,
    preds: dict[str, dict[str, dict[float, np.ndarray]]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Kendall tau-b between every condition pair, plus winner-change analysis."""
    conditions = [g["condition"] for g in grid]
    rank_of = {g["condition"]: {p: i for i, p in enumerate(g["ranking"])} for g in grid}
    taus: list[dict[str, Any]] = []
    for a, b in itertools.combinations(conditions, 2):
        ra = [rank_of[a][p] for p in RANKED_POLICIES]
        rb = [rank_of[b][p] for p in RANKED_POLICIES]
        tau, _ = kendalltau(ra, rb)
        inversions = sum(
            1
            for i, j in itertools.combinations(range(len(RANKED_POLICIES)), 2)
            if (ra[i] - ra[j]) * (rb[i] - rb[j]) < 0
        )
        taus.append({"a": a, "b": b, "kendall_tau_b": float(tau), "inversions": inversions})

    winners = {g["condition"]: g["winner"] for g in grid}
    distinct = sorted(set(winners.values()))
    changes: list[dict[str, Any]] = []
    for a, b in itertools.combinations(conditions, 2):
        if winners[a] == winners[b]:
            continue
        wa, wb = winners[a], winners[b]
        # Is the swap supported? Test both competitors in both conditions.
        support = {}
        for cond in (a, b):
            support[cond] = paired_integrated_bootstrap(
                y,
                preds[cond][wa],
                preds[cond][wb],
                n_boot=args.n_boot,
                seed=args.seed,
            )
        classification, one_condition_evidence, resolved_in_both = classify_reversal_support(
            support
        )
        changes.append(
            {
                "a": a,
                "b": b,
                "winner_a": wa,
                "winner_b": wb,
                "paired_delta_aunllc": support,
                # Preserve the binding predeclared at-least-one-condition flag,
                # but do not mislabel a reversal as statistically supported when
                # the competing-policy CI still includes zero in one condition.
                "predeclared_one_condition_evidence": one_condition_evidence,
                "resolved_in_both_conditions": resolved_in_both,
                "classification": classification,
            }
        )
    return {
        "kendall_tau_b": taus,
        "mean_kendall_tau_b": float(np.mean([t["kendall_tau_b"] for t in taus])),
        "min_kendall_tau_b": float(np.min([t["kendall_tau_b"] for t in taus])),
        "winners": winners,
        "distinct_winners": distinct,
        "winner_changes": changes,
        "n_supported_reversals": sum(
            1 for c in changes if c["classification"] == "SUPPORTED REVERSAL"
        ),
        "n_predeclared_one_condition_evidence": sum(
            1 for c in changes if c["predeclared_one_condition_evidence"]
        ),
    }


def _print_primary(
    rows: list[dict[str, Any]],
    integrated: dict[str, float],
    spend: list[dict[str, Any]],
    y: np.ndarray,
) -> None:
    print("\n" + "=" * 110)
    print(
        f"M4 PRIMARY — support_blind / shared_plus_marginal / mask 0.6 | "
        f"n={len(y):,} | prevalence={y.mean():.2%}"
    )
    print("=" * 110)
    print(
        f"{'policy':<24}{'beta':>6}{'spend':>8}{'disc':>8}{'NLL':>9}{'Brier':>9}"
        f"{'AUROC':>9}{'AP':>8}{'intcpt':>8}{'slope':>8}"
    )
    print("-" * 110)
    agg: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for s in spend:
        agg.setdefault((s["policy"], s["budget_fraction"]), []).append(s)
    for r in rows:
        k = (r["policy"], r["budget_fraction"])
        entries = agg.get(k, [])
        mean_spend = (
            float(np.mean([e["mean_realized_cost"] for e in entries])) if entries else 0.0
        )
        mean_disc = (
            float(np.mean([e["mean_disclosed_cells"] for e in entries])) if entries else 0.0
        )
        print(
            f"{r['policy']:<24}{r['budget_fraction']:>6.2f}{mean_spend:>8.2f}"
            f"{mean_disc:>8.1f}{r['nll']:>9.4f}{r['brier']:>9.4f}{r['auroc']:>9.4f}"
            f"{r['auprc']:>8.4f}{r['calibration_intercept']:>8.3f}"
            f"{r['calibration_slope']:>8.3f}"
        )
    print("\nINTEGRATED AUNLLC (lower is better):")
    for pol in sorted(integrated, key=lambda k: integrated[k]):
        flag = "" if pol in RANKED_POLICIES else "   [diagnostic]"
        print(f"  {integrated[pol]:.5f}  {pol}{flag}")


def _print_grid(grid: dict[str, Any], stability: dict[str, Any]) -> None:
    print("\n" + "=" * 110)
    print(f"M4 STABILITY GRID — n={grid['n_patients']:,} fixed paired subsample")
    print("=" * 110)
    print(f"{'condition':<48}{'winner':<24}{'ranking (best first)'}")
    print("-" * 110)
    for g in grid["results"]:
        print(f"{g['condition']:<48}{g['winner']:<24}{' > '.join(g['ranking'][:3])} ...")
    print(f"\ndistinct winners: {stability['distinct_winners']}")
    print(
        f"Kendall tau-b across condition pairs: mean="
        f"{stability['mean_kendall_tau_b']:.3f}  min={stability['min_kendall_tau_b']:.3f}"
    )
    print(f"supported reversals: {stability['n_supported_reversals']}")
    for c in stability["winner_changes"][:20]:
        print(f"  [{c['classification']}] {c['winner_a']} vs {c['winner_b']}")


if __name__ == "__main__":
    sys.exit(main())
