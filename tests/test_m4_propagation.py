"""M4 propagation invariants, exercised through the real `run_condition` path.

These exist because of a concrete failure. A smoke run at n=400 produced
bit-identical predictions for every policy, which looked exactly like a
state-propagation bug. A ten-stage differential trace located the loss at the
model call: with 150 model-training rows and the frozen ``min_child_weight=10``,
XGBoost could not make a single split, so all 200 trees were stumps, zero
features were used, and the model was a constant. No disclosure can move a
constant.

The pipeline was correct; the validation cohort was too small. The tests below
protect both halves of that lesson:

* a degenerate (constant) model must fail loudly rather than silently making
  every policy look identical;
* on a cohort large enough to fit a real model, a successful disclosure must
  actually change the stored prediction.
"""

from __future__ import annotations

import numpy as np
import pytest
from experiments.acquisition.m4_ranking_stability import (
    BUDGET_FRACTIONS,
    fit_folds,
    run_condition,
    score_condition,
)

from cliniverse.acquisition import load_panel_catalogue
from cliniverse.data import load_cohort
from cliniverse.data.splits import development_cohort
from cliniverse.exceptions import ConfigError
from twinbench.disclosure import Protocol

SEED = 20260809
#: Large enough that the frozen hyperparameters admit splits. Below roughly this
#: size the booster collapses to stumps; see the module docstring.
SUFFICIENT_N = 2000
DEGENERATE_N = 400


@pytest.fixture(scope="module")
def cohort():
    try:
        return development_cohort(load_cohort(download=False))
    except Exception as exc:  # pragma: no cover - depends on local data cache
        pytest.skip(f"development cohort unavailable: {exc}")


@pytest.mark.slow
def test_degenerate_model_is_rejected_rather_than_silently_constant(cohort) -> None:
    """A constant booster must raise, not quietly equalise every policy.

    Without this guard an undersized cohort produces a stump-only model, every
    policy scores identically, and the run is indistinguishable from a broken
    evaluator.
    """
    small = cohort.select(np.arange(DEGENERATE_N, dtype=np.int64))
    y = small.labels["mortality"].astype(np.float64)
    with pytest.raises(ConfigError, match="zero features"):
        fit_folds(small, y, SEED, 2)


@pytest.mark.slow
def test_fitted_folds_split_on_real_features(cohort) -> None:
    sub = cohort.select(np.arange(SUFFICIENT_N, dtype=np.int64))
    y = sub.labels["mortality"].astype(np.float64)
    for fm in fit_folds(sub, y, SEED, 2):
        assert fm.n_features_used > 0


@pytest.mark.slow
def test_successful_disclosure_changes_the_stored_prediction(cohort) -> None:
    """The invariant the whole milestone rests on.

    If acquiring information cannot move a stored prediction, every budget curve
    is flat and every ranking is meaningless. This exercises the real
    `run_condition` path, not the disclosure engine in isolation.
    """
    sub = cohort.select(np.arange(SUFFICIENT_N, dtype=np.int64))
    y = sub.labels["mortality"].astype(np.float64)
    folds = fit_folds(sub, y, SEED, 2)
    catalogue = load_panel_catalogue()

    predictions, spend, _ = run_condition(
        sub,
        y,
        folds,
        catalogue,
        Protocol.SUPPORT_BLIND,
        "shared_plus_marginal",
        0.6,
        SEED,
    )

    # There must actually be something to disclose, or the test proves nothing.
    disclosed = [
        s["mean_disclosed_cells"]
        for s in spend
        if s["policy"] == "fixed_domain_order" and s["budget_fraction"] == 1.0
    ]
    assert disclosed and max(disclosed) > 0, "no successful disclosure to test"

    baseline = predictions["no_acquisition"][1.0]
    acquiring = predictions["fixed_domain_order"][1.0]
    changed = int((~np.isclose(baseline, acquiring)).sum())
    assert changed > 0, (
        "a successful disclosure did not change any stored prediction; "
        "policy-specific state is not reaching the model"
    )


@pytest.mark.slow
def test_zero_budget_matches_no_acquisition_exactly(cohort) -> None:
    """At zero budget every policy must reduce to the no-acquisition state."""
    sub = cohort.select(np.arange(SUFFICIENT_N, dtype=np.int64))
    y = sub.labels["mortality"].astype(np.float64)
    folds = fit_folds(sub, y, SEED, 2)
    catalogue = load_panel_catalogue()

    predictions, _, _ = run_condition(
        sub,
        y,
        folds,
        catalogue,
        Protocol.SUPPORT_BLIND,
        "shared_plus_marginal",
        0.6,
        SEED,
    )
    baseline = predictions["no_acquisition"][0.0]
    for policy, by_budget in predictions.items():
        np.testing.assert_allclose(
            by_budget[0.0], baseline, err_msg=f"{policy} spent something at zero budget"
        )


@pytest.mark.slow
def test_budget_grid_is_fully_populated(cohort) -> None:
    """Every policy must yield a finite prediction at every predeclared budget."""
    sub = cohort.select(np.arange(SUFFICIENT_N, dtype=np.int64))
    y = sub.labels["mortality"].astype(np.float64)
    folds = fit_folds(sub, y, SEED, 2)
    catalogue = load_panel_catalogue()

    predictions, _, _ = run_condition(
        sub,
        y,
        folds,
        catalogue,
        Protocol.SUPPORT_BLIND,
        "shared_plus_marginal",
        0.6,
        SEED,
    )
    rows, integrated = score_condition(y, predictions)
    for policy, by_budget in predictions.items():
        assert set(by_budget) == set(BUDGET_FRACTIONS), policy
        for beta, arr in by_budget.items():
            assert np.isfinite(arr).all(), f"{policy} at beta={beta}"
    assert all(np.isfinite(v) for v in integrated.values())
    assert len(rows) == len(predictions) * len(BUDGET_FRACTIONS)
