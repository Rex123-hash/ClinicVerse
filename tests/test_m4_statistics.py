"""Regression tests for M4 endpoint and reversal terminology."""

from __future__ import annotations

import numpy as np
import pytest
from experiments.acquisition.m4_ranking_stability import (
    BUDGET_FRACTIONS,
    classify_reversal_support,
    integrate,
    paired_integrated_bootstrap,
)

from cliniverse.evaluation.metrics import negative_log_likelihood


def test_one_condition_interval_does_not_become_supported_reversal() -> None:
    support: dict[str, dict[str, float | bool]] = {
        "condition_a": {"low": -0.2, "high": 0.1, "excludes_zero": False},
        "condition_b": {"low": 0.01, "high": 0.2, "excludes_zero": True},
    }

    classification, predeclared_flag, resolved_in_both = classify_reversal_support(support)

    assert predeclared_flag is True
    assert resolved_in_both is False
    assert classification == "ONE-CONDITION EVIDENCE / REVERSAL UNRESOLVED"


def test_both_intervals_are_required_for_supported_reversal() -> None:
    support: dict[str, dict[str, float | bool]] = {
        "condition_a": {"low": -0.2, "high": -0.01, "excludes_zero": True},
        "condition_b": {"low": 0.01, "high": 0.2, "excludes_zero": True},
    }

    classification, predeclared_flag, resolved_in_both = classify_reversal_support(support)

    assert predeclared_flag is True
    assert resolved_in_both is True
    assert classification == "SUPPORTED REVERSAL"


def test_fast_bootstrap_matches_explicit_curve_rebuilds() -> None:
    rng = np.random.default_rng(4)
    y = np.array([0.0, 1.0] * 20)
    a = {
        beta: np.clip(rng.uniform(0.05, 0.95, len(y)) - 0.01 * beta, 0.01, 0.99)
        for beta in BUDGET_FRACTIONS
    }
    b = {
        beta: np.clip(rng.uniform(0.05, 0.95, len(y)) + 0.01 * beta, 0.01, 0.99)
        for beta in BUDGET_FRACTIONS
    }
    actual = paired_integrated_bootstrap(y, a, b, n_boot=50, seed=9)

    reference_rng = np.random.default_rng(9)
    reference: list[float] = []
    for _ in range(50):
        idx = reference_rng.integers(0, len(y), len(y))
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        curve_a = [negative_log_likelihood(yb, a[beta][idx]) for beta in BUDGET_FRACTIONS]
        curve_b = [negative_log_likelihood(yb, b[beta][idx]) for beta in BUDGET_FRACTIONS]
        reference.append(integrate(curve_b) - integrate(curve_a))
    low, high = np.percentile(reference, [2.5, 97.5])

    assert actual["difference"] == pytest.approx(
        integrate([negative_log_likelihood(y, b[k]) for k in BUDGET_FRACTIONS])
        - integrate([negative_log_likelihood(y, a[k]) for k in BUDGET_FRACTIONS]),
        abs=1e-15,
    )
    assert actual["low"] == pytest.approx(low, abs=1e-15)
    assert actual["high"] == pytest.approx(high, abs=1e-15)
