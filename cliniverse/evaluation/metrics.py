"""Classification metrics, calibration diagnostics and paired bootstrap inference.

Two decisions here are methodological rather than stylistic.

**Brier and log-loss are proper scoring rules, not calibration metrics.** They
combine calibration and refinement, so a better Brier does not on its own show
better calibration. The direct calibration readouts are the *calibration slope*
and *intercept* from a logistic recalibration of the predicted log-odds, plus
the reliability curve. All are reported together.

**Model comparison uses paired resampling, never overlapping intervals.**
Two standalone confidence intervals that overlap do not imply the difference is
insignificant, and two that do not overlap overstate the evidence. Because every
representation is scored on the *same* patients, the difference has far lower
variance than either estimate alone, and :func:`paired_bootstrap_difference`
resamples patients once and re-scores both predictors on that identical resample.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from cliniverse.exceptions import ConfigError

FloatArray = npt.NDArray[np.float64]

#: Predictions are clipped before any log is taken. A single 0.0 on a positive
#: case would otherwise send log-loss to infinity and destroy the whole run.
_EPS = 1e-15

MetricFn = Callable[[FloatArray, FloatArray], float]


@dataclasses.dataclass(frozen=True, slots=True)
class Interval:
    """A point estimate with a percentile bootstrap interval."""

    point: float
    low: float
    high: float
    n_boot: int

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.low:.4f}, {self.high:.4f}]"

    def as_dict(self) -> dict[str, float | int]:
        return {
            "point": self.point,
            "ci_low": self.low,
            "ci_high": self.high,
            "n_boot": self.n_boot,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class PairedDifference:
    """A paired difference ``b - a`` with a bootstrap interval over patients."""

    metric: str
    name_a: str
    name_b: str
    difference: float
    low: float
    high: float
    n_boot: int
    excludes_zero: bool

    def as_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)

    def __str__(self) -> str:
        verdict = "excludes 0" if self.excludes_zero else "includes 0"
        return (
            f"{self.metric}: {self.name_b} - {self.name_a} = "
            f"{self.difference:+.4f} [{self.low:+.4f}, {self.high:+.4f}] ({verdict})"
        )


# ------------------------------------------------------------------ core ----
def _validate(y: FloatArray, p: FloatArray) -> tuple[FloatArray, FloatArray]:
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    if y.shape != p.shape:
        raise ConfigError(f"labels {y.shape} and predictions {p.shape} differ in shape")
    if y.size == 0:
        raise ConfigError("cannot score an empty prediction set")
    if not np.isfinite(p).all():
        raise ConfigError("predictions contain non-finite values")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ConfigError("labels must be binary 0/1")
    return y, p


def brier_score(y: FloatArray, p: FloatArray) -> float:
    """Mean squared error of the probability forecast. A proper scoring rule."""
    y, p = _validate(y, p)
    return float(np.mean((p - y) ** 2))


def negative_log_likelihood(y: FloatArray, p: FloatArray) -> float:
    """Log loss. A proper scoring rule, not a calibration metric."""
    y, p = _validate(y, p)
    return float(log_loss(y, np.clip(p, _EPS, 1 - _EPS), labels=[0, 1]))


def auroc(y: FloatArray, p: FloatArray) -> float:
    y, p = _validate(y, p)
    return float(roc_auc_score(y, p))


def auprc(y: FloatArray, p: FloatArray) -> float:
    y, p = _validate(y, p)
    return float(average_precision_score(y, p))


def _logit(p: FloatArray) -> FloatArray:
    clipped = np.clip(p, _EPS, 1 - _EPS)
    return np.asarray(np.log(clipped / (1 - clipped)), dtype=np.float64)


def _recalibration_fit(y: FloatArray, p: FloatArray) -> tuple[float, float]:
    """Fit ``y ~ intercept + slope * logit(p)``; return ``(intercept, slope)``.

    Perfect calibration gives slope 1 and intercept 0. Slope below 1 indicates
    overconfident predictions (spread too wide); intercept away from 0 indicates
    a systematic shift in overall risk.
    """
    y, p = _validate(y, p)
    x = _logit(p).reshape(-1, 1)
    if len(np.unique(y)) < 2 or np.unique(x).size < 2:
        return float("nan"), float("nan")
    # Effectively unpenalised: we want the maximum-likelihood recalibration.
    model = LogisticRegression(C=1e12, solver="lbfgs", max_iter=1000)
    model.fit(x, y.astype(int))
    return float(model.intercept_[0]), float(model.coef_[0][0])


def calibration_slope(y: FloatArray, p: FloatArray) -> float:
    return _recalibration_fit(y, p)[1]


def calibration_intercept(y: FloatArray, p: FloatArray) -> float:
    return _recalibration_fit(y, p)[0]


def reliability_curve(
    y: FloatArray, p: FloatArray, n_bins: int = 10
) -> dict[str, list[float]]:
    """Equal-mass reliability bins.

    Equal-mass rather than equal-width: equal-width bins leave the top bins
    nearly empty at 14% prevalence, which makes the curve look erratic for
    reasons that have nothing to do with calibration.
    """
    y, p = _validate(y, p)
    if n_bins < 2:
        raise ConfigError(f"n_bins must be >= 2, got {n_bins}")
    quantiles = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(quantiles)
    if edges.size < 3:  # degenerate predictor, e.g. a constant
        return {"mean_predicted": [], "observed_rate": [], "count": []}
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, edges.size - 2)

    mean_pred: list[float] = []
    observed: list[float] = []
    counts: list[float] = []
    for b in range(edges.size - 1):
        sel = idx == b
        n = int(sel.sum())
        if n == 0:
            continue
        mean_pred.append(float(p[sel].mean()))
        observed.append(float(y[sel].mean()))
        counts.append(float(n))
    return {"mean_predicted": mean_pred, "observed_rate": observed, "count": counts}


@dataclasses.dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """The full metric set for one predictor."""

    auroc: float
    auprc: float
    brier: float
    nll: float
    calibration_slope: float
    calibration_intercept: float
    prevalence: float
    n: int

    def as_dict(self) -> dict[str, float | int]:
        return dataclasses.asdict(self)


def classification_metrics(y: FloatArray, p: FloatArray) -> ClassificationMetrics:
    y, p = _validate(y, p)
    intercept, slope = _recalibration_fit(y, p)
    return ClassificationMetrics(
        auroc=auroc(y, p),
        auprc=auprc(y, p),
        brier=brier_score(y, p),
        nll=negative_log_likelihood(y, p),
        calibration_slope=slope,
        calibration_intercept=intercept,
        prevalence=float(y.mean()),
        n=int(y.size),
    )


# ------------------------------------------------------------- bootstrap ----
def _patient_resamples(n: int, n_boot: int, seed: int) -> npt.NDArray[np.int64]:
    """Pre-draw all patient-level resample indices.

    Drawn once and reused so that paired comparisons see identical resamples.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n), dtype=np.int64)


def bootstrap_metric(
    y: FloatArray,
    p: FloatArray,
    metric: MetricFn,
    *,
    n_boot: int = 2000,
    seed: int = 20260809,
) -> Interval:
    """Percentile bootstrap CI over patients for one predictor."""
    y, p = _validate(y, p)
    point = float(metric(y, p))
    draws = _patient_resamples(y.size, n_boot, seed)
    stats: list[float] = []
    for idx in draws:
        yb = y[idx]
        if len(np.unique(yb)) < 2:  # a resample with one class is unscoreable
            continue
        stats.append(float(metric(yb, p[idx])))
    if not stats:
        return Interval(point=point, low=float("nan"), high=float("nan"), n_boot=0)
    low, high = np.percentile(stats, [2.5, 97.5])
    return Interval(point=point, low=float(low), high=float(high), n_boot=len(stats))


def paired_bootstrap_difference(
    y: FloatArray,
    p_a: FloatArray,
    p_b: FloatArray,
    metric: MetricFn,
    *,
    metric_name: str,
    name_a: str,
    name_b: str,
    n_boot: int = 2000,
    seed: int = 20260809,
) -> PairedDifference:
    """Bootstrap the difference ``metric(b) - metric(a)`` on identical patients.

    Both predictors are re-scored on the *same* resampled patients, so shared
    patient-level variance cancels. This is the correct test for "does B beat A",
    and it is what the milestone comparisons use.
    """
    y, p_a = _validate(y, p_a)
    _, p_b = _validate(y, p_b)
    point = float(metric(y, p_b)) - float(metric(y, p_a))
    draws = _patient_resamples(y.size, n_boot, seed)

    # Exact equality occurs for controls that are mathematically constrained to
    # produce the same predictions. Re-scoring identical arrays cannot change a
    # paired difference from zero, but sorting them thousands of times for AURC
    # is needlessly expensive. Still count valid patient resamples so n_boot has
    # the same semantics as the general path.
    if np.array_equal(p_a, p_b):
        n_valid = sum(len(np.unique(y[idx])) == 2 for idx in draws)
        return PairedDifference(
            metric=metric_name,
            name_a=name_a,
            name_b=name_b,
            difference=point,
            low=0.0,
            high=0.0,
            n_boot=n_valid,
            excludes_zero=False,
        )

    diffs: list[float] = []
    for idx in draws:
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        diffs.append(float(metric(yb, p_b[idx])) - float(metric(yb, p_a[idx])))
    if not diffs:
        return PairedDifference(
            metric=metric_name,
            name_a=name_a,
            name_b=name_b,
            difference=point,
            low=float("nan"),
            high=float("nan"),
            n_boot=0,
            excludes_zero=False,
        )
    low, high = (float(v) for v in np.percentile(diffs, [2.5, 97.5]))
    return PairedDifference(
        metric=metric_name,
        name_a=name_a,
        name_b=name_b,
        difference=point,
        low=low,
        high=high,
        n_boot=len(diffs),
        excludes_zero=bool(low > 0.0 or high < 0.0),
    )


METRIC_FUNCTIONS: dict[str, MetricFn] = {
    "auroc": auroc,
    "auprc": auprc,
    "brier": brier_score,
    "nll": negative_log_likelihood,
}
