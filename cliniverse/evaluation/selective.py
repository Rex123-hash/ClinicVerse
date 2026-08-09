"""Selective prediction and confidence summaries.

Risk-coverage analysis asks: if the model may decline to predict on the cases it
is least sure about, how fast does its error fall? A model whose confidence is
informative sheds error quickly as coverage drops. One whose confidence is
uninformative does not, even if its ranking is good.

That distinction is the point of M3: discrimination can survive information loss
while the *usefulness of the confidence* does not.

``predictive_entropy`` is named exactly that. A single model's entropy is not
decomposable into epistemic and aleatoric parts, so it is never called epistemic
uncertainty.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import numpy.typing as npt

from cliniverse.exceptions import ConfigError

FloatArray = npt.NDArray[np.float64]

_EPS = 1e-15


def predictive_entropy(p: FloatArray) -> FloatArray:
    """Binary predictive entropy in nats, per prediction."""
    q = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1 - _EPS)
    return np.asarray(-(q * np.log(q) + (1 - q) * np.log1p(-q)), dtype=np.float64)


def confidence(p: FloatArray) -> FloatArray:
    """Distance from the decision boundary: ``|p - 0.5| * 2`` in ``[0, 1]``."""
    return np.asarray(np.abs(np.asarray(p, dtype=np.float64) - 0.5) * 2.0)


@dataclasses.dataclass(frozen=True, slots=True)
class RiskCoverage:
    """A risk-coverage curve and its area."""

    coverage: FloatArray
    risk: FloatArray
    aurc: float

    def as_dict(self) -> dict[str, object]:
        return {
            "coverage": self.coverage.tolist(),
            "risk": self.risk.tolist(),
            "aurc": self.aurc,
        }


def risk_coverage_curve(y: FloatArray, p: FloatArray, *, loss: str = "brier") -> RiskCoverage:
    """Risk as a function of coverage, ordering predictions by confidence.

    Predictions are sorted most-confident first. At coverage ``k/n`` the risk is
    the mean loss over the ``k`` most confident predictions. AURC is the mean
    risk across all coverage levels — lower is better.

    Args:
        loss: ``"brier"`` (squared error) or ``"error"`` (0/1 at threshold 0.5).
            Squared error is the default because it responds to *how wrong* a
            probability is, which is what degrades under information loss.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    if y.shape != p.shape:
        raise ConfigError(f"labels {y.shape} and predictions {p.shape} differ in shape")
    if y.size == 0:
        raise ConfigError("cannot compute risk-coverage on an empty set")

    if loss == "brier":
        losses = (p - y) ** 2
    elif loss == "error":
        losses = ((p >= 0.5).astype(np.float64) != y).astype(np.float64)
    else:
        raise ConfigError(f"unknown loss {loss!r}; use 'brier' or 'error'")

    order = np.argsort(-confidence(p), kind="stable")
    ordered = losses[order]
    n = y.size
    cumulative_risk = np.cumsum(ordered) / np.arange(1, n + 1)
    coverage = np.arange(1, n + 1, dtype=np.float64) / n
    return RiskCoverage(
        coverage=coverage, risk=cumulative_risk, aurc=float(cumulative_risk.mean())
    )


def aurc(y: FloatArray, p: FloatArray) -> float:
    """Area under the risk-coverage curve (Brier loss). Lower is better."""
    return risk_coverage_curve(y, p, loss="brier").aurc


def aurc_error(y: FloatArray, p: FloatArray) -> float:
    """Area under the risk-coverage curve (0/1 loss). Lower is better."""
    return risk_coverage_curve(y, p, loss="error").aurc


def mean_predicted_probability(y: FloatArray, p: FloatArray) -> float:
    """Mean predicted risk. Signature matches the metric protocol; ``y`` unused."""
    del y
    return float(np.mean(np.asarray(p, dtype=np.float64)))


def mean_predictive_entropy(y: FloatArray, p: FloatArray) -> float:
    """Mean predictive entropy in nats. Signature matches the metric protocol."""
    del y
    return float(np.mean(predictive_entropy(np.asarray(p, dtype=np.float64))))


def downsample_curve(curve: RiskCoverage, n_points: int = 100) -> dict[str, list[float]]:
    """Thin a risk-coverage curve for storage without changing its shape."""
    n = curve.coverage.size
    if n <= n_points:
        idx = np.arange(n)
    else:
        idx = np.unique(np.linspace(0, n - 1, n_points).astype(int))
    return {
        "coverage": curve.coverage[idx].tolist(),
        "risk": curve.risk[idx].tolist(),
    }
