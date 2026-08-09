"""Post-hoc calibrators, fitted strictly on an isolated calibration partition.

M2's calibration diagnostics were descriptive: the recalibration was fitted on
the same out-of-fold labels it was reported against. M3 separates the two. A
calibrator here is fitted on a partition that trained neither the model nor its
preprocessing, and is then applied to a test partition that trained nothing at
all.

The ladder is deliberately short. Platt scaling is the defensible default for a
gradient-boosted classifier; isotonic is included but guarded by a minimum
calibration-set size, because an unconstrained step function on a small sample
overfits badly and would flatter itself. Temperature scaling is omitted: it is a
single-parameter special case of Platt for this model class and would add a row
without adding information.
"""

from __future__ import annotations

import abc
import dataclasses
import enum

import numpy as np
import numpy.typing as npt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from cliniverse.exceptions import ConfigError

FloatArray = npt.NDArray[np.float64]

_EPS = 1e-15

#: Below this many calibration points, isotonic regression is refused rather
#: than fitted badly.
MIN_ISOTONIC_CALIBRATION_N = 500


class CalibratorKind(enum.StrEnum):
    IDENTITY = "uncalibrated"
    PLATT = "platt"
    ISOTONIC = "isotonic"


def _logit(p: FloatArray) -> FloatArray:
    q = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1 - _EPS)
    return np.asarray(np.log(q / (1 - q)), dtype=np.float64)


@dataclasses.dataclass
class Calibrator(abc.ABC):
    """Maps raw model probabilities to calibrated probabilities."""

    kind: CalibratorKind

    @abc.abstractmethod
    def fit(self, p: FloatArray, y: FloatArray) -> Calibrator: ...

    @abc.abstractmethod
    def transform(self, p: FloatArray) -> FloatArray: ...

    def config(self) -> dict[str, object]:
        return {"kind": str(self.kind)}


@dataclasses.dataclass
class IdentityCalibrator(Calibrator):
    """No calibration. The raw model, reported alongside calibrated variants."""

    kind: CalibratorKind = CalibratorKind.IDENTITY

    def fit(self, p: FloatArray, y: FloatArray) -> IdentityCalibrator:
        del p, y
        return self

    def transform(self, p: FloatArray) -> FloatArray:
        return np.asarray(p, dtype=np.float64)


@dataclasses.dataclass
class PlattCalibrator(Calibrator):
    """Logistic calibration on the predicted log-odds."""

    kind: CalibratorKind = CalibratorKind.PLATT
    slope: float = 1.0
    intercept: float = 0.0
    fitted: bool = False

    def fit(self, p: FloatArray, y: FloatArray) -> PlattCalibrator:
        y = np.asarray(y, dtype=np.float64).ravel()
        x = _logit(p).reshape(-1, 1)
        if len(np.unique(y)) < 2:
            raise ConfigError("calibration partition contains a single class")
        model = LogisticRegression(C=1e12, solver="lbfgs", max_iter=1000)
        model.fit(x, y.astype(int))
        self.slope = float(model.coef_[0][0])
        self.intercept = float(model.intercept_[0])
        self.fitted = True
        return self

    def transform(self, p: FloatArray) -> FloatArray:
        if not self.fitted:
            raise ConfigError("PlattCalibrator used before fit()")
        z = self.intercept + self.slope * _logit(p)
        return np.asarray(1.0 / (1.0 + np.exp(-z)), dtype=np.float64)

    def config(self) -> dict[str, object]:
        return {"kind": str(self.kind), "slope": self.slope, "intercept": self.intercept}


@dataclasses.dataclass
class IsotonicCalibrator(Calibrator):
    """Monotone non-parametric calibration, clipped to the fitted range."""

    kind: CalibratorKind = CalibratorKind.ISOTONIC
    model: IsotonicRegression | None = None
    n_calibration: int = 0

    def fit(self, p: FloatArray, y: FloatArray) -> IsotonicCalibrator:
        p = np.asarray(p, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        if p.size < MIN_ISOTONIC_CALIBRATION_N:
            raise ConfigError(
                f"isotonic calibration needs at least {MIN_ISOTONIC_CALIBRATION_N} "
                f"points, got {p.size}"
            )
        if len(np.unique(y)) < 2:
            raise ConfigError("calibration partition contains a single class")
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.model.fit(p, y)
        self.n_calibration = int(p.size)
        return self

    def transform(self, p: FloatArray) -> FloatArray:
        if self.model is None:
            raise ConfigError("IsotonicCalibrator used before fit()")
        out = self.model.predict(np.asarray(p, dtype=np.float64))
        # Isotonic can emit exact 0 or 1, which makes log-loss infinite.
        return np.clip(np.asarray(out, dtype=np.float64), _EPS, 1 - _EPS)

    def config(self) -> dict[str, object]:
        return {"kind": str(self.kind), "n_calibration": self.n_calibration}


def build_calibrator(kind: CalibratorKind) -> Calibrator:
    if kind is CalibratorKind.IDENTITY:
        return IdentityCalibrator()
    if kind is CalibratorKind.PLATT:
        return PlattCalibrator()
    if kind is CalibratorKind.ISOTONIC:
        return IsotonicCalibrator()
    raise ConfigError(f"unknown calibrator {kind!r}")
