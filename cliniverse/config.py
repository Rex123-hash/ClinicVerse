"""Configuration loading.

Configuration lives in ``configs/*.yaml``, never as constants scattered through
modules. Everything here is frozen after load so a config object cannot be
mutated halfway through an experiment.
"""

from __future__ import annotations

import functools
import pathlib
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from cliniverse.exceptions import ConfigError

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"

VariableKind = Literal["demographic", "context", "vital", "setting", "lab"]
RepairOp = Literal["fahrenheit_to_celsius", "scale", "offset"]


class Repair(BaseModel):
    """A unit-entry correction applied before plausibility filtering."""

    model_config = ConfigDict(frozen=True)

    when: tuple[float, float]
    op: RepairOp
    factor: float | None = None
    offset: float | None = None

    @model_validator(mode="after")
    def _check_operands(self) -> Repair:
        if self.when[0] >= self.when[1]:
            raise ValueError(f"repair `when` must be increasing, got {self.when}")
        if self.op == "scale" and self.factor is None:
            raise ValueError("op=scale requires `factor`")
        if self.op == "offset" and self.offset is None:
            raise ValueError("op=offset requires `offset`")
        return self

    def apply(self, value: float) -> float:
        """Apply this repair to `value`. Caller checks `matches` first."""
        if self.op == "fahrenheit_to_celsius":
            return (value - 32.0) * 5.0 / 9.0
        if self.op == "scale":
            assert self.factor is not None
            return value * self.factor
        assert self.offset is not None
        return value + self.offset

    def matches(self, value: float) -> bool:
        return self.when[0] <= value <= self.when[1]


class VariableSpec(BaseModel):
    """Schema for one measured variable."""

    model_config = ConfigDict(frozen=True)

    name: str
    unit: str
    plausible: tuple[float, float]
    kind: VariableKind

    @model_validator(mode="after")
    def _check_bounds(self) -> VariableSpec:
        lo, hi = self.plausible
        if lo >= hi:
            raise ValueError(
                f"{self.name}: plausible bounds must increase, got {self.plausible}"
            )
        return self

    def is_plausible(self, value: float) -> bool:
        return self.plausible[0] <= value <= self.plausible[1]


class VariableConfig(BaseModel):
    """The full variable schema for a dataset."""

    model_config = ConfigDict(frozen=True)

    dataset: str
    version: str
    horizon_hours: int = Field(gt=0)
    statics: dict[str, VariableSpec]
    variables: dict[str, VariableSpec]
    repairs: dict[str, tuple[Repair, ...]] = Field(default_factory=dict)

    @property
    def variable_names(self) -> tuple[str, ...]:
        """Time-series variable names in a stable, sorted order.

        Sorted rather than file-order so that array column indices are
        reproducible regardless of how the YAML is edited.
        """
        return tuple(sorted(self.variables))

    @property
    def static_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.statics))

    def names_by_kind(self, *kinds: VariableKind) -> tuple[str, ...]:
        wanted = set(kinds)
        return tuple(n for n in self.variable_names if self.variables[n].kind in wanted)

    def clean(self, name: str, value: float) -> float | None:
        """Repair then validate a single raw reading.

        Returns the cleaned value, or ``None`` if it is implausible and must be
        treated as missing. Unknown variable names return ``None`` rather than
        raising: the raw files contain occasional undeclared parameters.
        """
        spec = self.variables.get(name) or self.statics.get(name)
        if spec is None:
            return None
        if value == -1.0 and spec.kind in ("demographic", "context"):
            return None  # dataset's static missing sentinel
        for repair in self.repairs.get(name, ()):
            if repair.matches(value):
                value = repair.apply(value)
                break
        return value if spec.is_plausible(value) else None


def _load_yaml(path: pathlib.Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"malformed YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping, got {type(data).__name__}")
    return data


@functools.cache
def load_variable_config(path: pathlib.Path | None = None) -> VariableConfig:
    """Load and validate ``configs/variables.yaml`` (cached)."""
    path = path or CONFIG_DIR / "variables.yaml"
    raw = _load_yaml(path)
    for section in ("statics", "variables"):
        raw[section] = {
            name: {"name": name, **spec} for name, spec in raw.get(section, {}).items()
        }
    try:
        return VariableConfig.model_validate(raw)
    except ValueError as exc:
        raise ConfigError(f"invalid variable config {path}: {exc}") from exc
