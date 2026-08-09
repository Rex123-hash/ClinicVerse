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
    """Schema for one measured variable.

    For statics, ``source_parameter`` names the raw parameter it is read from and
    ``at_hour`` pins the only hour at which that reading may be taken. This is a
    leakage control: without it, a parameter recorded repeatedly through the stay
    (``Weight``) silently delivers post-cutoff values into a "static" feature.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    unit: str
    plausible: tuple[float, float]
    kind: VariableKind
    source_parameter: str | None = None
    at_hour: int | None = Field(default=None, ge=0)

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

    @property
    def static_sources(self) -> dict[str, tuple[str, int]]:
        """Map raw parameter name -> (static field name, the only admissible hour).

        A raw parameter absent from this map can never populate a static field.
        """
        out: dict[str, tuple[str, int]] = {}
        for name, spec in self.statics.items():
            source = spec.source_parameter or name
            hour = 0 if spec.at_hour is None else spec.at_hour
            out[source] = (name, hour)
        return out

    def _clean_with(self, spec: VariableSpec, raw_name: str, value: float) -> float | None:
        if value == -1.0 and spec.kind in ("demographic", "context"):
            return None  # dataset's static missing sentinel
        for repair in self.repairs.get(raw_name, ()):
            if repair.matches(value):
                value = repair.apply(value)
                break
        return value if spec.is_plausible(value) else None

    def clean(self, name: str, value: float) -> float | None:
        """Repair then validate a single raw *time-series* reading.

        Returns the cleaned value, or ``None`` if it is implausible and must be
        treated as missing. Unknown names return ``None`` rather than raising:
        the raw files contain occasional undeclared parameters.
        """
        spec = self.variables.get(name) or self.statics.get(name)
        if spec is None:
            return None
        return self._clean_with(spec, name, value)

    def clean_static(self, raw_name: str, value: float) -> tuple[str, float] | None:
        """Repair and validate a raw reading destined for a static field.

        Returns ``(static_field_name, cleaned_value)``, or ``None`` if the
        parameter is not a static source or the value is implausible. The caller
        is responsible for enforcing the hour constraint from
        :attr:`static_sources`.
        """
        target = self.static_sources.get(raw_name)
        if target is None:
            return None
        field, _ = target
        cleaned = self._clean_with(self.statics[field], raw_name, value)
        return None if cleaned is None else (field, cleaned)


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
