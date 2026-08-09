"""Co-measured analyte groups ("panel-like subsets") — what may be acquired, and
at what relative cost.

A policy selects a feature-group action, paying one cost and receiving every
disclosable member value. Group-level acquisition with shared cost is **not novel** — see
Yu et al., ICLR 2023 (arXiv:2302.10261), which performs sequential panel-level
selection with shared group costs. We adopt the setting; we do not claim it.

Naming caveat: PhysioNet 2012 records analytes with timestamps, not laboratory
orders. These groups are co-measurement clusters derived empirically (see
``experiments/baselines/derive_panels.py`` and ``docs/EXPERIMENTS.md`` E-001),
so they are named ``*_like`` and must be described that way.

Costs are dimensionless relative units and a declared modeling assumption — never
real prices. Four regimes exist so that every conclusion can be tested for
sensitivity to them; see ``configs/panels.yaml``.
"""

from __future__ import annotations

import functools
import math
import pathlib
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cliniverse.config import CONFIG_DIR, VariableConfig, _load_yaml
from cliniverse.exceptions import ConfigError

DEFAULT_SCHEDULE = "default"


class Panel(BaseModel):
    """One benchmark action: a feature group returned for a single relative cost."""

    model_config = ConfigDict(frozen=True)

    name: str
    label: str
    members: tuple[str, ...] = Field(min_length=1)
    cost: float = Field(gt=0, allow_inf_nan=False)
    within_panel_jaccard: float | None = Field(default=None, ge=0, le=1)
    tier: str | None = None

    @model_validator(mode="after")
    def _check_members_unique(self) -> Panel:
        if len(set(self.members)) != len(self.members):
            raise ValueError(f"panel {self.name}: duplicate members {self.members}")
        return self

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def is_singleton(self) -> bool:
        return self.size == 1


class PanelCatalogue(BaseModel):
    """The full set of panel-like feature-group actions under one cost schedule."""

    model_config = ConfigDict(frozen=True)

    version: str
    panels: dict[str, Panel]
    schedule_name: str = DEFAULT_SCHEDULE
    default_regime: str | None = None
    alternative_schedules: dict[str, dict[str, float]] = Field(default_factory=dict)
    derived_from: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_partition(self) -> PanelCatalogue:
        """Every analyte must belong to exactly one panel.

        Overlapping panels would make "what did this purchase reveal?" ambiguous
        and would let a policy double-pay or double-count an analyte.
        """
        seen: dict[str, str] = {}
        for panel in self.panels.values():
            for member in panel.members:
                if member in seen:
                    raise ValueError(
                        f"analyte {member!r} appears in both {seen[member]!r} "
                        f"and {panel.name!r}; panels must partition the analytes"
                    )
                seen[member] = panel.name
        return self

    # ------------------------------------------------------------- lookup ----
    @functools.cached_property
    def _owner(self) -> dict[str, str]:
        return {m: p.name for p in self.panels.values() for m in p.members}

    @property
    def panel_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.panels))

    @property
    def covered_variables(self) -> tuple[str, ...]:
        return tuple(sorted(self._owner))

    def panel_for(self, variable: str) -> Panel:
        """The panel that must be purchased to observe ``variable``."""
        name = self._owner.get(variable)
        if name is None:
            raise ConfigError(f"variable {variable!r} is not in any panel")
        return self.panels[name]

    def cost_of(self, panel_name: str) -> float:
        try:
            return self.panels[panel_name].cost
        except KeyError as exc:
            raise ConfigError(f"unknown panel {panel_name!r}") from exc

    def total_cost(self, panel_names: tuple[str, ...]) -> float:
        """Cost of purchasing a set of panels. Duplicates are paid for once.

        Buying the same panel twice within one acquisition step reveals nothing
        new, so charging twice would misrepresent the policy's spend.
        """
        return sum(self.cost_of(p) for p in set(panel_names))

    # ---------------------------------------------------------- schedules ----
    def with_schedule(self, name: str) -> PanelCatalogue:
        """Return this catalogue re-priced under an alternative cost schedule."""
        if name == DEFAULT_SCHEDULE:
            return self
        if name not in self.alternative_schedules:
            raise ConfigError(
                f"unknown cost schedule {name!r}; "
                f"available: {sorted(self.alternative_schedules)}"
            )
        costs = self.alternative_schedules[name]
        missing = set(self.panels) - set(costs)
        if missing:
            raise ConfigError(
                f"cost schedule {name!r} is missing prices for {sorted(missing)}"
            )
        invalid = {
            panel: cost
            for panel, cost in costs.items()
            if not math.isfinite(cost) or cost <= 0
        }
        if invalid:
            raise ConfigError(
                f"cost schedule {name!r} has non-positive or non-finite prices: {invalid}"
            )
        return self.model_copy(
            update={
                "panels": {
                    key: panel.model_copy(update={"cost": costs[key]})
                    for key, panel in self.panels.items()
                },
                "schedule_name": name,
            }
        )

    # --------------------------------------------------------- validation ----
    def validate_against(self, variables: VariableConfig) -> None:
        """Check the catalogue against the variable schema.

        Raises:
            ConfigError: if a panel references an unknown variable, or if a
                laboratory variable is not purchasable through any panel.
        """
        known = set(variables.variable_names)
        unknown = set(self._owner) - known
        if unknown:
            raise ConfigError(f"catalogue references unknown variables: {sorted(unknown)}")

        labs = set(variables.names_by_kind("lab"))
        uncovered = labs - set(self._owner)
        if uncovered:
            raise ConfigError(
                f"laboratory variables not purchasable via any panel: {sorted(uncovered)}"
            )
        non_lab = set(self._owner) - labs
        if non_lab:
            raise ConfigError(
                "catalogue contains non-laboratory variables, which are "
                f"continuously monitored and not orderable: {sorted(non_lab)}"
            )


@functools.cache
def load_panel_catalogue(path: pathlib.Path | None = None) -> PanelCatalogue:
    """Load and validate ``configs/panels.yaml`` (cached)."""
    path = path or CONFIG_DIR / "panels.yaml"
    raw = _load_yaml(path)
    raw["panels"] = {
        name: {"name": name, **spec} for name, spec in raw.get("panels", {}).items()
    }
    alternatives = raw.pop("alternative_schedules", {}) or {}
    raw["alternative_schedules"] = {key: value["costs"] for key, value in alternatives.items()}
    # The per-panel `cost` entries already instantiate `default_regime`, so the
    # loaded catalogue is *named* for that regime. Leaving it as the generic
    # "default" would make re-pricing to the regime it already uses look like a
    # change of schedule.
    if raw.get("default_regime"):
        raw["schedule_name"] = raw["default_regime"]
    try:
        return PanelCatalogue.model_validate(raw)
    except ValueError as exc:
        raise ConfigError(f"invalid panel catalogue {path}: {exc}") from exc
