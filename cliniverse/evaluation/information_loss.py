"""Structured information loss applied to a cohort before feature construction.

Loss is applied to the values and the observation mask, not to summary columns,
so information genuinely disappears rather than being edited out afterwards. A
removed cell is indistinguishable from one that was never measured.

Two mechanisms and two controls:

``variable_matched_scattered`` matches the structured mask's per-patient,
per-variable counts and draws independently within each variable.

``cell_random``       remove individual observed laboratory cells at random.
``group_structured``  remove entire co-measurement groups — every observed cell
                      of that group across the whole window.

**Matched severity.** "Group loss is worse" would be uninteresting if group loss
simply removed more cells. Matching is therefore done *per patient*: group loss
runs first and records exactly how many cells it removed for that patient, and
the matched cell condition removes exactly that many. This controls amount but
not variable identity. Exact variable matching controls both; under the current
whole-window semantics it necessarily degenerates to the structured mask.

Eligible information is restricted to laboratory variables covered by the
co-measurement catalogue. Vitals and ventilator settings are never removed: they
are near-continuously monitored, so deleting them would not correspond to a
group-level acquisition event.

Terminology: these are **structured group-level information loss** conditions over
co-measurement groups. They are not verified clinical orders.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Final

import numpy as np
import numpy.typing as npt

from cliniverse.acquisition.catalogue import PanelCatalogue
from cliniverse.data.cohort import BoolArray, Cohort
from cliniverse.exceptions import ConfigError

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]

#: Mixed into per-patient seeds so loss draws cannot collide with split or
#: masking-mechanism draws that use the same base seed.
_SEED_SALT: Final = 0x4D33_0001


class LossCondition(enum.StrEnum):
    """How information is removed."""

    NONE = "none"
    CELL_RANDOM = "cell_random"
    GROUP_STRUCTURED = "group_structured"
    VARIABLE_MATCHED_SCATTERED = "variable_matched_scattered"


@dataclasses.dataclass(frozen=True, slots=True)
class LossOutcome:
    """A stressed cohort plus the realized severity actually achieved."""

    cohort: Cohort
    condition: LossCondition
    requested_severity: float
    removed_cells: IntArray
    eligible_cells: IntArray
    removed_by_variable: IntArray
    match_mismatch_cells: IntArray

    @property
    def realized_severity(self) -> FloatArray:
        """Per-patient fraction of eligible cells removed."""
        eligible = self.eligible_cells.astype(np.float64)
        removed = self.removed_cells.astype(np.float64)
        return np.divide(removed, eligible, out=np.zeros_like(removed), where=eligible > 0)

    def summary(self) -> dict[str, str | float | int]:
        realized = self.realized_severity
        eligible = self.eligible_cells
        scored = realized[eligible > 0]
        return {
            "condition": str(self.condition),
            "requested_severity": self.requested_severity,
            "n_patients": len(self.removed_cells),
            "n_patients_with_eligible_cells": int((eligible > 0).sum()),
            "total_eligible_cells": int(eligible.sum()),
            "total_removed_cells": int(self.removed_cells.sum()),
            "n_patients_with_match_mismatch": int((self.match_mismatch_cells > 0).sum()),
            "total_match_mismatch_cells": int(self.match_mismatch_cells.sum()),
            "realized_severity_mean": float(scored.mean()) if scored.size else 0.0,
            "realized_severity_median": float(np.median(scored)) if scored.size else 0.0,
            "realized_severity_p10": float(np.percentile(scored, 10)) if scored.size else 0.0,
            "realized_severity_p90": float(np.percentile(scored, 90)) if scored.size else 0.0,
        }


def eligible_columns(
    cohort: Cohort, catalogue: PanelCatalogue
) -> tuple[IntArray, dict[str, IntArray]]:
    """Column indices eligible for removal, and the columns of each group.

    Returns ``(all_eligible_columns, {group_name: columns})``. Only variables
    that exist in the cohort are included.
    """
    index = {name: i for i, name in enumerate(cohort.variable_names)}
    groups: dict[str, IntArray] = {}
    for name, panel in catalogue.panels.items():
        cols = [index[m] for m in panel.members if m in index]
        if cols:
            groups[name] = np.array(sorted(cols), dtype=np.int64)
    if not groups:
        raise ConfigError("catalogue covers no variables present in this cohort")
    all_cols = np.array(
        sorted({int(c) for cols in groups.values() for c in cols}), dtype=np.int64
    )
    return all_cols, groups


def _rng(seed: int, patient_index: int) -> np.random.Generator:
    return np.random.default_rng((seed ^ _SEED_SALT) + 1_000_003 * patient_index)


def _group_removal(
    observed: BoolArray,
    groups: dict[str, IntArray],
    severity: float,
    rng: np.random.Generator,
) -> tuple[BoolArray, int]:
    """Remove whole groups to approach ``severity`` as closely as possible.

    Groups are indivisible, so the realized fraction cannot generally equal the
    request. A naive "remove until the target is reached" rule overshoots badly —
    a single large group can take a requested 25% past 45% — which would make the
    severity ladder mean something quite different from its label.

    Instead each candidate group is accepted only when including it brings the
    removed count *closer* to the target than stopping would. Realized severity
    is still recorded per patient, and the matched cell condition matches the
    realized count rather than the request, so the pairing is exact either way.
    """
    removal = np.zeros_like(observed)
    eligible_total = sum(int(observed[:, cols].sum()) for cols in groups.values())
    if eligible_total == 0 or severity <= 0.0:
        return removal, 0

    present = [name for name, cols in groups.items() if bool(observed[:, cols].any())]
    order = rng.permutation(np.array(sorted(present), dtype=object))

    removed = 0
    target = severity * eligible_total
    for name in order:
        cols = groups[str(name)]
        block = np.zeros_like(observed)
        block[:, cols] = True
        newly = block & observed & ~removal
        gain = int(newly.sum())
        if gain == 0:
            continue
        # Accept only if it lands nearer the target than stopping here would.
        if abs((removed + gain) - target) >= abs(removed - target):
            continue
        removal |= newly
        removed += gain
    return removal, removed


def _cell_removal(
    observed: BoolArray, columns: IntArray, n_remove: int, rng: np.random.Generator
) -> tuple[BoolArray, int]:
    """Remove exactly ``n_remove`` observed cells drawn uniformly from ``columns``."""
    removal = np.zeros_like(observed)
    if n_remove <= 0:
        return removal, 0
    pool = np.zeros_like(observed)
    pool[:, columns] = True
    rows, cols = np.nonzero(pool & observed)
    if rows.size == 0:
        return removal, 0
    take = min(int(n_remove), int(rows.size))
    chosen = rng.choice(rows.size, size=take, replace=False)
    removal[rows[chosen], cols[chosen]] = True
    return removal, take


def _variable_matched_removal(
    observed: BoolArray,
    columns: IntArray,
    requested_by_variable: IntArray,
    rng: np.random.Generator,
) -> tuple[BoolArray, int, int]:
    """Scatter exact per-variable counts across observed timestamps.

    Counts are clipped only when a caller asks for more cells than naturally
    exist for a variable. The shortfall is returned explicitly; the M3 matched
    trio treats any such shortfall as an error because its reference counts come
    from the same patient's structured mask and must therefore be feasible.
    """
    removal = np.zeros_like(observed)
    removed = 0
    mismatch = 0
    for col in columns:
        column = int(col)
        requested = int(requested_by_variable[column])
        if requested <= 0:
            continue
        rows = np.flatnonzero(observed[:, column])
        take = min(requested, int(rows.size))
        if take:
            chosen = rng.choice(rows, size=take, replace=False)
            removal[chosen, column] = True
            removed += take
        mismatch += requested - take
    return removal, removed, mismatch


def apply_information_loss(
    cohort: Cohort,
    condition: LossCondition,
    severity: float,
    catalogue: PanelCatalogue,
    *,
    seed: int,
    match_counts: IntArray | None = None,
    match_variable_counts: IntArray | None = None,
) -> LossOutcome:
    """Remove information from ``cohort`` and return the stressed cohort.

    Args:
        condition: which mechanism to apply.
        severity: requested fraction of eligible cells to remove, in ``[0, 1]``.
        catalogue: defines which variables are eligible and how they group.
        seed: base seed; per-patient draws are derived from it and the index.
        match_counts: for ``CELL_RANDOM``, the exact per-patient number of cells
            to remove. Supplying the counts produced by a ``GROUP_STRUCTURED``
            run is what makes the two conditions severity-matched.
        match_variable_counts: for ``VARIABLE_MATCHED_SCATTERED``, exact counts
            indexed by patient and original cohort variable.
    """
    if not 0.0 <= severity <= 1.0:
        raise ConfigError(f"severity must be in [0, 1], got {severity}")
    columns, groups = eligible_columns(cohort, catalogue)

    eligible = np.array(
        [int(cohort.m[i][:, columns].sum()) for i in range(cohort.n_patients)],
        dtype=np.int64,
    )
    if condition is LossCondition.NONE:
        return LossOutcome(
            cohort=cohort,
            condition=condition,
            requested_severity=0.0,
            removed_cells=np.zeros(cohort.n_patients, dtype=np.int64),
            eligible_cells=eligible,
            removed_by_variable=np.zeros(
                (cohort.n_patients, len(cohort.variable_names)), dtype=np.int64
            ),
            match_mismatch_cells=np.zeros(cohort.n_patients, dtype=np.int64),
        )

    if (
        condition is LossCondition.CELL_RANDOM
        and match_counts is not None
        and len(match_counts) != cohort.n_patients
    ):
        raise ConfigError(
            f"match_counts has {len(match_counts)} entries for {cohort.n_patients} patients"
        )

    expected_variable_shape = (cohort.n_patients, len(cohort.variable_names))
    if condition is LossCondition.VARIABLE_MATCHED_SCATTERED:
        if match_variable_counts is None:
            raise ConfigError("variable-matched scattered loss requires match_variable_counts")
        if match_variable_counts.shape != expected_variable_shape:
            raise ConfigError(
                "match_variable_counts has shape "
                f"{match_variable_counts.shape}, expected {expected_variable_shape}"
            )

    x = cohort.x.copy()
    m = cohort.m.copy()
    removed = np.zeros(cohort.n_patients, dtype=np.int64)
    removed_by_variable = np.zeros(expected_variable_shape, dtype=np.int64)
    mismatch_cells = np.zeros(cohort.n_patients, dtype=np.int64)

    for i in range(cohort.n_patients):
        rng = _rng(seed, i)
        observed = cohort.m[i]
        if condition is LossCondition.GROUP_STRUCTURED:
            removal, n = _group_removal(observed, groups, severity, rng)
        elif condition is LossCondition.VARIABLE_MATCHED_SCATTERED:
            assert match_variable_counts is not None
            removal, n, mismatch = _variable_matched_removal(
                observed, columns, match_variable_counts[i], rng
            )
            mismatch_cells[i] = mismatch
        else:
            n_target = (
                int(match_counts[i])
                if match_counts is not None
                else round(float(severity * eligible[i]))
            )
            removal, n = _cell_removal(observed, columns, n_target, rng)
        if n:
            x[i][removal] = np.nan
            m[i][removal] = False
        removed[i] = n
        removed_by_variable[i] = removal.sum(axis=0, dtype=np.int64)

    stressed = dataclasses.replace(cohort, x=x, m=m)
    return LossOutcome(
        cohort=stressed,
        condition=condition,
        requested_severity=severity,
        removed_cells=removed,
        eligible_cells=eligible,
        removed_by_variable=removed_by_variable,
        match_mismatch_cells=mismatch_cells,
    )


def matched_pair(
    cohort: Cohort,
    severity: float,
    catalogue: PanelCatalogue,
    *,
    seed: int,
) -> tuple[LossOutcome, LossOutcome]:
    """Build a severity-matched (group, cell) pair for the same patients.

    Group loss runs first; the matched cell condition then removes exactly the
    same number of cells per patient. This is the control that prevents "group
    loss is worse" from being explained by "group loss removed more".
    """
    group = apply_information_loss(
        cohort, LossCondition.GROUP_STRUCTURED, severity, catalogue, seed=seed
    )
    cell = apply_information_loss(
        cohort,
        LossCondition.CELL_RANDOM,
        severity,
        catalogue,
        seed=seed + 1,
        match_counts=group.removed_cells,
    )
    if not np.array_equal(group.removed_cells, cell.removed_cells):
        mismatch = int((group.removed_cells != cell.removed_cells).sum())
        raise ConfigError(
            f"severity matching failed for {mismatch} patients; the paired "
            "comparison would confound structure with amount"
        )
    return group, cell


def matched_trio(
    cohort: Cohort,
    severity: float,
    catalogue: PanelCatalogue,
    *,
    seed: int,
) -> tuple[LossOutcome, LossOutcome, LossOutcome]:
    """Build structured, variable-matched, and count-matched loss variants.

    The variable-matched control removes exactly the structured condition's
    number of cells from every corresponding variable for the same patient.
    Under M3's current semantics, structured loss removes every timestamp of a
    selected variable, so exact matching has zero scattering freedom and the
    two masks are identical. Encoding that result is scientifically important:
    it prevents the count-matched contrast from being misread as a pure
    co-occurrence-structure effect.
    """
    group = apply_information_loss(
        cohort, LossCondition.GROUP_STRUCTURED, severity, catalogue, seed=seed
    )
    variable = apply_information_loss(
        cohort,
        LossCondition.VARIABLE_MATCHED_SCATTERED,
        severity,
        catalogue,
        seed=seed + 2,
        match_variable_counts=group.removed_by_variable,
    )
    cell = apply_information_loss(
        cohort,
        LossCondition.CELL_RANDOM,
        severity,
        catalogue,
        seed=seed + 1,
        match_counts=group.removed_cells,
    )
    if not np.array_equal(group.removed_cells, cell.removed_cells):
        mismatch = int((group.removed_cells != cell.removed_cells).sum())
        raise ConfigError(f"count matching failed for {mismatch} patients")
    if not np.array_equal(group.removed_by_variable, variable.removed_by_variable):
        mismatch = int(
            np.any(group.removed_by_variable != variable.removed_by_variable, axis=1).sum()
        )
        raise ConfigError(f"variable matching failed for {mismatch} patients")
    if bool(variable.match_mismatch_cells.any()):
        raise ConfigError(
            "variable matching required unavailable cells for "
            f"{int((variable.match_mismatch_cells > 0).sum())} patients"
        )
    return group, variable, cell
