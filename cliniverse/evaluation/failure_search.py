"""Exhaustive search for discrimination-silent reliability failures.

Predeclared in ``docs/M5_DESIGN.md``. The search asks one narrow question: after
controlling for *how much* information is withheld, does it matter *which*
information is withheld?

Two design choices make that question answerable rather than rhetorical.

**Exhaustive enumeration.** The action catalogue partitions the eligible
laboratory analytes into ten co-measurement groups, so the configuration space is
exactly the 1,023 non-empty subsets of those groups. Evaluating all of them means
the selected configuration's standing is an exact rank within a fully known
distribution; no stochastic-search or early-stopping artefact can be mistaken for
a finding, and the multiplicity story needs no assumption.

**Amount matching.** Every configuration is scored against a control that removes
exactly as many cells from the same patient, drawn uniformly from that patient's
observed eligible cells. The reported quantity is always the *excess* over that
control. Without it, "this configuration is damaging" would largely restate "this
configuration removed more cells", which is precisely the error that adversarial repair
#3 corrected in M3.

Withholding is whole-window analyte removal, identical in semantics to M3's
``group_structured`` condition but with the group set *specified* rather than
drawn to hit a severity target. Per M3-B this identifies **analyte-set identity**
and never co-occurrence coherence, and the groups remain reconstructed
co-measurement clusters (``*_like``), never verified laboratory orders.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from cliniverse.acquisition.catalogue import PanelCatalogue
from cliniverse.data.cohort import BoolArray, Cohort
from cliniverse.evaluation.information_loss import (
    LossCondition,
    apply_information_loss,
    eligible_columns,
)
from cliniverse.exceptions import ConfigError

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]

#: A configuration: a sorted tuple of catalogue group names to withhold.
GroupSubset = tuple[str, ...]


def enumerate_group_subsets(group_names: Sequence[str]) -> tuple[GroupSubset, ...]:
    """Every non-empty subset of ``group_names``, in a deterministic order.

    Ordered by size and then lexicographically, so the enumeration index of a
    configuration is stable across runs and machines and can be quoted in an
    artifact without ambiguity.
    """
    names = tuple(sorted(set(group_names)))
    if not names:
        raise ConfigError("cannot enumerate configurations over an empty group set")
    return tuple(
        combination
        for size in range(1, len(names) + 1)
        for combination in itertools.combinations(names, size)
    )


@dataclasses.dataclass(frozen=True, slots=True)
class SubsetLoss:
    """A cohort with one configuration's information withheld."""

    cohort: Cohort
    subset: GroupSubset
    removed_cells: IntArray
    eligible_cells: IntArray
    removed_by_variable: IntArray

    @property
    def realized_severity(self) -> FloatArray:
        """Per-patient fraction of eligible laboratory cells withheld."""
        eligible = self.eligible_cells.astype(np.float64)
        removed = self.removed_cells.astype(np.float64)
        return np.divide(removed, eligible, out=np.zeros_like(removed), where=eligible > 0)

    def summary(self) -> dict[str, float | int | list[str]]:
        severity = self.realized_severity
        scored = severity[self.eligible_cells > 0]
        return {
            "subset": list(self.subset),
            "n_groups": len(self.subset),
            "total_removed_cells": int(self.removed_cells.sum()),
            "total_eligible_cells": int(self.eligible_cells.sum()),
            "mean_removed_cells_per_patient": float(self.removed_cells.mean()),
            "mean_realized_severity": float(scored.mean()) if scored.size else 0.0,
            "median_realized_severity": float(np.median(scored)) if scored.size else 0.0,
        }


def subset_columns(cohort: Cohort, subset: GroupSubset, catalogue: PanelCatalogue) -> IntArray:
    """Cohort column indices covered by a configuration's groups."""
    if not subset:
        raise ConfigError("a configuration must name at least one group")
    _, groups = eligible_columns(cohort, catalogue)
    unknown = sorted(set(subset) - set(groups))
    if unknown:
        raise ConfigError(
            f"configuration names groups absent from this cohort/catalogue: {unknown}"
        )
    if len(set(subset)) != len(subset):
        raise ConfigError(f"configuration repeats a group: {subset}")
    return np.array(sorted({int(c) for name in subset for c in groups[name]}), dtype=np.int64)


def subset_removal_mask(
    cohort: Cohort, subset: GroupSubset, catalogue: PanelCatalogue
) -> BoolArray:
    """``(N, T, V)`` mask of the observed cells a configuration withholds.

    Deterministic: no RNG, no severity target, no per-patient draw. Every
    observed cell of every member analyte is withheld across the whole window.
    """
    columns = subset_columns(cohort, subset, catalogue)
    removal: BoolArray = np.zeros_like(cohort.m)
    removal[:, :, columns] = True
    removal &= cohort.m
    return removal


def apply_group_subset_loss(
    cohort: Cohort, subset: GroupSubset, catalogue: PanelCatalogue
) -> SubsetLoss:
    """Withhold every observed cell of a configuration's analytes.

    Values become NaN and observation-mask cells become false, so a withheld cell
    is indistinguishable from one that was never measured. Loss is applied to the
    cohort, before feature construction; there is no post-hoc feature deletion.
    """
    removal = subset_removal_mask(cohort, subset, catalogue)
    columns, _ = eligible_columns(cohort, catalogue)

    x = cohort.x.copy()
    m = cohort.m.copy()
    x[removal] = np.nan
    m[removal] = False

    return SubsetLoss(
        cohort=dataclasses.replace(cohort, x=x, m=m),
        subset=tuple(subset),
        removed_cells=removal.sum(axis=(1, 2)).astype(np.int64),
        eligible_cells=cohort.m[:, :, columns].sum(axis=(1, 2)).astype(np.int64),
        removed_by_variable=removal.sum(axis=1).astype(np.int64),
    )


def control_seed(base_seed: int, subset: GroupSubset, repetition: int) -> int:
    """Deterministic, well-separated seed for one control draw.

    A linear stride would risk colliding with the per-patient seed arithmetic
    inside ``information_loss`` once a thousand configurations are in play, which
    would silently correlate control draws across configurations. Hashing the
    identity of the draw avoids that and stays reproducible across machines.
    """
    key = f"{base_seed}|{'+'.join(subset)}|{repetition}".encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


def matched_random_control(
    cohort: Cohort,
    removed_cells: IntArray,
    catalogue: PanelCatalogue,
    *,
    seed: int,
) -> Cohort:
    """Remove the same number of cells per patient, drawn uniformly at random.

    This delegates to the existing, tested, review-audited ``CELL_RANDOM`` path
    with ``match_counts`` supplied; M5 introduces no new sampling logic for the
    control. The ``severity`` argument is ignored whenever ``match_counts`` is
    given, and is passed as zero to make that explicit.

    The draw covers **all** eligible laboratory analytes, including those the
    configuration withheld. That is the M3 count-matched semantics, retained
    unchanged.
    """
    outcome = apply_information_loss(
        cohort,
        LossCondition.CELL_RANDOM,
        0.0,
        catalogue,
        seed=seed,
        match_counts=removed_cells,
    )
    if not np.array_equal(outcome.removed_cells, removed_cells):
        mismatch = int((outcome.removed_cells != removed_cells).sum())
        raise ConfigError(
            f"amount matching failed for {mismatch} patients; the excess statistic "
            "would confound analyte identity with the amount removed"
        )
    return outcome.cohort


@dataclasses.dataclass(frozen=True, slots=True)
class ConfigurationScore:
    """One configuration's scores on one patient set."""

    index: int
    subset: GroupSubset
    n_groups: int
    mean_removed_cells: float
    mean_realized_severity: float
    delta_nll_excess: float
    delta_brier_excess: float
    nll: float
    nll_control: float
    brier: float
    brier_control: float
    auroc: float
    #: Diagnostic only: AUROC of the final control draw, not an average over the
    #: repetitions. The primary and co-primary excesses do average over draws.
    auroc_control_last_draw: float
    auprc: float
    calibration_intercept: float
    calibration_slope: float
    mean_predicted_risk: float

    def as_dict(self) -> dict[str, object]:
        payload = dataclasses.asdict(self)
        payload["subset"] = list(self.subset)
        return payload


def select_top_k(
    discovery: Sequence[ConfigurationScore],
    *,
    k: int,
    clean_auroc: float,
    delta: float,
) -> tuple[GroupSubset, ...]:
    """Lock the top ``k`` configurations by discovery excess NLL.

    This function deliberately accepts **only** discovery-set scores. The
    selection cannot consult a confirmation number even by accident, which is
    what makes the lock in ``M5_DESIGN.md`` §9.1 auditable rather than merely
    promised.

    Eligibility is the AUROC-preservation constraint: a configuration that
    destroys discrimination would be caught by ordinary monitoring and is not the
    phenomenon under study.
    """
    if k <= 0:
        raise ConfigError(f"k must be positive, got {k}")
    if not 0.0 <= delta <= 1.0:
        raise ConfigError(f"delta must be in [0, 1], got {delta}")
    eligible = [s for s in discovery if clean_auroc - s.auroc <= delta]
    if not eligible:
        raise ConfigError(
            f"no configuration preserved AUROC within delta={delta} of the clean "
            f"reference {clean_auroc:.4f}; the search space cannot answer the "
            "predeclared question"
        )
    # Ties broken by enumeration index so the lock is reproducible.
    ordered = sorted(eligible, key=lambda s: (-s.delta_nll_excess, s.index))
    return tuple(s.subset for s in ordered[: min(k, len(ordered))])


def spearman_permutation_test(
    a: FloatArray,
    b: FloatArray,
    *,
    n_permutations: int,
    seed: int,
) -> dict[str, float | int]:
    """One-sided permutation test for a positive Spearman correlation.

    Configurations share groups, so they are not independent and a bootstrap
    interval over configurations would be approximate. Permuting one vector is
    exact under the null of no association between discovery and confirmation
    rankings, which is the hypothesis T4 actually tests.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ConfigError(f"vectors differ in shape: {a.shape} vs {b.shape}")
    if a.size < 3:
        raise ConfigError("a rank correlation needs at least three configurations")

    rank_a = _rankdata(a)
    rank_b = _rankdata(b)
    observed = _pearson(rank_a, rank_b)

    rng = np.random.default_rng(seed)
    at_least_as_extreme = 0
    for _ in range(n_permutations):
        if _pearson(rank_a, rng.permutation(rank_b)) >= observed:
            at_least_as_extreme += 1
    # Add-one correction: a permutation p-value is never exactly zero.
    p_value = (at_least_as_extreme + 1) / (n_permutations + 1)
    return {
        "spearman_rho": float(observed),
        "p_value_one_sided": float(p_value),
        "n_permutations": int(n_permutations),
    }


def _rankdata(values: FloatArray) -> FloatArray:
    """Average ranks, matching ``scipy.stats.rankdata`` for tied values."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    for stop in range(1, values.size + 1):
        if stop == values.size or sorted_values[stop] != sorted_values[start]:
            if stop - start > 1:
                ranks[order[start:stop]] = ranks[order[start:stop]].mean()
            start = stop
    return ranks


def _pearson(a: FloatArray, b: FloatArray) -> float:
    a_centred = a - a.mean()
    b_centred = b - b.mean()
    denominator = float(np.sqrt((a_centred**2).sum() * (b_centred**2).sum()))
    if denominator == 0.0:
        return 0.0
    return float((a_centred * b_centred).sum() / denominator)


def holm_bonferroni(p_values: Sequence[float], *, alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down rejections for a small family of tests."""
    indexed = sorted(enumerate(p_values), key=lambda pair: pair[1])
    n = len(p_values)
    rejected = [False] * n
    for step, (position, p_value) in enumerate(indexed):
        if p_value <= alpha / (n - step):
            rejected[position] = True
        else:
            break
    return rejected
