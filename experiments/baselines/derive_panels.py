"""Derive the laboratory feature-group catalogue from co-presence structure.

The group-level acquisition framing rests on a factual claim: laboratory
analytes in this dataset are recorded together in clusters within hourly bins.
This co-presence must be measured, not assumed, and does not identify real
orders or specimens.

Method. For every (patient, hour) cell in which at least one laboratory analyte
was measured, record which analytes were measured. For each variable pair (i, j)
compute:

    jaccard(i, j)   = P(i and j measured | i or j measured)
    conditional(j|i) = P(j measured | i measured)

Then agglomeratively cluster analytes on the Jaccard distance. If the clusters
recover recognised clinical panels (BMP, CBC, LFT, ABG) without being told about
them, the panel model is empirically supported. If they do not, the framing is
wrong and we need to know before building on it.

Usage:
    python experiments/baselines/derive_panels.py [--sets a b] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from cliniverse.config import load_variable_config
from cliniverse.data import load_cohort
from cliniverse.data.cohort import Cohort
from cliniverse.log import get_logger

log = get_logger(__name__)


def co_measurement_matrices(
    cohort: Cohort, lab_names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (jaccard, conditional, marginal_cell_counts) over lab analytes.

    ``conditional[i, j] = P(j measured | i measured)`` in the same hour-cell.
    """
    cols = [cohort.variable_index(n) for n in lab_names]
    # (n_patients, n_hours, n_labs) -> flatten patient/hour into active lab bins.
    mask = cohort.m[:, :, cols].reshape(-1, len(cols))
    # Keep patient-hours where at least one lab was recorded.
    active = mask[mask.any(axis=1)]
    log.info("active lab patient-hours", n_hours=int(active.shape[0]), n_labs=len(lab_names))

    counts = active.astype(np.int64)
    both = counts.T @ counts  # |i and j|
    marginal = np.diag(both).astype(np.float64)  # |i|
    either = marginal[:, None] + marginal[None, :] - both

    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = np.where(either > 0, both / either, 0.0)
        conditional = np.where(marginal[:, None] > 0, both / marginal[:, None], 0.0)
    return jaccard, conditional, marginal


def cluster_panels(
    jaccard: np.ndarray, lab_names: tuple[str, ...], threshold: float
) -> dict[int, list[str]]:
    """Agglomeratively cluster analytes on Jaccard distance (average linkage)."""
    distance = 1.0 - jaccard
    np.fill_diagonal(distance, 0.0)
    distance = np.clip((distance + distance.T) / 2.0, 0.0, 1.0)  # enforce symmetry
    z = linkage(squareform(distance, checks=False), method="average")
    labels = fcluster(z, t=threshold, criterion="distance")

    groups: dict[int, list[str]] = {}
    for name, label in zip(lab_names, labels, strict=True):
        groups.setdefault(int(label), []).append(name)
    return groups


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sets", nargs="+", default=["a", "b"])
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Jaccard distance cut for cluster formation (default 0.5, i.e. "
        "analytes co-present in >50%% of active hours where either appears)",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("experiments/baselines/results/panel_derivation.json"),
    )
    args = parser.parse_args(argv)

    config = load_variable_config()
    lab_names = config.names_by_kind("lab")
    cohort = load_cohort(sets=tuple(args.sets))

    jaccard, conditional, marginal = co_measurement_matrices(cohort, lab_names)
    groups = cluster_panels(jaccard, lab_names, args.threshold)

    print(f"\nAnalytes: {len(lab_names)}  |  sets: {args.sets}")
    print(f"Jaccard distance threshold: {args.threshold}\n")
    print("=" * 78)
    print("DERIVED CLUSTERS (unsupervised — no clinical panel definitions supplied)")
    print("=" * 78)
    for label in sorted(groups, key=lambda k: (-len(groups[k]), k)):
        members = sorted(groups[label])
        if len(members) > 1:
            idx = [lab_names.index(m) for m in members]
            sub = jaccard[np.ix_(idx, idx)]
            cohesion = float(sub[np.triu_indices(len(idx), k=1)].mean())
            print(f"  cluster {label}: {members}")
            print(f"      mean within-cluster Jaccard = {cohesion:.3f}")
        else:
            print(f"  cluster {label}: {members}  (singleton)")

    print("\n" + "=" * 78)
    print("STRONGEST PAIRWISE CO-MEASUREMENT (Jaccard)")
    print("=" * 78)
    pairs = [
        (jaccard[i, j], lab_names[i], lab_names[j])
        for i in range(len(lab_names))
        for j in range(i + 1, len(lab_names))
    ]
    for score, a, b in sorted(pairs, reverse=True)[:15]:
        print(f"  {score:.3f}  {a:<12} {b}")

    print("\n" + "=" * 78)
    print("MARGINAL PRESENCE FREQUENCY (share of active lab hours containing analyte)")
    print("=" * 78)
    lab_cols = [cohort.variable_index(n) for n in lab_names]
    n_active_hours = int(np.count_nonzero(cohort.m[:, :, lab_cols].any(axis=2)))
    for i in np.argsort(-marginal):
        share = marginal[i] / n_active_hours
        print(f"  {share:.3f}  {lab_names[i]:<12} ({int(marginal[i]):,} draws)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sets": args.sets,
        "n_patients": cohort.n_patients,
        "n_active_lab_patient_hours": n_active_hours,
        "threshold": args.threshold,
        "lab_names": list(lab_names),
        "jaccard": jaccard.round(4).tolist(),
        "conditional": conditional.round(4).tolist(),
        "clusters": {str(k): sorted(v) for k, v in groups.items()},
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
