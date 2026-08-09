"""Machine-readable run artifacts.

Every reported number must be traceable to an executed run. An artifact records
the provenance needed to recompute it independently: source revision, config and
cohort fingerprints, the split, the representation, the model and its selected
hyperparameters, the seed, and the raw out-of-fold predictions.

Predictions are retained deliberately. A reviewer who does not trust our metric
implementations can recompute every figure from ``predictions.npz`` alone.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import platform
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

from cliniverse.data.cohort import Cohort
from cliniverse.data.splits import Split


def stable_hash(payload: object) -> str:
    """Deterministic hash of any JSON-encodable structure."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def git_sha() -> str:
    """Best-effort source revision; ``unknown`` outside a checkout."""
    executable = shutil.which("git")
    if executable is None:
        return "unknown"
    try:
        result = subprocess.run(  # noqa: S603 - resolved executable, fixed args
            [executable, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def git_is_dirty() -> bool:
    """Whether the working tree has uncommitted changes.

    Recorded so a result produced from an unclean tree is visibly less
    reproducible than one produced from a committed revision.
    """
    executable = shutil.which("git")
    if executable is None:
        return False
    try:
        result = subprocess.run(  # noqa: S603 - resolved executable, fixed args
            [executable, "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return bool(result.stdout.strip())


def cohort_fingerprint(cohort: Cohort) -> str:
    """Content hash of the exact cohort a run consumed."""
    return stable_hash(
        {
            "record_ids": cohort.record_ids.tolist(),
            "source_set": cohort.source_set.tolist(),
            "n_hours": cohort.n_hours,
            "variable_names": list(cohort.variable_names),
            "static_names": list(cohort.static_names),
            "values_sha256": hashlib.sha256(
                np.nan_to_num(cohort.x, nan=-9999.0).tobytes()
            ).hexdigest(),
            "mask_sha256": hashlib.sha256(cohort.m.tobytes()).hexdigest(),
        }
    )


def split_hash(splits: list[Split]) -> str:
    """Content hash of a fold assignment, so a run pins the exact partition."""
    return stable_hash(
        [
            {
                "fold": s.fold,
                "seed": s.seed,
                "train": sorted(int(i) for i in s.train),
                "validation": sorted(int(i) for i in s.validation),
            }
            for s in sorted(splits, key=lambda s: s.fold)
        ]
    )


@dataclasses.dataclass(slots=True)
class RunArtifact:
    """One (representation, model) evaluation."""

    run_id: str
    representation: str
    model: str
    cutoff_hours: int
    seed: int
    n_features: int
    feature_names: list[str]
    hyperparameters: dict[str, Any]
    search_space: dict[str, Any]
    selection_metric: str
    per_fold_selection: list[dict[str, Any]]
    metrics: dict[str, Any]
    intervals: dict[str, Any]
    reliability: dict[str, list[float]]
    predictions: np.ndarray
    labels: np.ndarray
    record_ids: np.ndarray
    provenance: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        """Everything except the raw prediction vectors."""
        return {
            "run_id": self.run_id,
            "representation": self.representation,
            "model": self.model,
            "cutoff_hours": self.cutoff_hours,
            "seed": self.seed,
            "n_features": self.n_features,
            "hyperparameters": self.hyperparameters,
            "search_space": self.search_space,
            "selection_metric": self.selection_metric,
            "per_fold_selection": self.per_fold_selection,
            "metrics": self.metrics,
            "intervals": self.intervals,
            "reliability": self.reliability,
            "provenance": self.provenance,
        }


def build_provenance(
    *,
    cohort: Cohort,
    splits: list[Split],
    config_payload: object,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the provenance block shared by every artifact in a run."""
    payload: dict[str, Any] = {
        "git_sha": git_sha(),
        "git_dirty": git_is_dirty(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config_hash": stable_hash(config_payload),
        "cohort_fingerprint": cohort_fingerprint(cohort),
        "split_hash": split_hash(splits),
        "n_patients": cohort.n_patients,
        "sets": sorted(set(cohort.source_set.tolist())),
    }
    if extra:
        payload.update(extra)
    return payload


def write_run(
    artifacts: list[RunArtifact],
    directory: pathlib.Path,
    *,
    manifest_name: str = "results.json",
    predictions_name: str = "predictions.npz",
) -> tuple[pathlib.Path, pathlib.Path]:
    """Write summaries and raw predictions.

    Returns ``(manifest_path, predictions_path)``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / manifest_name
    predictions_path = directory / predictions_name

    manifest = {
        "schema": "cliniverse.evaluation.artifacts/1",
        "runs": [a.summary() for a in artifacts],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
        newline="\n",
    )

    arrays: dict[str, np.ndarray] = {}
    for a in artifacts:
        arrays[f"pred__{a.run_id}"] = a.predictions
    if artifacts:
        arrays["labels"] = artifacts[0].labels
        arrays["record_ids"] = artifacts[0].record_ids
    # Keyword expansion is required: it is what names each array in the archive.
    # numpy's stub types **kwargs as `bool` (for `allow_pickle`), so the correct
    # call does not type-check. Passing the dict positionally would store it as a
    # single `arr_0` entry and break every reader.
    np.savez_compressed(predictions_path, **arrays)  # type: ignore[arg-type]
    return manifest_path, predictions_path
