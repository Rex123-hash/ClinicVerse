"""Case specifications and manifests.

A *case* is one evaluation episode: a patient, a masking mechanism and seed, a
disclosure protocol, a cost regime, and a budget. Cases are described rather
than stored — a manifest holds specifications and a content hash, never the
patient data itself, so nothing derived from PhysioNet is committed and any run
can be regenerated from the open source files.

The reproducibility contract is the point: regenerating a manifest from the same
seed must produce the same content hash, and this is asserted in the test suite.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import subprocess
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from cliniverse.acquisition.catalogue import PanelCatalogue
from cliniverse.data.cohort import BoolArray, Cohort
from cliniverse.exceptions import ConfigError
from twinbench.disclosure import DEFAULT_EPOCH_HOURS, DisclosureEngine, Protocol
from twinbench.masking import MaskingMechanism

MANIFEST_VERSION = "2"


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _git_sha() -> str:
    """Best-effort source revision for provenance outside a Git checkout too."""
    executable = shutil.which("git")
    if executable is None:
        return "unknown"
    try:
        return subprocess.run(  # noqa: S603 - resolved git executable, fixed arguments
            [executable, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


class CaseSpec(BaseModel):
    """One episode, described completely enough to rebuild it."""

    model_config = ConfigDict(frozen=True)

    record_id: int
    patient_index: int
    mechanism_id: str
    masking_seed: int
    masking_rate: float = Field(ge=0, le=1, allow_inf_nan=False)
    protocol: Protocol
    cost_regime: str
    budget: float = Field(ge=0, allow_inf_nan=False)
    epoch_hours: tuple[int, ...]
    catalogue_version: str
    catalogue_hash: str

    def key(self) -> str:
        """Stable, order-independent identity string for hashing."""
        return "|".join(
            (
                str(self.record_id),
                str(self.patient_index),
                self.mechanism_id,
                str(self.masking_seed),
                f"{self.masking_rate:.12g}",
                str(self.protocol),
                self.cost_regime,
                f"{self.budget:.6f}",
                ",".join(str(h) for h in self.epoch_hours),
                self.catalogue_version,
                self.catalogue_hash,
            )
        )


class CaseManifest(BaseModel):
    """A set of cases plus enough provenance to reproduce it."""

    model_config = ConfigDict(frozen=True)

    version: str = MANIFEST_VERSION
    dataset: str
    sets: tuple[str, ...]
    cutoff_hours: int
    cases: tuple[CaseSpec, ...]
    dataset_fingerprint: str
    config_hash: str
    git_sha: str

    @property
    def content_hash(self) -> str:
        """Hash of every case key, order-independent.

        Order independence matters: a manifest built by iterating patients in a
        different order describes the same benchmark and must hash the same.
        """
        h = hashlib.sha256()
        h.update(
            _stable_hash(
                {
                    "version": self.version,
                    "dataset": self.dataset,
                    "sets": sorted(self.sets),
                    "cutoff_hours": self.cutoff_hours,
                    "dataset_fingerprint": self.dataset_fingerprint,
                    "config_hash": self.config_hash,
                    "git_sha": self.git_sha,
                }
            ).encode()
        )
        for key in sorted(case.key() for case in self.cases):
            h.update(key.encode())
            h.update(b"\n")
        return h.hexdigest()

    def to_json(self) -> str:
        payload: dict[str, Any] = json.loads(self.model_dump_json())
        payload["content_hash"] = self.content_hash
        return json.dumps(payload, indent=2, sort_keys=True)

    def write(self, path: pathlib.Path) -> pathlib.Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8", newline="\n")
        return path

    @classmethod
    def read(cls, path: pathlib.Path) -> CaseManifest:
        raw = json.loads(path.read_text(encoding="utf-8"))
        declared = raw.pop("content_hash", None)
        manifest = cls.model_validate(raw)
        if declared is not None and declared != manifest.content_hash:
            raise ConfigError(
                f"{path.name}: content hash mismatch — declared {declared}, "
                f"recomputed {manifest.content_hash}. The manifest or the code "
                "that builds it has changed."
            )
        return manifest


def build_manifest(
    cohort: Cohort,
    mechanism: MaskingMechanism,
    catalogue: PanelCatalogue,
    *,
    dataset: str = "physionet-cinc-2012",
    sets: tuple[str, ...] = ("a", "b"),
    cutoff_hours: int = 24,
    protocol: Protocol = Protocol.SUPPORT_BLIND,
    cost_regime: str = "shared_plus_marginal",
    budget: float = 5.0,
    epoch_hours: tuple[int, ...] = DEFAULT_EPOCH_HOURS,
    git_sha: str | None = None,
) -> CaseManifest:
    """Describe one case per patient in ``cohort``."""
    if cutoff_hours <= 0 or cutoff_hours > cohort.n_hours:
        raise ConfigError(f"cutoff_hours must be in [1, {cohort.n_hours}], got {cutoff_hours}")
    if not epoch_hours or epoch_hours[-1] != cutoff_hours:
        raise ConfigError(
            f"final epoch must equal cutoff_hours ({cutoff_hours}), got {epoch_hours}"
        )
    actual_sets = tuple(sorted(set(cohort.source_set.tolist())))
    if tuple(sorted(sets)) != actual_sets:
        raise ConfigError(f"declared sets {sets} do not match cohort sets {actual_sets}")
    # Validate the requested schedule before persisting any cases.
    if cost_regime != catalogue.schedule_name:
        catalogue.with_schedule(cost_regime)

    catalogue_hash = _stable_hash(catalogue.model_dump(mode="json"))
    schema = {
        "variable_names": cohort.variable_names,
        "static_names": cohort.static_names,
        "n_hours": cohort.n_hours,
    }
    truncated = cohort.truncate(cutoff_hours)
    dataset_fingerprint = _stable_hash(
        {
            "record_ids": truncated.record_ids.tolist(),
            "source_set": truncated.source_set.tolist(),
            "values_sha256": hashlib.sha256(
                np.nan_to_num(truncated.x, nan=-9999.0).tobytes()
            ).hexdigest(),
            "mask_sha256": hashlib.sha256(truncated.m.tobytes()).hexdigest(),
            "statics_sha256": hashlib.sha256(
                np.nan_to_num(truncated.statics, nan=-9999.0).tobytes()
            ).hexdigest(),
        }
    )
    cases = tuple(
        CaseSpec(
            record_id=int(cohort.record_ids[i]),
            patient_index=i,
            mechanism_id=mechanism.mechanism_id,
            masking_seed=mechanism.seed,
            masking_rate=mechanism.rate,
            protocol=protocol,
            cost_regime=cost_regime,
            budget=budget,
            epoch_hours=epoch_hours,
            catalogue_version=catalogue.version,
            catalogue_hash=catalogue_hash,
        )
        for i in range(cohort.n_patients)
    )
    return CaseManifest(
        dataset=dataset,
        sets=sets,
        cutoff_hours=cutoff_hours,
        cases=cases,
        dataset_fingerprint=dataset_fingerprint,
        config_hash=_stable_hash(schema),
        git_sha=git_sha or _git_sha(),
    )


def hidden_mask_for(
    cohort: Cohort,
    case: CaseSpec,
    mechanism: MaskingMechanism,
    catalogue: PanelCatalogue,
) -> BoolArray:
    """Rebuild the hidden mask for one case.

    Deterministic given the mechanism's seed and the patient index, so a case
    reproduces identically without storing any patient data.
    """
    if mechanism.mechanism_id != case.mechanism_id:
        raise ConfigError(
            f"mechanism {mechanism.mechanism_id!r} does not match case {case.mechanism_id!r}"
        )
    if case.patient_index < 0 or case.patient_index >= cohort.n_patients:
        raise ConfigError(f"patient index {case.patient_index} is outside the cohort")
    if int(cohort.record_ids[case.patient_index]) != case.record_id:
        raise ConfigError(
            f"case record {case.record_id} does not match patient index {case.patient_index}"
        )
    cutoff = case.epoch_hours[-1]
    observed = cohort.m[case.patient_index]
    hidden: BoolArray = np.zeros_like(observed, dtype=bool)
    hidden[:cutoff] = mechanism.hidden_for(
        observed[:cutoff], case.patient_index, catalogue, cohort.variable_names
    )
    return hidden


def engine_for(
    cohort: Cohort,
    case: CaseSpec,
    mechanism: MaskingMechanism,
    catalogue: PanelCatalogue,
) -> DisclosureEngine:
    """Construct the disclosure engine for one case, priced by its cost regime."""
    catalogue_hash = _stable_hash(catalogue.model_dump(mode="json"))
    if case.catalogue_version != catalogue.version or case.catalogue_hash != catalogue_hash:
        raise ConfigError("catalogue version or content does not match the case manifest")
    priced = (
        catalogue
        if case.cost_regime == catalogue.schedule_name
        else catalogue.with_schedule(case.cost_regime)
    )
    return DisclosureEngine(
        cohort,
        case.patient_index,
        hidden_mask_for(cohort, case, mechanism, catalogue),
        priced,
        budget=case.budget,
        protocol=case.protocol,
        epoch_hours=case.epoch_hours,
    )
