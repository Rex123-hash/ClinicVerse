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
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cliniverse.acquisition.catalogue import PanelCatalogue
from cliniverse.data.cohort import BoolArray, Cohort
from cliniverse.exceptions import ConfigError
from twinbench.disclosure import DEFAULT_EPOCH_HOURS, DisclosureEngine, Protocol
from twinbench.masking import MaskingMechanism

MANIFEST_VERSION = "1"


class CaseSpec(BaseModel):
    """One episode, described completely enough to rebuild it."""

    model_config = ConfigDict(frozen=True)

    record_id: int
    patient_index: int
    mechanism_id: str
    protocol: Protocol
    cost_regime: str
    budget: float = Field(ge=0)
    epoch_hours: tuple[int, ...]

    def key(self) -> str:
        """Stable, order-independent identity string for hashing."""
        return "|".join(
            (
                str(self.record_id),
                str(self.patient_index),
                self.mechanism_id,
                str(self.protocol),
                self.cost_regime,
                f"{self.budget:.6f}",
                ",".join(str(h) for h in self.epoch_hours),
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
    config_hash: str = ""

    @property
    def content_hash(self) -> str:
        """Hash of every case key, order-independent.

        Order independence matters: a manifest built by iterating patients in a
        different order describes the same benchmark and must hash the same.
        """
        h = hashlib.sha256()
        h.update(f"{self.version}|{self.dataset}|{','.join(self.sets)}|".encode())
        h.update(f"{self.cutoff_hours}|".encode())
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
    *,
    dataset: str = "physionet-cinc-2012",
    sets: tuple[str, ...] = ("a", "b"),
    cutoff_hours: int = 24,
    protocol: Protocol = Protocol.SUPPORT_BLIND,
    cost_regime: str = "shared_plus_marginal",
    budget: float = 5.0,
    epoch_hours: tuple[int, ...] = DEFAULT_EPOCH_HOURS,
) -> CaseManifest:
    """Describe one case per patient in ``cohort``."""
    cases = tuple(
        CaseSpec(
            record_id=int(cohort.record_ids[i]),
            patient_index=i,
            mechanism_id=mechanism.mechanism_id,
            protocol=protocol,
            cost_regime=cost_regime,
            budget=budget,
            epoch_hours=epoch_hours,
        )
        for i in range(cohort.n_patients)
    )
    return CaseManifest(dataset=dataset, sets=sets, cutoff_hours=cutoff_hours, cases=cases)


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
    observed = cohort.m[case.patient_index]
    return mechanism.hidden_for(observed, case.patient_index, catalogue, cohort.variable_names)


def engine_for(
    cohort: Cohort,
    case: CaseSpec,
    mechanism: MaskingMechanism,
    catalogue: PanelCatalogue,
) -> DisclosureEngine:
    """Construct the disclosure engine for one case, priced by its cost regime."""
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
