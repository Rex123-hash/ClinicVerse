"""PhysioNet/CinC Challenge 2012 loader.

Downloads (open access, ODC-BY v1.0 — no credentialing), parses the long-format
per-patient records, repairs known unit-entry errors, drops physiologically
impossible readings, and bins onto an hourly grid.

Degenerate records — patients with zero time-series observations — are retained
and flagged, never silently dropped: they are a real property of the dataset and
a useful robustness fixture.
"""

from __future__ import annotations

import csv
import hashlib
import pathlib
import tarfile
import urllib.error
import urllib.request
from typing import Final

import numpy as np

from cliniverse.config import VariableConfig, load_variable_config
from cliniverse.data.cohort import BoolArray, Cohort, FloatArray
from cliniverse.exceptions import DataError, DownloadError
from cliniverse.log import get_logger

log = get_logger(__name__)

BASE_URL: Final = "https://physionet.org/files/challenge-2012/1.0.0"
RECORD_SETS: Final = ("a", "b", "c")
DEFAULT_CACHE: Final = pathlib.Path("data/raw/physionet2012")

#: Expected sizes in bytes, verified 2026-08-09. A size mismatch means the
#: upstream artifact changed and every cached derivative must be regenerated.
EXPECTED_BYTES: Final[dict[str, int]] = {
    "Outcomes-a.txt": 79_219,
    "Outcomes-b.txt": 79_149,
    "Outcomes-c.txt": 79_191,
    "set-a.tar.gz": 6_632_372,
    "set-b.tar.gz": 6_652_690,
    "set-c.tar.gz": 6_600_293,
}

#: Parameters that appear in the record files but are not measurements.
_NON_MEASUREMENT: Final = frozenset({"RecordID"})

#: Within-hour aggregation. Mean for continuous signals; max for the ventilation
#: indicator, where "ventilated at any point this hour" is the meaningful summary.
_MAX_AGGREGATED: Final = frozenset({"MechVent"})

LABEL_COLUMNS: Final = {
    "mortality": "In-hospital_death",
    "length_of_stay": "Length_of_stay",
}

#: Provided in the outcomes file and computed from the same 48h window as the
#: features. Legitimate as *baselines*; using them as model inputs is leakage.
LEAKY_OUTCOME_COLUMNS: Final = frozenset({"SAPS-I", "SOFA", "Survival"})


# --------------------------------------------------------------- download ----
def _fetch(url: str, dest: pathlib.Path, *, expected_bytes: int | None = None) -> pathlib.Path:
    if dest.exists() and dest.stat().st_size > 0:
        if expected_bytes is not None and dest.stat().st_size != expected_bytes:
            raise DownloadError(
                f"cached {dest.name} is {dest.stat().st_size} bytes, "
                f"expected {expected_bytes}; delete it and re-download"
            )
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading", url=url, dest=str(dest))
    try:
        with urllib.request.urlopen(url, timeout=300) as resp:  # noqa: S310 - fixed https host
            payload = resp.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise DownloadError(f"could not download {url}: {exc}") from exc
    if expected_bytes is not None and len(payload) != expected_bytes:
        raise DownloadError(
            f"{dest.name}: downloaded {len(payload)} bytes, expected {expected_bytes}"
        )
    dest.write_bytes(payload)
    return dest


def download_dataset(
    cache_dir: pathlib.Path | str = DEFAULT_CACHE,
    sets: tuple[str, ...] = RECORD_SETS,
    *,
    verify_size: bool = True,
) -> pathlib.Path:
    """Ensure outcomes and the requested record sets are present locally.

    Returns the cache directory. Idempotent.
    """
    cache = pathlib.Path(cache_dir)
    unknown = set(sets) - set(RECORD_SETS)
    if unknown:
        raise DataError(f"unknown record set(s): {sorted(unknown)}")

    for which in sets:
        name = f"Outcomes-{which}.txt"
        _fetch(
            f"{BASE_URL}/{name}",
            cache / name,
            expected_bytes=EXPECTED_BYTES[name] if verify_size else None,
        )
        archive_name = f"set-{which}.tar.gz"
        archive = _fetch(
            f"{BASE_URL}/{archive_name}",
            cache / archive_name,
            expected_bytes=EXPECTED_BYTES[archive_name] if verify_size else None,
        )
        set_dir = cache / f"set-{which}"
        if not set_dir.is_dir():
            log.info("extracting", archive=archive.name)
            with tarfile.open(archive) as tf:
                # filter="data" rejects absolute paths and traversal members.
                tf.extractall(cache, filter="data")
            if not set_dir.is_dir():
                raise DataError(f"{archive_name} did not yield {set_dir}")
    return cache


# ------------------------------------------------------------------ parse ----
def _parse_time(raw: str) -> int | None:
    """``'HH:MM'`` -> elapsed whole hours since admission, or None if malformed."""
    hh, _, mm = raw.partition(":")
    try:
        hours, minutes = int(hh), int(mm)
    except ValueError:
        return None
    if hours < 0 or not 0 <= minutes < 60:
        return None
    return hours


def parse_record(
    path: pathlib.Path, config: VariableConfig
) -> tuple[dict[str, float], list[tuple[int, str, float]]]:
    """Parse one record file.

    Returns ``(statics, observations)`` where ``observations`` is a list of
    ``(hour, variable, cleaned_value)``. Readings that are implausible after
    repair, or that fall outside the horizon, are dropped.
    """
    statics: dict[str, float] = {}
    observations: list[tuple[int, str, float]] = []

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return statics, observations
        if [h.strip() for h in header] != ["Time", "Parameter", "Value"]:
            raise DataError(f"{path.name}: unexpected header {header!r}")

        for row in reader:
            if len(row) != 3:
                continue  # tolerate ragged trailing lines
            raw_time, param, raw_value = (c.strip() for c in row)
            if not param or param in _NON_MEASUREMENT:
                continue
            try:
                value = float(raw_value)
            except ValueError:
                continue

            cleaned = config.clean(param, value)
            if cleaned is None:
                continue

            if param in config.statics:
                statics[param] = cleaned
                continue

            hour = _parse_time(raw_time)
            if hour is None or hour >= config.horizon_hours:
                continue
            observations.append((hour, param, cleaned))

    return statics, observations


def _read_outcomes(path: pathlib.Path) -> dict[int, dict[str, float]]:
    outcomes: dict[int, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                rid = int(row["RecordID"])
            except (KeyError, ValueError) as exc:
                raise DataError(f"{path.name}: bad RecordID in row {row!r}") from exc
            outcomes[rid] = {
                k: float(v) for k, v in row.items() if k != "RecordID" and v not in ("", None)
            }
    return outcomes


# ------------------------------------------------------------------- load ----
def load_cohort(
    cache_dir: pathlib.Path | str = DEFAULT_CACHE,
    sets: tuple[str, ...] = RECORD_SETS,
    *,
    config: VariableConfig | None = None,
    download: bool = True,
) -> Cohort:
    """Load one or more record sets into a :class:`Cohort`.

    Args:
        cache_dir: where raw files live (downloaded on demand).
        sets: which record sets to include.
        config: variable schema; defaults to ``configs/variables.yaml``.
        download: if False, fail rather than fetch missing files.
    """
    config = config or load_variable_config()
    cache = pathlib.Path(cache_dir)
    if download:
        download_dataset(cache, sets)

    var_names = config.variable_names
    static_names = config.static_names
    var_index = {name: i for i, name in enumerate(var_names)}
    static_index = {name: i for i, name in enumerate(static_names)}
    n_hours = config.horizon_hours

    record_ids: list[int] = []
    source: list[str] = []
    x_rows: list[FloatArray] = []
    m_rows: list[BoolArray] = []
    s_rows: list[FloatArray] = []
    sm_rows: list[BoolArray] = []
    label_rows: dict[str, list[float]] = {task: [] for task in LABEL_COLUMNS}

    for which in sets:
        set_dir = cache / f"set-{which}"
        if not set_dir.is_dir():
            raise DataError(f"missing record directory {set_dir}; run download_dataset()")
        outcomes = _read_outcomes(cache / f"Outcomes-{which}.txt")

        files = sorted(set_dir.glob("*.txt"))
        if not files:
            raise DataError(f"no record files in {set_dir}")
        log.info("parsing set", set=which, n_records=len(files))

        for path in files:
            try:
                rid = int(path.stem)
            except ValueError as exc:
                raise DataError(f"non-numeric record filename {path.name}") from exc
            if rid not in outcomes:
                # A record without a label cannot participate in supervised
                # evaluation; skipping is correct, but it must be visible.
                log.warning("record has no outcome row, skipping", record_id=rid, set=which)
                continue

            statics, observations = parse_record(path, config)

            # Accumulate into sum/count so within-hour duplicates average cleanly.
            totals = np.zeros((n_hours, len(var_names)), dtype=np.float64)
            counts = np.zeros((n_hours, len(var_names)), dtype=np.int32)
            maxima = np.full((n_hours, len(var_names)), -np.inf, dtype=np.float64)
            for hour, param, value in observations:
                col = var_index.get(param)
                if col is None:
                    continue
                totals[hour, col] += value
                counts[hour, col] += 1
                maxima[hour, col] = max(maxima[hour, col], value)

            mask = counts > 0
            values = np.full((n_hours, len(var_names)), np.nan, dtype=np.float32)
            with np.errstate(invalid="ignore"):
                mean = np.divide(totals, counts, out=np.zeros_like(totals), where=mask)
            values[mask] = mean[mask].astype(np.float32)
            for name in _MAX_AGGREGATED & set(var_index):
                col = var_index[name]
                sel = mask[:, col]
                values[sel, col] = maxima[sel, col].astype(np.float32)

            static_vec = np.full(len(static_names), np.nan, dtype=np.float32)
            static_mask = np.zeros(len(static_names), dtype=bool)
            for name, value in statics.items():
                col = static_index.get(name)
                if col is not None:
                    static_vec[col] = np.float32(value)
                    static_mask[col] = True

            record_ids.append(rid)
            source.append(which)
            x_rows.append(values)
            m_rows.append(mask)
            s_rows.append(static_vec)
            sm_rows.append(static_mask)

            row = outcomes[rid]
            label_rows["mortality"].append(row[LABEL_COLUMNS["mortality"]])
            label_rows["length_of_stay"].append(row[LABEL_COLUMNS["length_of_stay"]])

    if not record_ids:
        raise DataError(f"no records loaded from sets {sets}")

    los = np.asarray(label_rows["length_of_stay"], dtype=np.float32)
    labels: dict[str, FloatArray] = {
        "mortality": np.asarray(label_rows["mortality"], dtype=np.float32),
        "length_of_stay": los,
        # LOS uses -1 as a missing sentinel; propagate as NaN rather than
        # silently encoding "unknown stay" as "short stay".
        "prolonged_stay": np.where(los < 0, np.nan, (los > 3).astype(np.float32)).astype(
            np.float32
        ),
    }

    cohort = Cohort(
        record_ids=np.asarray(record_ids, dtype=np.int64),
        source_set=np.asarray(source, dtype=np.str_),
        x=np.stack(x_rows).astype(np.float32),
        m=np.stack(m_rows).astype(bool),
        statics=np.stack(s_rows).astype(np.float32),
        statics_mask=np.stack(sm_rows).astype(bool),
        labels=labels,
        variable_names=var_names,
        static_names=static_names,
    )
    log.info("cohort loaded", sets=sets, **cohort.describe())
    return cohort


def cohort_fingerprint(cohort: Cohort) -> str:
    """Stable content hash of a cohort, for reproducibility assertions."""
    h = hashlib.sha256()
    h.update(cohort.record_ids.tobytes())
    h.update(np.nan_to_num(cohort.x, nan=-9999.0).tobytes())
    h.update(cohort.m.tobytes())
    for task in sorted(cohort.labels):
        h.update(task.encode())
        h.update(np.nan_to_num(cohort.labels[task], nan=-9999.0).tobytes())
    return h.hexdigest()
