"""Verify PhysioNet/CinC Challenge 2012 access and reproduce the dataset statistics
quoted in docs/research_assessment.md section 5.1.

This script is intentionally dependency-light (requests-free, stdlib + pandas) and
read-only. It downloads ~20 MB of open-licensed data (ODC-BY v1.0) into a cache
directory, then prints the statistics.

Usage:
    python scripts/verify_physionet2012.py [--cache DIR] [--set {a,b,c}]

Every number printed here is computed from the downloaded files. Nothing is hardcoded.
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import sys
import tarfile
import urllib.request

import pandas as pd

BASE_URL = "https://physionet.org/files/challenge-2012/1.0.0"
OUTCOME_FILES = ("Outcomes-a.txt", "Outcomes-b.txt", "Outcomes-c.txt")

# Recorded once at 00:00; not part of the time-series signal.
#
# `Weight` is deliberately NOT in this set. It is recorded repeatedly through the
# stay (95.9% of set-a Weight rows are after hour 0), so it is a time-series
# variable, giving 37 rather than 36. Excluding it was an error in the first
# version of this script and in the parser it was written to check.
STATIC_PARAMS = frozenset({"RecordID", "Age", "Gender", "Height", "ICUType"})


def _download(url: str, dest: pathlib.Path) -> pathlib.Path:
    """Download `url` to `dest` unless already cached."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached  {dest.name} ({dest.stat().st_size:,} bytes)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetch   {url}")
    with urllib.request.urlopen(url, timeout=300) as resp:
        dest.write_bytes(resp.read())
    print(f"          -> {dest.name} ({dest.stat().st_size:,} bytes)")
    return dest


def fetch_dataset(cache: pathlib.Path, which: str) -> tuple[pathlib.Path, list[pathlib.Path]]:
    """Ensure outcomes + the requested record set are present. Returns (set_dir, records)."""
    print(f"Cache: {cache}")
    print("Downloading outcome files:")
    for name in OUTCOME_FILES:
        _download(f"{BASE_URL}/{name}", cache / name)

    print(f"Downloading set-{which}:")
    archive = _download(f"{BASE_URL}/set-{which}.tar.gz", cache / f"set-{which}.tar.gz")
    set_dir = cache / f"set-{which}"
    if not set_dir.exists():
        print(f"  extract {archive.name}")
        with tarfile.open(archive) as tf:
            # filter="data" rejects absolute paths / traversal members (Python >= 3.12).
            tf.extractall(cache, filter="data")

    records = sorted(set_dir.glob("*.txt"))
    if not records:
        raise FileNotFoundError(f"no record files found under {set_dir}")
    return set_dir, records


def report_outcomes(cache: pathlib.Path) -> None:
    print("\n" + "=" * 72)
    print("OUTCOMES (all three sets)")
    print("=" * 72)
    for name in OUTCOME_FILES:
        df = pd.read_csv(cache / name)
        n = len(df)
        deaths = int(df["In-hospital_death"].sum())
        print(f"{name:<16} n={n:5d}  deaths={deaths:4d}  rate={deaths / n:6.2%}")
    cols = list(pd.read_csv(cache / OUTCOME_FILES[0]).columns)
    print(f"columns: {cols}")


def report_timeseries(records: list[pathlib.Path], which: str) -> None:
    print("\n" + "=" * 72)
    print(f"SET-{which.upper()} TIME-SERIES STRUCTURE")
    print("=" * 72)
    print(f"record files: {len(records)}")

    present: collections.Counter[str] = collections.Counter()
    obs_rows: collections.Counter[str] = collections.Counter()
    rows_per_rec: list[int] = []
    times_per_rec: list[int] = []
    empty: list[str] = []

    for path in records:
        df = pd.read_csv(path)
        ts = df[~df["Parameter"].isin(STATIC_PARAMS)]
        rows_per_rec.append(len(ts))
        times_per_rec.append(ts["Time"].nunique())
        if len(ts) == 0:
            empty.append(path.stem)
            continue
        present.update(ts["Parameter"].unique())
        obs_rows.update(ts["Parameter"].value_counts().to_dict())

    n_rec = len(records)
    s_rows, s_times = pd.Series(rows_per_rec), pd.Series(times_per_rec)
    print(
        f"\nobservation rows per record:    median={s_rows.median():.0f} "
        f"mean={s_rows.mean():.0f} min={s_rows.min()} max={s_rows.max()}"
    )
    print(
        f"distinct timestamps per record: median={s_times.median():.0f} "
        f"mean={s_times.mean():.1f} min={s_times.min()} max={s_times.max()}"
    )
    print(f"\nDEGENERATE RECORDS (zero time-series observations): {len(empty)} {empty}")
    tiny = sorted(
        (p.stem, n) for p, n in zip(records, rows_per_rec, strict=True) if 0 < n < 20
    )
    print(f"records with <20 observations: {len(tiny)} {tiny}")

    print(f"\ndistinct time-series parameters: {len(present)}")
    cov = pd.DataFrame(
        {"records_with_var": pd.Series(present), "total_obs": pd.Series(obs_rows)}
    ).sort_values("records_with_var", ascending=False)
    cov["coverage"] = cov["records_with_var"] / n_rec
    cov["obs_per_covered_rec"] = cov["total_obs"] / cov["records_with_var"]
    pd.set_option("display.width", 200)
    print("\nper-variable coverage (fraction of records where EVER measured):")
    print(cov.to_string(float_format=lambda v: f"{v:8.3f}"))

    n_vars = len(present)
    total_cells = n_rec * 48 * n_vars
    observed = int(cov["total_obs"].sum())
    print(f"\nnaive (record x 48h x {n_vars} var) grid cells = {total_cells:,}")
    print(f"observation rows                             = {observed:,}")
    print(
        f"=> upper bound on grid occupancy             = {observed / total_cells:.2%}"
        f"  (i.e. >= {1 - observed / total_cells:.2%} missing)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=pathlib.Path,
        default=pathlib.Path("data/raw/physionet2012"),
        help="directory to cache downloads (default: data/raw/physionet2012)",
    )
    parser.add_argument(
        "--set",
        dest="which",
        choices=("a", "b", "c"),
        default="a",
        help="which record set to analyse in detail (default: a)",
    )
    args = parser.parse_args(argv)

    _, records = fetch_dataset(args.cache, args.which)
    report_outcomes(args.cache)
    report_timeseries(records, args.which)
    print("\nLicense: Open Data Commons Attribution License v1.0 (ODC-BY). Open access.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
