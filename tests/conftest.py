"""Shared fixtures.

Tests that need the real dataset are marked ``slow`` and skip cleanly when the
data has not been downloaded, so the fast suite runs anywhere.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from cliniverse.config import VariableConfig, load_variable_config
from cliniverse.data.cohort import Cohort
from cliniverse.data.physionet2012 import DEFAULT_CACHE, load_cohort

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def variable_config() -> VariableConfig:
    return load_variable_config()


@pytest.fixture(scope="session")
def data_cache() -> pathlib.Path:
    return REPO_ROOT / DEFAULT_CACHE


@pytest.fixture(scope="session")
def has_set_a(data_cache: pathlib.Path) -> bool:
    return (data_cache / "set-a").is_dir()


@pytest.fixture(scope="session")
def cohort_a(data_cache: pathlib.Path, has_set_a: bool) -> Cohort:
    if not has_set_a:
        pytest.skip("set-a not downloaded; run scripts/verify_physionet2012.py")
    return load_cohort(data_cache, sets=("a",), download=False)


@pytest.fixture
def toy_cohort() -> Cohort:
    """A tiny hand-built cohort: 4 patients, 3 hours, 2 variables.

    Patient 3 is degenerate (no observations), mirroring the real dataset.
    """
    n, t, v = 4, 3, 2
    x = np.full((n, t, v), np.nan, dtype=np.float32)
    m = np.zeros((n, t, v), dtype=bool)

    # patient 0: both variables at hour 0
    x[0, 0, 0], x[0, 0, 1] = 1.0, 2.0
    m[0, 0, 0] = m[0, 0, 1] = True
    # patient 1: variable 0 across all hours
    x[1, :, 0] = np.array([3.0, 4.0, 5.0], dtype=np.float32)
    m[1, :, 0] = True
    # patient 2: variable 1 at the final hour only
    x[2, 2, 1] = 6.0
    m[2, 2, 1] = True
    # patient 3: nothing observed

    return Cohort(
        record_ids=np.arange(n, dtype=np.int64),
        source_set=np.array(["a", "a", "b", "b"], dtype=np.str_),
        x=x,
        m=m,
        statics=np.zeros((n, 1), dtype=np.float32),
        statics_mask=np.ones((n, 1), dtype=bool),
        labels={"mortality": np.array([0, 1, 0, 1], dtype=np.float32)},
        variable_names=("alpha", "beta"),
        static_names=("Age",),
    )
