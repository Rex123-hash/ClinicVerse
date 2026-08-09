"""Cliniverse exception hierarchy.

Narrow, typed exceptions so callers can distinguish "the data is wrong" from
"you configured this wrong" from "this claim is not permitted".
"""

from __future__ import annotations


class CliniverseError(Exception):
    """Base class for every error raised by this package."""


class DataError(CliniverseError):
    """Raw data is missing, malformed, or fails a structural invariant."""


class DownloadError(DataError):
    """A dataset artifact could not be retrieved or failed integrity checks."""


class ConfigError(CliniverseError):
    """A configuration file is missing, malformed, or internally inconsistent."""


class LeakageError(CliniverseError):
    """An operation would leak information across a split boundary.

    Raised eagerly rather than warned about: silent leakage invalidates every
    downstream number.
    """


class BudgetError(CliniverseError):
    """An acquisition policy attempted to spend beyond its budget, or to acquire
    an observation outside the acquirable support."""


class SafetyError(CliniverseError):
    """An output would constitute a prohibited clinical claim."""
