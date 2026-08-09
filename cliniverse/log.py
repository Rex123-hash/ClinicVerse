"""Structured logging.

A single ``configure`` entrypoint so scripts, tests and the API all emit the same
shape of log record. Console-friendly by default; JSON when ``json_output=True``
so Cloud Run logs stay parseable.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_CONFIGURED = False


def configure(level: str = "INFO", *, json_output: bool = False) -> None:
    """Configure structlog + stdlib logging. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger, configuring logging on first use."""
    configure()
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger
