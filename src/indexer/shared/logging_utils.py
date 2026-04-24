"""Logging helpers for the indexing pipeline."""

from __future__ import annotations

import json
import logging
from typing import Any


def configure_logging(log_level: str) -> None:
    """Configure application logging with a structured message format."""

    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module name."""

    return logging.getLogger(name)


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Emit a structured log event."""

    if not fields:
        logger.info(event)
        return

    logger.info("%s %s", event, json.dumps(fields, default=str, sort_keys=True))
