"""Logging configuration helpers."""

from __future__ import annotations

import logging


def configure_logging(log_level: str = "INFO") -> None:
    """Set the application log format once for local development and tests."""

    if logging.getLogger().handlers:
        logging.getLogger().setLevel(log_level.upper())
        return

    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
