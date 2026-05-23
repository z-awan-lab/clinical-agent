"""Logging configuration.

Use ``setup_logging()`` from scripts and notebooks. Library code uses
``logging.getLogger(__name__)`` and never configures handlers itself.
"""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a sensible default format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
