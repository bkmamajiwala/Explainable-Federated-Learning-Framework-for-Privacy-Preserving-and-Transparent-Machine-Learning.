"""Structured logging for the federated workflow.

Logs are printed to the console with a context tag and also written to
``results/logs/`` so the full federated workflow can be audited afterwards.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

_LOG_FORMAT = "[%(levelname)s] %(message)s"
_TAG_FORMAT = "[%(levelname)s][%(tag)s] %(message)s"

_FORMATTERS: dict[str, logging.Formatter] = {}


class _TagFilter(logging.Filter):
    def __init__(self, tag: str) -> None:
        super().__init__()
        self.tag = tag

    def filter(self, record: logging.LogRecord) -> bool:
        record.tag = self.tag  # type: ignore[attr-defined]
        return True


def _get_formatter(tag: str | None) -> logging.Formatter:
    if tag is None:
        return _FORMATTERS.setdefault(
            "plain", logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S")
        )
    return _FORMATTERS.setdefault(
        tag, logging.Formatter(_TAG_FORMAT, datefmt="%H:%M:%S")
    )


def setup_logging(
    tag: str | None = None,
    log_dir: str | Path | None = None,
    level: int = logging.INFO,
    propagate: bool = True,
) -> logging.Logger:
    """Create a logger with a workflow tag (SERVER / CLIENT / ROUND ...).

    Parameters
    ----------
    tag:
        Short tag rendered inside each log line, e.g. ``SERVER``.
    log_dir:
        Directory for the file handler. ``None`` disables file logging.
    level:
        Minimum logging level.
    propagate:
        Whether to propagate to the root logger (used to avoid duplicate
        lines when a logger is re-created for the same tag).
    """
    logger = logging.getLogger(f"xfl.{tag}" if tag else "xfl")
    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = propagate

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(_get_formatter(tag))
    logger.addHandler(console)

    if log_dir is not None:
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(path / f"run_{stamp}.log", encoding="utf-8")
        fh.setFormatter(_get_formatter(tag))
        logger.addHandler(fh)

    if tag is not None:
        logger.addFilter(_TagFilter(tag))
    return logger


def get_logger(tag: str | None = None) -> logging.Logger:
    return logging.getLogger(f"xfl.{tag}" if tag else "xfl")