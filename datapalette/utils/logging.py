"""Logging setup."""

from __future__ import annotations

import logging
import os
import sys

__all__ = ["setup_logging"]


def setup_logging(log_file: str | None = None, level: int = logging.INFO) -> None:
    """Configure root logger with console and optional file handler.

    Parameters
    ----------
    log_file : str or None
        If given, also write logs to this file path.
    level : int
        Logging level. Default ``logging.INFO``.
    """
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]

    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
