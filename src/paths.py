"""Paths shared by source runs and packaged builds."""

from __future__ import annotations

import sys
from pathlib import Path


def application_directory() -> Path:
    """Return the project directory or the directory containing the executable."""

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


__all__ = ["application_directory"]
