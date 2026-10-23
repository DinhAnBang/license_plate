"""Paths shared by source runs and packaged builds."""

from __future__ import annotations

import sys
from pathlib import Path


def application_directory() -> Path:
    """Return the project directory or the directory containing the executable."""

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def resource_directory() -> Path:
    """Locate bundled read-only files, including PyInstaller onefile data."""

    if getattr(sys, "frozen", False):
        bundle_directory = getattr(sys, "_MEIPASS", None)
        if bundle_directory is not None:
            return Path(bundle_directory).resolve()
    return application_directory()


__all__ = ["application_directory", "resource_directory"]
