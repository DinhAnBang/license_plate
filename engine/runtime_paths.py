"""Resolve bundled resources separately from persistent application data."""

from __future__ import annotations

import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def is_frozen() -> bool:
    """Return whether the process is running from a PyInstaller bundle."""

    return bool(getattr(sys, "frozen", False))


def get_resource_root() -> Path:
    """Return the root containing bundled, read-only application resources."""

    if is_frozen():
        bundle_root = getattr(sys, "_MEIPASS", None)
        if not bundle_root:
            raise RuntimeError("Frozen application is missing the PyInstaller resource root.")
        return Path(bundle_root).resolve()
    return SOURCE_ROOT


def get_app_root() -> Path:
    """Return the persistent root used for inputs and generated output."""

    if is_frozen():
        return Path(sys.executable).resolve().parent
    return SOURCE_ROOT


def get_model_path(*parts: str, resource_root: str | Path | None = None) -> Path:
    """Resolve one model path inside the resource bundle."""

    root = Path(resource_root).resolve() if resource_root is not None else get_resource_root()
    return root.joinpath("models", *parts).resolve()


def get_output_root(app_root: str | Path | None = None) -> Path:
    """Resolve the request output root outside the one-file extraction area."""

    root = Path(app_root).resolve() if app_root is not None else get_app_root()
    return (root / "output" / "requests").resolve()
