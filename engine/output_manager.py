"""Validated, request-scoped output paths for the persistent engine."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .protocol import validate_request_id


@dataclass(frozen=True)
class RequestOutputPaths:
    request_id: str
    request_dir: Path
    annotated: Path
    result_json: Path
    crops_dir: Path


class RequestOutputManager:
    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()
        output_root = (self.project_root / "output").resolve()
        self.root = (output_root / "requests").resolve()
        if output_root != self.project_root / "output" or self.root != output_root / "requests":
            raise ValueError("Request output root must be inside the project output directory.")

    def paths_for(self, request_id: str, media_type: str) -> RequestOutputPaths:
        request_id = validate_request_id(request_id)
        if media_type not in {"image", "video"}:
            raise ValueError(f"Unsupported output media type: {media_type}")
        request_dir = self.root / request_id
        return RequestOutputPaths(
            request_id=request_id,
            request_dir=request_dir,
            annotated=request_dir / ("annotated.jpg" if media_type == "image" else "annotated.mp4"),
            result_json=request_dir / "result.json",
            crops_dir=request_dir / "crops",
        )

    def _check(self, paths: RequestOutputPaths) -> None:
        validate_request_id(paths.request_id)
        if paths.request_dir.parent != self.root or paths.request_dir.name != paths.request_id:
            raise ValueError("Request directory is outside the managed output root.")
        if self.root.resolve() != self.root:
            raise ValueError("Request output root moved outside the project.")
        if paths.request_dir.is_symlink() or (
            hasattr(paths.request_dir, "is_junction") and paths.request_dir.is_junction()
        ):
            raise ValueError("Request directory may not be a symlink or junction.")
        if paths.request_dir.exists() and paths.request_dir.resolve().parent != self.root.resolve():
            raise ValueError("Request directory resolves outside its managed parent.")

    def cleanup(self, paths: RequestOutputPaths) -> None:
        """Remove only one previously validated request directory."""

        self._check(paths)
        if paths.request_dir.is_dir():
            shutil.rmtree(paths.request_dir)
        elif paths.request_dir.exists():
            raise ValueError("Request output path is not a directory.")

    def prepare(self, paths: RequestOutputPaths) -> None:
        """Reuse of the same id starts from a clean, deterministic namespace."""

        self._check(paths)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cleanup(paths)
        paths.crops_dir.mkdir(parents=True, exist_ok=False)
