"""Production video JSON writer and serializer boundary."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .result_serializer import (
    ProductionResultError,
    build_video_result,
    validate_video_result,
)


VideoResultError = ProductionResultError


class VideoResultWriter:
    """Write the same clean video DTO returned by the engine over stdout."""

    def __init__(
        self,
        output_dir: str | Path = "output/json",
        project_root: str | Path | None = None,
    ) -> None:
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.output_dir = Path(output_dir)
        if not self.output_dir.is_absolute():
            self.output_dir = self.project_root / self.output_dir
        self.output_dir = self.output_dir.resolve()

    def prepare(self, source_path: str | Path, *, json_path: str | Path | None = None) -> Path:
        """Remove only the previous JSON at the selected output path."""

        path = self.path_for(source_path, json_path=json_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_file():
                path.unlink()
        except OSError as exc:
            raise VideoResultError(f"Could not prepare JSON path '{path}': {exc}") from exc
        return path

    def path_for(self, source_path: str | Path, *, json_path: str | Path | None = None) -> Path:
        if json_path is not None:
            path = Path(json_path)
            return (path if path.is_absolute() else self.project_root / path).resolve()
        source = Path(source_path)
        return self.output_dir / f"{source.stem}.json"

    def write(
        self,
        source_path: str | Path,
        width: int,
        height: int,
        fps: float,
        frames: int,
        plates: Sequence[Mapping[str, Any]],
        request_id: str | None = None,
        *,
        output_video: str | Path | None = None,
        processing_total_ms: float = 0.0,
        average_frame_ms: float | None = None,
    ) -> str:
        """Build, write, reopen, and validate one production video JSON."""

        result = self.build_result(
            source_path=source_path,
            width=width,
            height=height,
            fps=fps,
            frames=frames,
            plates=plates,
            request_id=request_id,
            output_video=output_video,
            processing_total_ms=processing_total_ms,
            average_frame_ms=average_frame_ms,
        )
        path = self.write_built_result(source_path, result)
        json_path = self.path_for(source_path)
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                reopened = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise VideoResultError(f"Could not reopen JSON '{json_path}': {exc}") from exc
        self.validate_result(reopened, self.project_root)
        return path

    def write_built_result(
        self,
        source_path: str | Path,
        result: Mapping[str, Any],
        *,
        json_path: str | Path | None = None,
    ) -> str:
        """Persist an already-built public DTO without changing its schema."""

        self.validate_result(result, self.project_root)
        path = self.path_for(source_path, json_path=json_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
        except OSError as exc:
            raise VideoResultError(f"Could not write JSON '{path}': {exc}") from exc
        return self._project_relative_path(path)

    def build_result(
        self,
        source_path: str | Path,
        width: int,
        height: int,
        fps: float,
        frames: int,
        plates: Sequence[Mapping[str, Any]],
        request_id: str | None = None,
        *,
        output_video: str | Path | None = None,
        processing_total_ms: float = 0.0,
        average_frame_ms: float | None = None,
    ) -> dict[str, Any]:
        """Build the clean public DTO from internal best-crop summaries."""

        del width, height  # Kept in the API for compatibility; public R1 omits bbox/size.
        source = Path(source_path)
        if output_video is None:
            output_video = self.project_root / "output" / "videos" / f"{source.stem}_tracked.mp4"
        frame_count = int(frames)
        average = (
            float(average_frame_ms)
            if average_frame_ms is not None
            else (float(processing_total_ms) / frame_count if frame_count else 0.0)
        )
        return build_video_result(
            request_id=request_id,
            output_video=output_video,
            fps=fps,
            frames=frames,
            processing_total_ms=processing_total_ms,
            average_frame_ms=average,
            plates=plates,
            project_root=self.project_root,
        )

    @staticmethod
    def validate_result(result: Mapping[str, Any], project_root: str | Path) -> None:
        """Validate the public schema and ensure every crop reference exists."""

        validate_video_result(result)
        del project_root
        for plate in result["plates"]:
            crop_path = Path(plate["crop_path"])
            if not crop_path.is_file():
                raise VideoResultError(f"Best crop does not exist: {crop_path}")

    def _project_relative_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.project_root).as_posix()
        except ValueError:
            return path.resolve().as_posix()
