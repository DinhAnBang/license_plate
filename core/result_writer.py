"""Official, portable JSON writer for final video track results."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2


class VideoResultError(RuntimeError):
    """Raised when an official video result cannot be built or validated."""


class VideoResultWriter:
    """Transform final Phase 5 summaries into the stable Phase 6 JSON schema."""

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

    def prepare(self, source_path: str | Path) -> Path:
        """Remove only the previous JSON for this exact video stem."""

        json_path = self.path_for(source_path)
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if json_path.is_file():
                json_path.unlink()
        except OSError as exc:
            raise VideoResultError(f"Could not prepare JSON path '{json_path}': {exc}") from exc
        return json_path

    def path_for(self, source_path: str | Path) -> Path:
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
    ) -> str:
        """Build, write, reopen, and validate one official video JSON."""

        result = self.build_result(
            source_path=source_path,
            width=width,
            height=height,
            fps=fps,
            frames=frames,
            plates=plates,
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
    ) -> str:
        """Persist an already-built result without reloading it for transport."""

        self.validate_result(result, self.project_root)
        json_path = self.path_for(source_path)
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with json_path.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
        except OSError as exc:
            raise VideoResultError(f"Could not write JSON '{json_path}': {exc}") from exc
        return self._project_relative_path(json_path)

    def build_result(
        self,
        source_path: str | Path,
        width: int,
        height: int,
        fps: float,
        frames: int,
        plates: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Build JSON-safe official data without NumPy, Path, or Track objects."""

        width = self._positive_int(width, "width")
        height = self._positive_int(height, "height")
        fps = self._positive_float(fps, "fps")
        frames = self._non_negative_int(frames, "frames")
        source = self._resolve_source(source_path)

        official_plates: list[dict[str, Any]] = []
        for candidate in plates:
            official_plates.append(
                self._build_plate(
                    candidate,
                    width=width,
                    height=height,
                    fps=fps,
                )
            )
        official_plates.sort(key=lambda plate: (plate["first"], plate["id"]))

        result: dict[str, Any] = {
            "status": "ok",
            "type": "video",
            "source": self._project_relative_path(source),
            "size": {"w": width, "h": height},
            "fps": round(fps, 6),
            "frames": frames,
            "duration": round(frames / fps, 6),
            "count": len(official_plates),
            "plates": official_plates,
        }
        self.validate_result(result, self.project_root)
        return result

    def _build_plate(
        self,
        candidate: Mapping[str, Any],
        width: int,
        height: int,
        fps: float,
    ) -> dict[str, Any]:
        track_id = self._positive_int(candidate.get("track_id"), "track_id")
        first = self._positive_int(candidate.get("first_frame"), "first_frame")
        last = self._positive_int(candidate.get("last_frame"), "last_frame")
        hits = self._positive_int(candidate.get("hits"), "hits")
        best_frame = self._positive_int(candidate.get("best_frame"), "best_frame")
        if last < first:
            raise VideoResultError("track last_frame must be >= first_frame")
        if not first <= best_frame <= last:
            raise VideoResultError("best_frame must be between first_frame and last_frame")

        conf = self._bounded_float(candidate.get("conf"), "conf")
        quality = self._bounded_float(candidate.get("quality"), "quality")
        box = self._box(candidate.get("box"), width, height)
        crop_path = self._validated_crop_path(candidate.get("crop"))
        raw_text = candidate.get("raw_text", "")
        text = candidate.get("text", "")
        if not isinstance(raw_text, str) or not isinstance(text, str):
            raise VideoResultError("OCR text fields must be strings")
        ocr_conf = self._bounded_float(candidate.get("ocr_conf", 0.0), "ocr_conf")

        return {
            "id": track_id,
            "first": first,
            "last": last,
            "hits": hits,
            "best_frame": best_frame,
            "time": round((best_frame - 1) / fps, 6),
            "conf": round(conf, 6),
            "quality": round(quality, 6),
            "box": box,
            "crop": self._project_relative_path(crop_path),
            "raw_text": raw_text,
            "text": text,
            "ocr_conf": round(ocr_conf, 6),
        }

    def _validated_crop_path(self, value: Any) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise VideoResultError("best crop path must be a non-empty string")
        crop_path = Path(value)
        if not crop_path.is_absolute():
            crop_path = self.project_root / crop_path
        crop_path = crop_path.resolve()
        if not crop_path.is_file():
            raise VideoResultError(f"Best crop does not exist: {crop_path}")
        image = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0 or image.shape[0] <= 0 or image.shape[1] <= 0:
            raise VideoResultError(f"Best crop cannot be reopened: {crop_path}")
        return crop_path

    @staticmethod
    def _box(value: Any, width: int, height: int) -> list[int]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
            raise VideoResultError("plate box must contain four coordinates")
        try:
            box = [int(value_part) for value_part in value]
        except (TypeError, ValueError, OverflowError) as exc:
            raise VideoResultError("plate box coordinates must be integers") from exc
        x1, y1, x2, y2 = box
        if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
            raise VideoResultError(f"plate box is outside image bounds: {box}")
        return box

    @staticmethod
    def validate_result(result: Mapping[str, Any], project_root: str | Path) -> None:
        """Validate the official schema and crop references after serialization."""

        required_top = {
            "status",
            "type",
            "source",
            "size",
            "fps",
            "frames",
            "duration",
            "count",
            "plates",
        }
        if not required_top.issubset(result.keys()):
            raise VideoResultError("Official JSON is missing required top-level fields")
        if result["status"] != "ok" or result["type"] != "video":
            raise VideoResultError("Official JSON has invalid status or type")
        size = result["size"]
        if not isinstance(size, Mapping) or size.get("w", 0) <= 0 or size.get("h", 0) <= 0:
            raise VideoResultError("Official JSON has invalid size")
        fps = result["fps"]
        frames = result["frames"]
        duration = result["duration"]
        if not isinstance(fps, (int, float)) or isinstance(fps, bool) or fps <= 0:
            raise VideoResultError("Official JSON has invalid fps")
        if not isinstance(frames, int) or isinstance(frames, bool) or frames < 0:
            raise VideoResultError("Official JSON has invalid frames")
        if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration < 0:
            raise VideoResultError("Official JSON has invalid duration")
        if result["duration"] != round(frames / fps, 6):
            raise VideoResultError("Official JSON duration is inconsistent with frames/fps")

        plates = result["plates"]
        if not isinstance(plates, list) or result["count"] != len(plates):
            raise VideoResultError("Official JSON count does not equal plates length")
        root = Path(project_root).resolve()
        previous_sort_key: tuple[int, int] | None = None
        plate_fields = {
            "id",
            "first",
            "last",
            "hits",
            "best_frame",
            "time",
            "conf",
            "quality",
            "box",
            "crop",
            "raw_text",
            "text",
            "ocr_conf",
        }
        for plate in plates:
            if not isinstance(plate, Mapping) or not plate_fields.issubset(plate.keys()):
                raise VideoResultError("Official JSON plate is missing required fields")
            first = plate["first"]
            last = plate["last"]
            best_frame = plate["best_frame"]
            if not all(isinstance(plate[key], int) and not isinstance(plate[key], bool) for key in ("id", "first", "last", "hits", "best_frame")):
                raise VideoResultError("Official JSON frame and ID fields must be integers")
            if plate["id"] < 1 or first < 1 or last < first or plate["hits"] < 1:
                raise VideoResultError("Official JSON has invalid track metadata")
            if not first <= best_frame <= last:
                raise VideoResultError("Official JSON best_frame is outside track bounds")
            for key in ("time", "conf", "quality", "ocr_conf"):
                if not isinstance(plate[key], (int, float)) or isinstance(plate[key], bool):
                    raise VideoResultError(f"Official JSON {key} must be numeric")
            if not 0.0 <= plate["time"] or not 0.0 <= plate["conf"] <= 1.0 or not 0.0 <= plate["quality"] <= 1.0 or not 0.0 <= plate["ocr_conf"] <= 1.0:
                raise VideoResultError("Official JSON has a value outside its valid range")
            if not isinstance(plate["raw_text"], str) or not isinstance(plate["text"], str):
                raise VideoResultError("Official JSON OCR text fields must be strings")
            if round((best_frame - 1) / fps, 6) != plate["time"]:
                raise VideoResultError("Official JSON time is inconsistent with best_frame/fps")
            box = plate["box"]
            if not isinstance(box, list) or len(box) != 4:
                raise VideoResultError("Official JSON box must be an integer list of length four")
            if not all(isinstance(value, int) and not isinstance(value, bool) for value in box):
                raise VideoResultError("Official JSON box values must be integers")
            if not 0 <= box[0] < box[2] <= size["w"] or not 0 <= box[1] < box[3] <= size["h"]:
                raise VideoResultError("Official JSON box is outside image bounds")
            if not isinstance(plate["crop"], str) or not plate["crop"]:
                raise VideoResultError("Official JSON crop must be a non-empty string")
            crop_path = Path(plate["crop"])
            if not crop_path.is_absolute():
                crop_path = root / crop_path
            if not crop_path.is_file():
                raise VideoResultError(f"Official JSON crop does not exist: {crop_path}")
            if previous_sort_key is not None and (first, plate["id"]) < previous_sort_key:
                raise VideoResultError("Official JSON plates are not sorted by first frame and ID")
            previous_sort_key = (first, plate["id"])

    def _resolve_source(self, source_path: str | Path) -> Path:
        source = Path(source_path)
        if not source.is_absolute():
            source = self.project_root / source
        return source.resolve()

    def _project_relative_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.project_root).as_posix()
        except ValueError:
            return path.resolve().as_posix()

    @staticmethod
    def _positive_int(value: Any, name: str) -> int:
        if isinstance(value, bool):
            raise VideoResultError(f"{name} must be an integer")
        try:
            result = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise VideoResultError(f"{name} must be an integer") from exc
        if result <= 0:
            raise VideoResultError(f"{name} must be greater than zero")
        return result

    @staticmethod
    def _non_negative_int(value: Any, name: str) -> int:
        if isinstance(value, bool):
            raise VideoResultError(f"{name} must be an integer")
        try:
            result = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise VideoResultError(f"{name} must be an integer") from exc
        if result < 0:
            raise VideoResultError(f"{name} must be non-negative")
        return result

    @staticmethod
    def _positive_float(value: Any, name: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise VideoResultError(f"{name} must be numeric") from exc
        if not math.isfinite(result) or result <= 0.0:
            raise VideoResultError(f"{name} must be greater than zero")
        return result

    @staticmethod
    def _bounded_float(value: Any, name: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise VideoResultError(f"{name} must be numeric") from exc
        if not math.isfinite(result) or not 0.0 <= result <= 1.0:
            raise VideoResultError(f"{name} must be between 0 and 1")
        return result
