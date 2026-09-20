"""Production JSON DTOs for image and video responses.

The processors keep richer internal dictionaries for tracking, quality, OCR
voting, and diagnostics. This module is the single public-result boundary used
by both the persistent stdout response and ``result.json``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


IMAGE_TOP_LEVEL_FIELDS = frozenset(
    {"status", "request_id", "input_type", "output_image", "processing", "count", "plates"}
)
IMAGE_PLATE_FIELDS = frozenset(
    {"plate_text", "detection_confidence", "ocr_confidence", "crop_path"}
)
VIDEO_TOP_LEVEL_FIELDS = frozenset(
    {"status", "request_id", "input_type", "output_video", "processing", "count", "plates"}
)
VIDEO_PROCESSING_FIELDS = frozenset({"total_ms", "frames", "average_frame_ms"})
VIDEO_PLATE_FIELDS = frozenset(
    {
        "plate_text",
        "detection_confidence",
        "ocr_confidence",
        "first_detected_frame",
        "last_detected_frame",
        "first_detected_time_sec",
        "last_detected_time_sec",
        "best_frame_index",
        "best_frame_time_sec",
        "crop_path",
    }
)


class ProductionResultError(ValueError):
    """Raised when an internal result cannot be mapped to the public contract."""


def build_image_result(
    *,
    request_id: str | None,
    output_image: str | Path,
    processing_total_ms: float,
    plates: Sequence[Mapping[str, Any]],
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Map internal image plate dictionaries to the public image DTO."""

    result = {
        "status": "success",
        "request_id": request_id,
        "input_type": "image",
        "output_image": _absolute_path(output_image, project_root),
        "processing": {"total_ms": _finite_non_negative(processing_total_ms, "total_ms")},
        "count": len(plates),
        "plates": [
            {
                "plate_text": _string(candidate.get("text", ""), "text"),
                "detection_confidence": _bounded_float(
                    candidate.get("conf"), "conf"
                ),
                "ocr_confidence": _bounded_float(
                    candidate.get("ocr_conf", 0.0), "ocr_conf"
                ),
                "crop_path": _absolute_path(candidate.get("crop"), project_root),
            }
            for candidate in plates
        ],
    }
    validate_image_result(result)
    return result


def build_video_result(
    *,
    request_id: str | None,
    output_video: str | Path,
    fps: float,
    frames: int,
    processing_total_ms: float,
    average_frame_ms: float,
    plates: Sequence[Mapping[str, Any]],
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Map internal best-crop dictionaries to the public video DTO."""

    fps_value = _positive_float(fps, "fps")
    frame_count = _non_negative_int(frames, "frames")
    public_plates: list[dict[str, Any]] = []
    for candidate in plates:
        first = _positive_int(candidate.get("first_frame"), "first_frame")
        last = _positive_int(candidate.get("last_frame"), "last_frame")
        best = _positive_int(candidate.get("best_frame"), "best_frame")
        if last < first or not first <= best <= last:
            raise ProductionResultError(
                "best_frame must be inside the real detection interval"
            )
        public_plates.append(
            {
                "plate_text": _string(candidate.get("text", ""), "text"),
                # ``conf`` belongs to the same internal best-candidate object
                # as ``best_frame`` and ``crop``.
                "detection_confidence": _bounded_float(candidate.get("conf"), "conf"),
                "ocr_confidence": _bounded_float(
                    candidate.get("ocr_conf", 0.0), "ocr_conf"
                ),
                "first_detected_frame": first,
                "last_detected_frame": last,
                "first_detected_time_sec": _frame_time(first, fps_value),
                "last_detected_time_sec": _frame_time(last, fps_value),
                "best_frame_index": best,
                "best_frame_time_sec": _frame_time(best, fps_value),
                "crop_path": _absolute_path(candidate.get("crop"), project_root),
            }
        )

    public_plates.sort(
        key=lambda plate: (plate["first_detected_frame"], plate["best_frame_index"])
    )
    total_ms = _finite_non_negative(processing_total_ms, "total_ms")
    # The public average is derived from the public total, so the two fields
    # remain mathematically consistent after JSON precision normalization.
    average_ms = round(total_ms / frame_count, 6) if frame_count else 0.0
    result = {
        "status": "success",
        "request_id": request_id,
        "input_type": "video",
        "output_video": _absolute_path(output_video, project_root),
        "processing": {
            "total_ms": total_ms,
            "frames": frame_count,
            "average_frame_ms": average_ms,
        },
        "count": len(public_plates),
        "plates": public_plates,
    }
    validate_video_result(result)
    return result


def validate_image_result(result: Mapping[str, Any]) -> None:
    _validate_exact_fields(result, IMAGE_TOP_LEVEL_FIELDS, "image top-level")
    _validate_common_result(result, "image")
    if not isinstance(result["output_image"], str) or not Path(result["output_image"]).is_absolute():
        raise ProductionResultError("image output_image must be an absolute path")
    processing = result["processing"]
    if not isinstance(processing, Mapping) or set(processing) != {"total_ms"}:
        raise ProductionResultError("image processing must contain only total_ms")
    _finite_non_negative(processing["total_ms"], "total_ms")
    _validate_plates(result["plates"], IMAGE_PLATE_FIELDS, "image")
    for plate in result["plates"]:
        _validate_path(plate["crop_path"], "crop_path")


def validate_video_result(result: Mapping[str, Any]) -> None:
    _validate_exact_fields(result, VIDEO_TOP_LEVEL_FIELDS, "video top-level")
    _validate_common_result(result, "video")
    if not isinstance(result["output_video"], str) or not Path(result["output_video"]).is_absolute():
        raise ProductionResultError("video output_video must be an absolute path")
    processing = result["processing"]
    if not isinstance(processing, Mapping) or set(processing) != VIDEO_PROCESSING_FIELDS:
        raise ProductionResultError(
            "video processing must contain total_ms, frames, and average_frame_ms"
        )
    _finite_non_negative(processing["total_ms"], "total_ms")
    _non_negative_int(processing["frames"], "frames")
    _finite_non_negative(processing["average_frame_ms"], "average_frame_ms")
    _validate_plates(result["plates"], VIDEO_PLATE_FIELDS, "video")
    for plate in result["plates"]:
        _validate_path(plate["crop_path"], "crop_path")
        first = _positive_int(plate["first_detected_frame"], "first_detected_frame")
        last = _positive_int(plate["last_detected_frame"], "last_detected_frame")
        best = _positive_int(plate["best_frame_index"], "best_frame_index")
        if last < first or not first <= best <= last:
            raise ProductionResultError("video detection interval is invalid")
        for field in (
            "first_detected_time_sec",
            "last_detected_time_sec",
            "best_frame_time_sec",
        ):
            _finite_non_negative(plate[field], field)


def _validate_common_result(result: Mapping[str, Any], input_type: str) -> None:
    if result["status"] != "success" or result["input_type"] != input_type:
        raise ProductionResultError("invalid production result status or input_type")
    if result["request_id"] is not None and (
        not isinstance(result["request_id"], str) or not result["request_id"]
    ):
        raise ProductionResultError("request_id must be a non-empty string or null")
    if not isinstance(result["count"], int) or isinstance(result["count"], bool):
        raise ProductionResultError("count must be an integer")
    if result["count"] < 0 or not isinstance(result["plates"], list):
        raise ProductionResultError("invalid count or plates")
    if result["count"] != len(result["plates"]):
        raise ProductionResultError("count must equal len(plates)")


def _validate_exact_fields(
    result: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    if set(result) != expected:
        raise ProductionResultError(
            f"{label} fields mismatch: expected {sorted(expected)}, found {sorted(result)}"
        )


def _validate_plates(
    plates: Any, expected: frozenset[str], input_type: str
) -> None:
    if not isinstance(plates, list):
        raise ProductionResultError(f"{input_type} plates must be a list")
    for plate in plates:
        if not isinstance(plate, Mapping) or set(plate) != expected:
            raise ProductionResultError(f"{input_type} plate fields mismatch")
        _string(plate["plate_text"], "plate_text")
        _bounded_float(plate["detection_confidence"], "detection_confidence")
        _bounded_float(plate["ocr_confidence"], "ocr_confidence")


def _validate_path(value: Any, name: str) -> None:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ProductionResultError(f"{name} must be an absolute path")


def _absolute_path(value: Any, project_root: str | Path | None = None) -> str:
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str) and value.strip():
        path = Path(value)
    else:
        raise ProductionResultError("path must be a non-empty string or Path")
    if not path.is_absolute() and project_root is not None:
        path = Path(project_root) / path
    return str(path.resolve())


def _frame_time(frame: int, fps: float) -> float:
    return round((frame - 1) / fps, 6)


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ProductionResultError(f"{name} must be a string")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ProductionResultError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProductionResultError(f"{name} must be an integer") from exc
    if result < 1:
        raise ProductionResultError(f"{name} must be positive")
    return result


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ProductionResultError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProductionResultError(f"{name} must be an integer") from exc
    if result < 0:
        raise ProductionResultError(f"{name} must be non-negative")
    return result


def _positive_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProductionResultError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ProductionResultError(f"{name} must be positive")
    return result


def _finite_non_negative(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProductionResultError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ProductionResultError(f"{name} must be finite and non-negative")
    return round(result, 6)


def _bounded_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProductionResultError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ProductionResultError(f"{name} must be between 0 and 1")
    return round(result, 6)
