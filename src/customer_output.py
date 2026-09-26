"""Small customer-facing output derived from the full development result."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2


def _license_text(plate: dict[str, Any]) -> str | None:
    """Return the most useful public plate representation, if any."""

    for key in ("formatted", "text", "normalized_text", "raw_text"):
        value = plate.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def is_customer_result(
    item: dict[str, Any], *, min_confidence: float | None = None,
) -> bool:
    """Return whether one internal row is safe for the release contract."""

    plate = item.get("plate", {})
    confidence = float(plate.get("confidence", 0.0))
    if plate.get("status") != "ok":
        return False
    if plate.get("format_valid") is not True:
        return False
    if not _license_text(plate):
        return False
    return min_confidence is None or confidence >= min_confidence


def build_customer_payload(
    result: dict[str, Any], *, min_confidence: float | None = None,
) -> dict[str, Any]:
    """Reduce the internal result to the release/EXE JSON contract.

    Development output remains unchanged.  The public confidence is the final
    plate/OCR fusion confidence produced by the complete pipeline.
    """

    if min_confidence is not None and not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")

    vehicles: list[dict[str, Any]] = []
    for item in result.get("vehicles", []):
        vehicle = item.get("vehicle", {})
        plate = item.get("plate", {})
        confidence = float(plate.get("confidence", 0.0))
        if not is_customer_result(item, min_confidence=min_confidence):
            continue
        identity = item.get("track_id", item.get("vehicle_index"))
        vehicles.append(
            {
                "track_id": identity,
                "vehicle_type": vehicle.get("class_name"),
                "license": _license_text(plate),
                "confidence": confidence,
                "status": plate.get("status"),
            }
        )

    return {
        "status": result.get("status", "ok"),
        "vehicles": vehicles,
    }


def write_customer_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write UTF-8 customer JSON with stable human-readable formatting."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_customer_annotated_image(
    source: str | Path,
    destination: str | Path,
    result: dict[str, Any],
    *,
    min_confidence: float | None = None,
) -> None:
    """Write a release image containing only accepted plate results."""

    frame = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError(f"Could not decode image: {source}")

    for item in result.get("vehicles", []):
        if not is_customer_result(item, min_confidence=min_confidence):
            continue
        vehicle = item.get("vehicle", {})
        plate = item.get("plate", {})
        vehicle_bbox = vehicle.get("bbox_xyxy")
        plate_bbox = plate.get("best_bbox_xyxy")
        if (
            not isinstance(vehicle_bbox, list)
            or len(vehicle_bbox) != 4
            or not isinstance(plate_bbox, list)
            or len(plate_bbox) != 4
        ):
            continue

        vehicle_box = tuple(int(value) for value in vehicle_bbox)
        plate_box = tuple(int(value) for value in plate_bbox)
        cv2.rectangle(frame, vehicle_box[:2], vehicle_box[2:], (0, 200, 0), 2)
        cv2.rectangle(frame, plate_box[:2], plate_box[2:], (0, 0, 255), 2)
        label = _license_text(plate) or ""
        cv2.putText(
            frame,
            label,
            (max(0, plate_box[0]), max(15, plate_box[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise OSError(f"Could not write annotated image: {output_path}")


__all__ = [
    "build_customer_payload",
    "is_customer_result",
    "write_customer_annotated_image",
    "write_customer_json",
]
