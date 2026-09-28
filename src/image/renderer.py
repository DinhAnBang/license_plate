"""Image annotation helpers; no detection or OCR decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..core.customer_result import is_customer_result, license_text


def draw_box(frame: np.ndarray, bbox: tuple[int, int, int, int], label: str, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = (int(value) for value in bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (max(0, x1), max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def plate_display_label(plate: dict[str, Any]) -> str:
    for key in ("formatted", "text", "normalized_text", "raw_text"):
        value = plate.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unreadable"


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
        label = license_text(plate) or ""
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
