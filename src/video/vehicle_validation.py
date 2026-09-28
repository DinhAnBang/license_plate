"""Validation of per-frame vehicle detections before tracking.

The validator keeps every raw model detection and records why a detection is
accepted or rejected. Rejected boxes remain available for diagnosis.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ..config import VehicleValidationConfig
from ..core.vehicle_detector import VehicleDetection
from ..core.plate_geometry import bbox_iou as _bbox_iou




@dataclass(frozen=True, slots=True)
class ValidatedVehicleDetection:
    detection: VehicleDetection
    accepted: bool
    reasons: tuple[str, ...]


_ALLOWED_CLASSES = {"car", "motorcycle", "bus", "truck"}


def validate_frame_detections(
    detections: tuple[VehicleDetection, ...] | list[VehicleDetection],
    *,
    frame_width: int,
    frame_height: int,
    config: VehicleValidationConfig | None = None,
) -> tuple[ValidatedVehicleDetection, ...]:
    """Validate detections without hiding any model output."""

    selected = config or VehicleValidationConfig()
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")

    result: list[ValidatedVehicleDetection] = []
    frame_area = frame_width * frame_height
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        width = x2 - x1
        height = y2 - y1
        reasons: list[str] = []
        if detection.class_name not in _ALLOWED_CLASSES:
            reasons.append("unknown_vehicle_class")
        if float(detection.confidence) < selected.min_confidence:
            reasons.append("low_confidence")
        if x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height:
            reasons.append("bbox_out_of_frame")
        if width < selected.min_width_pixels or height < selected.min_height_pixels:
            reasons.append("too_small")
        is_wide_edge_box = (
            width / frame_width >= selected.wide_edge_bbox_ratio
            and x1 <= frame_width * 0.02
            and frame_width - x2 <= frame_width * 0.02
        )
        if (
            is_wide_edge_box
            and float(detection.confidence) < selected.wide_edge_min_confidence
        ):
            reasons.append("wide_edge_bbox_low_confidence")
        if width > 0 and height > 0:
            area_ratio = (width * height) / frame_area
            aspect_ratio = width / height
            if area_ratio < selected.min_area_ratio:
                reasons.append("too_small")
            if not selected.min_aspect_ratio <= aspect_ratio <= selected.max_aspect_ratio:
                reasons.append("abnormal_aspect_ratio")
        else:
            reasons.append("invalid_bbox")

        unique_reasons = tuple(dict.fromkeys(reasons))
        result.append(
            ValidatedVehicleDetection(
                detection=detection,
                accepted=not unique_reasons,
                reasons=unique_reasons,
            )
        )
    kept: list[ValidatedVehicleDetection] = []
    final: list[ValidatedVehicleDetection | None] = [None] * len(result)
    for index in sorted(
        range(len(result)),
        key=lambda item: float(result[item].detection.confidence),
        reverse=True,
    ):
        item = result[index]
        if not item.accepted:
            final[index] = item
            continue
        duplicate = any(
            _bbox_iou(item.detection.bbox, previous.detection.bbox)
            >= selected.duplicate_iou_threshold
            for previous in kept
        )
        if duplicate:
            final[index] = ValidatedVehicleDetection(
                detection=item.detection,
                accepted=False,
                reasons=(*item.reasons, "duplicate_vehicle_box"),
            )
        else:
            kept.append(item)
            final[index] = item
    return tuple(item for item in final if item is not None)


def rejection_reason_counts(
    validated: tuple[ValidatedVehicleDetection, ...] | list[ValidatedVehicleDetection],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in validated:
        counts.update(item.reasons)
    return dict(sorted(counts.items()))


__all__ = [
    "ValidatedVehicleDetection",
    "VehicleValidationConfig",
    "rejection_reason_counts",
    "validate_frame_detections",
]
