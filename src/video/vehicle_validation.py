"""Validation of per-frame vehicle detections before tracking.

The validator keeps every raw model detection and records why a detection is
accepted or rejected. Rejected boxes remain available for diagnosis.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ..vehicle_detector import VehicleDetection


@dataclass(frozen=True, slots=True)
class VehicleValidationConfig:
    """Conservative geometry and confidence checks for video detections."""

    min_confidence: float = 0.50
    min_width_pixels: int = 8
    min_height_pixels: int = 8
    min_area_ratio: float = 0.00002
    min_aspect_ratio: float = 0.03
    max_aspect_ratio: float = 12.0
    duplicate_iou_threshold: float = 0.75
    wide_edge_bbox_ratio: float = 0.98
    wide_edge_min_confidence: float = 0.50

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if self.min_width_pixels < 1 or self.min_height_pixels < 1:
            raise ValueError("minimum bbox dimensions must be positive")
        if self.min_area_ratio < 0.0:
            raise ValueError("min_area_ratio must be non-negative")
        if not 0.0 < self.min_aspect_ratio <= self.max_aspect_ratio:
            raise ValueError("invalid aspect ratio limits")
        if not 0.0 <= self.duplicate_iou_threshold <= 1.0:
            raise ValueError("duplicate_iou_threshold must be between 0 and 1")
        if not 0.0 < self.wide_edge_bbox_ratio <= 1.0:
            raise ValueError("wide_edge_bbox_ratio must be between 0 and 1")
        if not 0.0 <= self.wide_edge_min_confidence <= 1.0:
            raise ValueError("wide_edge_min_confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class ValidatedVehicleDetection:
    detection: VehicleDetection
    accepted: bool
    reasons: tuple[str, ...]


_ALLOWED_CLASSES = {"car", "motorcycle", "bus", "truck"}


def _bbox_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    first_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    second_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


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
