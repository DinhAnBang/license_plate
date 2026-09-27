"""Adapt vehicle observations to plate detector ROIs and frame coordinates."""

from __future__ import annotations

import numpy as np

from ..plate_detector import PlateDetection, PlateDetector
from .plate_buffer import BufferedPlateCandidate
from .plate_quality import PlateQualityMetrics
from .plate_types import ImagePlateCandidate
from ..vehicle_detector import VehicleDetection


def crop_vehicle_roi(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Clamp a global vehicle bbox and return its non-empty frame ROI."""

    if not isinstance(frame, np.ndarray) or frame.ndim != 3:
        raise ValueError("frame must be an HxWxC NumPy array")
    frame_height, frame_width = frame.shape[:2]
    if frame_height <= 0 or frame_width <= 0:
        return None

    x1, y1, x2, y2 = (int(value) for value in bbox)
    x1 = int(np.clip(x1, 0, frame_width))
    y1 = int(np.clip(y1, 0, frame_height))
    x2 = int(np.clip(x2, 0, frame_width))
    y2 = int(np.clip(y2, 0, frame_height))
    if x2 <= x1 or y2 <= y1:
        return None

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return roi, (x1, y1, x2, y2)


def local_bbox_to_global(
    local_bbox: tuple[int, int, int, int],
    parent_bbox: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be greater than zero")
    local_x1, local_y1, local_x2, local_y2 = local_bbox
    parent_x1, parent_y1, _, _ = parent_bbox
    global_x1 = int(np.clip(parent_x1 + local_x1, 0, frame_width - 1))
    global_y1 = int(np.clip(parent_y1 + local_y1, 0, frame_height - 1))
    global_x2 = int(np.clip(parent_x1 + local_x2, 0, frame_width))
    global_y2 = int(np.clip(parent_y1 + local_y2, 0, frame_height))
    return global_x1, global_y1, global_x2, global_y2


def detect_vehicle_plates(
    frame: np.ndarray,
    vehicle: VehicleDetection,
    vehicle_index: int,
    plate_detector: PlateDetector,
    frame_index: int,
) -> list[ImagePlateCandidate]:
    """Adapt all post-NMS plate detections to global-frame candidates."""

    frame_height, frame_width = frame.shape[:2]
    cropped = crop_vehicle_roi(frame, vehicle.bbox)
    if cropped is None:
        return []
    vehicle_roi, vehicle_bbox = cropped
    results: list[ImagePlateCandidate] = []
    for plate in plate_detector.detect(vehicle_roi):
        global_bbox = local_bbox_to_global(
            plate.bbox, vehicle_bbox, frame_width, frame_height,
        )
        if global_bbox[2] <= global_bbox[0] or global_bbox[3] <= global_bbox[1]:
            continue
        results.append(
            ImagePlateCandidate(
                frame_index=frame_index,
                vehicle_index=vehicle_index,
                vehicle_class_id=vehicle.class_id,
                vehicle_class_name=vehicle.class_name,
                vehicle_confidence=vehicle.confidence,
                vehicle_bbox=vehicle_bbox,
                plate_class_id=plate.class_id,
                plate_class_name=plate.class_name,
                plate_confidence=plate.confidence,
                plate_bbox=global_bbox,
            )
        )
    return results


def buffered_plate_candidate(
    plate: ImagePlateCandidate,
    crop: np.ndarray,
    quality: PlateQualityMetrics,
) -> BufferedPlateCandidate:
    """Preserve one resolved plate and its crop for image OCR."""

    return BufferedPlateCandidate(
        frame_index=plate.frame_index, vehicle_index=plate.vehicle_index,
        plate_class_id=plate.plate_class_id,
        plate_class_name=plate.plate_class_name,
        plate_confidence=plate.plate_confidence,
        bbox=plate.plate_bbox, quality=quality, crop=crop,
    )
