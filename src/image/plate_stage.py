"""Adapt vehicle observations to plate detector ROIs and frame coordinates."""

from __future__ import annotations

import numpy as np

from ..core.plate_detector import PlateDetector
from ..core.plate_geometry import crop_vehicle_roi, local_bbox_to_global
from .plate_buffer import BufferedPlateCandidate
from ..core.plate_quality import PlateQualityMetrics
from .plate_types import ImagePlateCandidate
from ..core.vehicle_detector import VehicleDetection


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
