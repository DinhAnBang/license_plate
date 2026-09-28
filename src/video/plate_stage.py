"""Detect plate candidates inside tracked video vehicles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.plate_detector import PlateDetector
from ..core.plate_geometry import crop_vehicle_roi, local_bbox_to_global
from .tracking_runtime import TrackedVideoObservation


BBox = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class VideoPlateCandidate:
    """One plate candidate in frame coordinates and its tracked vehicle."""

    frame_index: int
    candidate_index: int
    track_id: int
    vehicle_class_id: int
    vehicle_class_name: str
    vehicle_confidence: float
    vehicle_bbox: BBox
    plate_class_id: int
    plate_class_name: str
    plate_confidence: float
    plate_bbox: BBox
    plate_bbox_before_clamp: BBox | None = None
    coordinate_clamped: bool = False


def _local_bbox_to_global(local_bbox: BBox, parent_bbox: BBox, frame: np.ndarray) -> BBox:
    """Map a plate bbox from vehicle-ROI coordinates to frame coordinates.

    The plate model is expected to return coordinates inside the ROI, but a
    malformed prediction must not escape that ROI. Clamp in local
    coordinates first, then translate. Clamping only to the full frame would
    allow a plate prediction to be assigned outside its vehicle bbox.
    """

    frame_height, frame_width = frame.shape[:2]
    return local_bbox_to_global(
        local_bbox, parent_bbox, frame_width, frame_height, clip_to_vehicle=True,
    )


def detect_frame_plate_candidates(
    observation: TrackedVideoObservation,
    plate_detector: PlateDetector,
) -> tuple[VideoPlateCandidate, ...]:
    """Run the plate model on every accepted tracked vehicle ROI."""

    candidates: list[VideoPlateCandidate] = []
    for tracked in observation.tracked:
        cropped = crop_vehicle_roi(observation.image, tracked.detection.bbox)
        if cropped is None:
            continue
        vehicle_roi, vehicle_bbox = cropped
        for plate in plate_detector.detect(vehicle_roi):
            raw_bbox = (
                vehicle_bbox[0] + int(plate.bbox[0]),
                vehicle_bbox[1] + int(plate.bbox[1]),
                vehicle_bbox[0] + int(plate.bbox[2]),
                vehicle_bbox[1] + int(plate.bbox[3]),
            )
            plate_bbox = _local_bbox_to_global(
                plate.bbox, vehicle_bbox, observation.image,
            )
            candidates.append(
                VideoPlateCandidate(
                    frame_index=observation.frame_index,
                    candidate_index=len(candidates),
                    track_id=tracked.track_id,
                    vehicle_class_id=tracked.detection.class_id,
                    vehicle_class_name=tracked.detection.class_name,
                    vehicle_confidence=float(tracked.detection.confidence),
                    vehicle_bbox=vehicle_bbox,
                    plate_class_id=plate.class_id,
                    plate_class_name=plate.class_name,
                    plate_confidence=float(plate.confidence),
                    plate_bbox=plate_bbox,
                    plate_bbox_before_clamp=raw_bbox,
                    coordinate_clamped=raw_bbox != plate_bbox,
                )
            )
    return tuple(candidates)


__all__ = ["VideoPlateCandidate", "detect_frame_plate_candidates"]
