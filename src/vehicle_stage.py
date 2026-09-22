"""Vehicle detection and tracker setup shared by image/video execution."""

from __future__ import annotations

import numpy as np

from .config import TrackingConfig
from .tracking import ByteTracker
from .vehicle_detector import VehicleDetection, VehicleDetector


def detect_image_vehicles(
    image: np.ndarray, detector: VehicleDetector, image_confidence: float,
) -> list[VehicleDetection]:
    return [
        vehicle for vehicle in detector.detect(image)
        if vehicle.confidence >= image_confidence
    ]


def create_video_tracker(
    fps: float, detector_confidence: float, config: TrackingConfig, debug: bool,
) -> ByteTracker:
    return ByteTracker(
        fps=fps, track_low_threshold=detector_confidence,
        track_high_threshold=config.high_threshold,
        new_track_threshold=config.new_track_threshold,
        match_cost_threshold=config.match_cost_threshold,
        second_match_cost_threshold=config.second_match_cost_threshold,
        unconfirmed_match_cost_threshold=config.unconfirmed_match_cost_threshold,
        track_buffer_seconds=config.track_buffer_seconds,
        min_confirmed_hits=config.min_confirmed_hits,
        duplicate_iou_threshold=config.duplicate_iou_threshold,
        cross_class_duplicate_iou_threshold=config.cross_class_duplicate_iou_threshold,
        cross_class_duplicate_area_ratio_threshold=config.cross_class_duplicate_area_ratio_threshold,
        debug=debug,
    )
