"""Vehicle detection adapter for still-image execution."""

from __future__ import annotations

import numpy as np

from ..core.vehicle_detector import VehicleDetection, VehicleDetector


def detect_image_vehicles(
    image: np.ndarray, detector: VehicleDetector, image_confidence: float,
) -> list[VehicleDetection]:
    return [
        vehicle for vehicle in detector.detect(image)
        if vehicle.confidence >= image_confidence
    ]
