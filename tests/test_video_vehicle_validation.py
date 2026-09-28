"""Video vehicle validation regressions."""

from src.core.vehicle_detector import VehicleDetection
from src.video.vehicle_validation import (
    VehicleValidationConfig,
    validate_frame_detections,
)


def test_vehicle_validation_keeps_raw_detection_and_explains_rejection():
    detections = [
        VehicleDetection(2, "car", 0.90, (4, 5, 30, 25)),
        VehicleDetection(2, "car", 0.10, (4, 5, 30, 25)),
        VehicleDetection(2, "car", 0.90, (4, 5, 7, 7)),
    ]

    validated = validate_frame_detections(
        detections,
        frame_width=64,
        frame_height=48,
        config=VehicleValidationConfig(min_confidence=0.25),
    )

    assert len(validated) == 3
    assert validated[0].accepted is True
    assert validated[1].reasons == ("low_confidence",)
    assert "too_small" in validated[2].reasons


def test_vehicle_validation_removes_cross_class_duplicate_box():
    detections = [
        VehicleDetection(2, "car", 0.80, (10, 10, 50, 50)),
        VehicleDetection(7, "truck", 0.60, (11, 11, 49, 49)),
    ]

    validated = validate_frame_detections(
        detections,
        frame_width=100,
        frame_height=100,
    )

    assert validated[0].accepted is True
    assert validated[1].accepted is False
    assert validated[1].reasons == ("duplicate_vehicle_box",)


def test_vehicle_validation_rejects_low_confidence_wide_edge_box():
    detections = [
        VehicleDetection(2, "car", 0.44, (4, 80, 718, 1279)),
        VehicleDetection(2, "car", 0.80, (0, 80, 720, 1279)),
    ]

    validated = validate_frame_detections(
        detections,
        frame_width=720,
        frame_height=1280,
        config=VehicleValidationConfig(min_confidence=0.25),
    )

    assert validated[0].accepted is False
    assert validated[0].reasons == ("wide_edge_bbox_low_confidence",)
    assert validated[1].accepted is True


def test_vehicle_validation_default_hides_confidence_below_half():
    validated = validate_frame_detections(
        [VehicleDetection(2, "car", 0.49, (4, 5, 30, 25))],
        frame_width=64,
        frame_height=48,
    )

    assert validated[0].accepted is False
    assert validated[0].reasons == ("low_confidence",)
