"""Tests for Module 2 vehicle detection orchestration."""

import json

import cv2
import numpy as np

from src.vehicle_detector import VehicleDetection
from src.video.vehicle_stage import detect_frame_vehicles, run_video_vehicle_detection
from src.video.vehicle_validation import (
    VehicleValidationConfig,
    validate_frame_detections,
)


class FakeVehicleDetector:
    model_path = "fake-yolo26.onnx"
    detect_call_count = 0

    def detect(self, frame):
        self.detect_call_count += 1
        return [VehicleDetection(2, "car", 0.91, (4, 5, 30, 25))]


def _write_video(path, frame_count=2):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48),
    )
    assert writer.isOpened()
    for _ in range(frame_count):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()


def test_detect_frame_vehicles_keeps_detector_results():
    detector = FakeVehicleDetector()
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    detections = detect_frame_vehicles(frame, detector)

    assert detections[0].class_name == "car"
    assert detections[0].bbox == (4, 5, 30, 25)
    assert detector.detect_call_count == 1


def test_video_vehicle_detection_writes_json_and_annotation(tmp_path):
    source = tmp_path / "source.mp4"
    _write_video(source)
    output_dir = tmp_path / "output"

    result = run_video_vehicle_detection(
        source, output_dir, detector=FakeVehicleDetector(), save_annotated=True,
    )

    payload = json.loads((output_dir / "source_vehicles.json").read_text())
    assert result["summary"]["frames_read"] == 2
    assert result["summary"]["total_vehicle_detections"] == 2
    assert payload["frames"][0]["detections"][0]["vehicle_index"] == 0
    assert payload["frames"][0]["raw_detections"][0]["accepted"] is True
    assert payload["summary"]["rejected_vehicle_detections"] == 0
    assert (output_dir / "source_vehicles_annotated.mp4").is_file()


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
