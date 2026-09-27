"""Tests for Module 2 vehicle detection orchestration."""

import json

import cv2
import numpy as np

from src.vehicle_detector import VehicleDetection
from src.video.vehicle_stage import detect_frame_vehicles, run_video_vehicle_detection


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
    assert (output_dir / "source_vehicles_annotated.mp4").is_file()
