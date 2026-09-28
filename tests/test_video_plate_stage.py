"""Tests for Module 5 video plate-candidate detection."""

import json
from types import SimpleNamespace

import cv2
import numpy as np

from src.core.plate_detector import PlateDetection
from src.core.vehicle_detector import VehicleDetection
from src.video.plate_stage import (
    VideoPlateCandidate,
    _local_bbox_to_global,
)
from src.video.plate_ownership import VideoPlateDecision, resolve_frame_plate_ownership
from src.video.vehicle_validation import VehicleValidationConfig


class FakeVehicleDetector:
    model_path = "fake-vehicle.onnx"
    confidence_threshold = 0.10
    iou_threshold = 0.45

    def __init__(self):
        self.detect_call_count = 0

    def detect(self, _frame):
        self.detect_call_count += 1
        return [VehicleDetection(2, "car", 0.90, (4, 5, 30, 25))]


class FakePlateDetector:
    model_path = "fake-plate.onnx"
    confidence_threshold = 0.25

    def __init__(self):
        self.detect_call_count = 0

    def detect(self, _vehicle_roi):
        self.detect_call_count += 1
        assert _vehicle_roi.shape[:2] == (20, 26)
        return [PlateDetection(1, "dai", 0.88, (5, 8, 20, 15))]


def _write_video(path, frame_count=2):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48),
    )
    assert writer.isOpened()
    for _ in range(frame_count):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()


def _candidate(track_id, plate_confidence, plate_bbox=(20, 30, 50, 40)):
    return VideoPlateCandidate(
        frame_index=0,
        candidate_index=track_id,
        track_id=track_id,
        vehicle_class_id=2,
        vehicle_class_name="car",
        vehicle_confidence=0.90,
        vehicle_bbox=(10 + track_id, 10, 80 + track_id, 80),
        plate_class_id=1,
        plate_class_name="dai",
        plate_confidence=plate_confidence,
        plate_bbox=plate_bbox,
    )


def test_plate_ownership_marks_cross_track_conflict_without_hiding_raw():
    decisions = resolve_frame_plate_ownership(
        [_candidate(1, 0.80), _candidate(2, 0.90)],
    )

    assert len(decisions) == 2
    assert decisions[0].ownership_status == "rejected_conflict"
    assert decisions[1].ownership_status == "selected"
    assert decisions[0].conflict_group == decisions[1].conflict_group == 0


def test_plate_ownership_prefers_tight_vehicle_when_large_roi_contains_small_vehicle():
    # The same physical plate is detected once in a small vehicle ROI and
    # once in a much larger neighbouring ROI. The larger ROI can have a higher
    # model confidence simply because it has more pixels, but it must not win
    # ownership when the plate is at its extreme left edge.
    small_vehicle = VideoPlateCandidate(
        frame_index=217,
        candidate_index=0,
        track_id=5,
        vehicle_class_id=2,
        vehicle_class_name="car",
        vehicle_confidence=0.78,
        vehicle_bbox=(10, 437, 180, 568),
        plate_class_id=0,
        plate_class_name="vuong",
        plate_confidence=0.36,
        plate_bbox=(100, 492, 130, 511),
    )
    large_neighbour = VideoPlateCandidate(
        frame_index=217,
        candidate_index=1,
        track_id=4,
        vehicle_class_id=2,
        vehicle_class_name="car",
        vehicle_confidence=0.81,
        vehicle_bbox=(92, 406, 720, 1271),
        plate_class_id=0,
        plate_class_name="vuong",
        plate_confidence=0.71,
        plate_bbox=(100, 492, 131, 510),
    )

    decisions = resolve_frame_plate_ownership([small_vehicle, large_neighbour])

    assert decisions[0].ownership_status == "selected"
    assert decisions[1].ownership_status == "rejected_conflict"


def test_same_track_plate_alternatives_recorded_but_only_one_selected():
    decisions = resolve_frame_plate_ownership(
        [_candidate(1, 0.80, (20, 30, 50, 40)), _candidate(1, 0.70, (55, 30, 75, 40))],
    )

    assert len(decisions) == 2
    assert [item.ownership_status for item in decisions] == ["selected", "rejected_same_track"]


def test_two_nearby_vehicles_keep_their_own_distinct_plates():
    first = _candidate(1, 0.7, (20, 30, 50, 40))
    second = _candidate(2, 0.8, (53, 30, 78, 40))
    decisions = resolve_frame_plate_ownership([first, second])
    assert [item.ownership_status for item in decisions] == ["selected", "selected"]


def test_video_ownership_threshold_is_applied_without_changing_default():
    candidates = [
        _candidate(1, 0.7, (20, 30, 50, 40)),
        _candidate(2, 0.8, (30, 30, 60, 40)),
    ]
    default = resolve_frame_plate_ownership(candidates)
    tuned = resolve_frame_plate_ownership(candidates, conflict_iou_threshold=0.45)
    assert [item.ownership_status for item in default] == ["selected", "selected"]
    assert [item.ownership_status for item in tuned] == ["rejected_conflict", "selected"]


def test_clamped_candidate_is_reported_but_never_owned():
    from dataclasses import replace

    candidate = replace(
        _candidate(1, 0.9), plate_bbox_before_clamp=(0, 30, 50, 40),
        coordinate_clamped=True,
    )
    decisions = resolve_frame_plate_ownership([candidate])
    assert decisions[0].ownership_status == "rejected_outside_vehicle"


def test_plate_bbox_is_clamped_inside_vehicle_bbox():
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    mapped = _local_bbox_to_global(
        (-10, -5, 40, 50),
        (4, 5, 30, 25),
        frame,
    )

    assert mapped == (4, 5, 30, 25)
    assert mapped[0] >= 4
    assert mapped[1] >= 5
    assert mapped[2] <= 30
    assert mapped[3] <= 25
