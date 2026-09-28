"""Tests for Module 4 stable vehicle tracking."""

import numpy as np
import cv2
import json

from src.vehicle_detector import VehicleDetection
from src.video.tracking import VehicleTracker, VehicleTrackingConfig
from src.video.tracking_stage import run_video_vehicle_tracking


def detection(x1, y1, x2, y2, class_id=2, name="car", confidence=0.9):
    return VehicleDetection(class_id, name, confidence, (x1, y1, x2, y2))


def sig(value):
    return np.full(64, value, dtype=np.float32)


def test_same_vehicle_keeps_one_id():
    tracker = VehicleTracker(fps=10.0)
    first = tracker.update([detection(10, 10, 50, 40)], [sig(1.0)], frame_index=0)[0]
    second = tracker.update([detection(12, 11, 52, 41)], [sig(1.0)], frame_index=1)[0]

    assert first.track_id == second.track_id == 1
    assert tracker.created_track_count == 1


def test_two_nearby_vehicles_are_assigned_globally():
    tracker = VehicleTracker(fps=10.0)
    first = tracker.update(
        [detection(10, 10, 50, 40), detection(80, 10, 120, 40)],
        [sig(1.0), sig(2.0)],
        frame_index=0,
    )
    second = tracker.update(
        [detection(82, 11, 122, 41), detection(12, 11, 52, 41)],
        [sig(2.0), sig(1.0)],
        frame_index=1,
    )

    assert second[0].track_id == first[1].track_id
    assert second[1].track_id == first[0].track_id
    assert tracker.created_track_count == 2


def test_short_occlusion_keeps_track_alive():
    tracker = VehicleTracker(
        fps=10.0,
        config=VehicleTrackingConfig(max_lost_seconds=0.5),
    )
    original = tracker.update([detection(10, 10, 50, 40)], [sig(1.0)], frame_index=0)[0]
    assert tracker.update([], [], frame_index=1) == ()
    assert tracker.update([], [], frame_index=2) == ()
    recovered = tracker.update([detection(12, 11, 52, 41)], [sig(1.0)], frame_index=3)[0]

    assert recovered.track_id == original.track_id
    assert tracker.created_track_count == 1


def test_removed_track_is_not_resurrected_by_identical_appearance():
    tracker = VehicleTracker(
        fps=10.0,
        config=VehicleTrackingConfig(
            max_lost_seconds=0.1,
            archive_seconds=5.0,
            reidentification_similarity=0.80,
        ),
    )
    original = tracker.update([detection(10, 10, 50, 40)], [sig(1.0)], frame_index=0)[0]
    tracker.update([], [], frame_index=1)
    tracker.update([], [], frame_index=2)
    recovered = tracker.update([detection(200, 100, 250, 140)], [sig(1.0)], frame_index=3)[0]

    assert recovered.track_id != original.track_id
    assert recovered.reidentified is False
    assert tracker.created_track_count == 2
    assert tracker.reidentified_count == 0
    assert original.track_id in tracker.removed_track_ids


def test_weak_detection_rescues_existing_track_without_creating_new_one():
    tracker = VehicleTracker(fps=10.0)
    confidences = [0.8, 0.7, 0.3, 0.25, 0.8]
    ids = []
    for frame_index, confidence in enumerate(confidences):
        visible = tracker.update(
            [detection(10 + frame_index * 2, 10, 50 + frame_index * 2, 40, confidence=confidence)],
            frame_index=frame_index,
        )
        ids.append(visible[0].track_id)
    assert ids == [1] * 5
    assert tracker.counters["second_stage_matches"] == 2
    assert tracker.created_track_count == 1
    assert tracker.update(
        [detection(200, 10, 240, 40, confidence=0.3)], frame_index=5
    ) == ()
    assert tracker.created_track_count == 1


def test_high_match_threshold_does_not_force_new_track_creation():
    tracker = VehicleTracker(fps=10.0)
    assert tracker.update(
        [detection(10, 10, 50, 40, confidence=0.54)], frame_index=0
    ) == ()
    created = tracker.update(
        [detection(11, 10, 51, 40, confidence=0.56)], frame_index=1
    )
    assert len(created) == 1
    assert created[0].track_id == 1


def test_lost_reactivated_then_removed_after_fps_scaled_timeout():
    tracker = VehicleTracker(fps=10.0, config=VehicleTrackingConfig(max_lost_seconds=0.3))
    first = tracker.update([detection(10, 10, 50, 40)], frame_index=0)[0]
    for frame_index in (1, 2, 3):
        assert tracker.update([], frame_index=frame_index) == ()
        assert tracker.lost_track_ids == (first.track_id,)
    recovered = tracker.update([detection(11, 10, 51, 40)], frame_index=4)[0]
    assert recovered.track_id == first.track_id
    assert recovered.reidentified
    for frame_index in (5, 6, 7, 8):
        tracker.update([], frame_index=frame_index)
    assert tracker.lost_track_ids == ()
    assert tracker.removed_track_ids == (first.track_id,)
    assert tracker.history[first.track_id]["removal_reason"] == "timeout"


def test_nearby_crossing_vehicles_keep_unique_ids():
    tracker = VehicleTracker(fps=10.0)
    start = tracker.update(
        [detection(10, 10, 50, 40), detection(70, 10, 110, 40)],
        frame_index=0,
    )
    assert [item.track_id for item in start] == [1, 2]
    for frame_index, left, right in [(1, 20, 60), (2, 30, 50), (3, 40, 40)]:
        visible = tracker.update(
            [detection(left, 10, left + 40, 40), detection(right, 10, right + 40, 40)],
            frame_index=frame_index,
        )
        assert len({item.track_id for item in visible}) == 2
    assert tracker.created_track_count == 2


def test_crossing_vehicles_keep_direction_after_overlap():
    tracker = VehicleTracker(fps=10.0)
    for frame_index, (left_to_right, right_to_left) in enumerate(
        ((10, 70), (20, 60), (30, 50), (40, 40), (50, 30), (60, 20))
    ):
        visible = tracker.update(
            [
                detection(left_to_right, 10, left_to_right + 40, 40),
                detection(right_to_left, 10, right_to_left + 40, 40),
            ],
            frame_index=frame_index,
        )
        assert [item.track_id for item in visible] == [1, 2]
    assert tracker.created_track_count == 2


def test_duplicate_detection_cannot_create_second_track():
    tracker = VehicleTracker(fps=10.0)
    visible = tracker.update(
        [detection(10, 10, 50, 40), detection(11, 10, 51, 40)],
        frame_index=0,
    )
    assert len(visible) == 1
    assert tracker.created_track_count == 1
    assert tracker.counters["duplicate_detections_suppressed"] == 1


def test_one_detection_cannot_update_two_existing_tracks():
    tracker = VehicleTracker(fps=10.0)
    tracker.update(
        [detection(10, 10, 50, 40), detection(40, 10, 80, 40)],
        frame_index=0,
    )
    visible = tracker.update(
        [detection(25, 10, 65, 40)],
        frame_index=1,
    )
    assert len(visible) == 1
    assert len(tracker.lost_track_ids) == 1
    assert len(tracker.active_track_ids) == 1


def test_persistent_duplicate_tracks_are_removed_with_reason():
    tracker = VehicleTracker(fps=10.0)
    tracker.update(
        [detection(10, 10, 50, 40), detection(40, 10, 80, 40)],
        frame_index=0,
    )
    tracker.update(
        [detection(25, 10, 65, 40), detection(26, 10, 66, 40)],
        frame_index=1,
    )
    visible = tracker.update(
        [detection(26, 10, 66, 40), detection(27, 10, 67, 40)],
        frame_index=2,
    )
    assert len(visible) == 1
    assert tracker.counters["duplicate_tracks_removed"] == 1
    assert len(tracker.removed_track_ids) == 1
    assert tracker.history[tracker.removed_track_ids[0]]["removal_reason"] == "duplicate"


def test_kalman_predicts_motion_before_matching():
    tracker = VehicleTracker(fps=10.0)
    for frame_index, left in enumerate((10, 20, 30)):
        tracker.update([detection(left, 10, left + 40, 40)], frame_index=frame_index)
    track = tracker._tracks[1]
    prior = track.detection.bbox
    track.predict(tracker.kalman)
    assert track.predicted_bbox[0] > prior[0]


def test_video_runtime_passes_weak_detections_to_tracker_and_keeps_json_contract(tmp_path):
    class Detector:
        model_path = "fake.onnx"
        confidence_threshold = 0.10

        def __init__(self):
            self.calls = 0

        def detect(self, _frame):
            confidence = (0.8, 0.3, 0.25, 0.8)[self.calls]
            self.calls += 1
            return [detection(10, 10, 50, 40, confidence=confidence)]

    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    assert writer.isOpened()
    for _ in range(4):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    output = run_video_vehicle_tracking(
        video, tmp_path / "tracking", detector=Detector(), save_annotated=False
    )
    payload = json.loads((tmp_path / "tracking" / "clip_tracking.json").read_text())
    assert output["summary"]["unique_track_ids_created"] == 1
    assert output["summary"]["tracking_counters"]["second_stage_matches"] == 2
    assert [frame["detections"][0]["track_id"] for frame in payload["frames"]] == [1] * 4
    assert [frame["detections"][0]["confidence"] for frame in payload["frames"]] == [0.8, 0.3, 0.25, 0.8]


def test_different_classes_do_not_share_track():
    tracker = VehicleTracker(fps=10.0)
    car = tracker.update([detection(10, 10, 50, 40)], [sig(1.0)], frame_index=0)[0]
    motorcycle = tracker.update(
        [detection(200, 11, 240, 41, class_id=3, name="motorcycle")],
        [sig(1.0)],
        frame_index=1,
    )[0]

    assert motorcycle.track_id != car.track_id


def test_class_flip_keeps_same_vehicle_track():
    tracker = VehicleTracker(fps=10.0)
    first = tracker.update(
        [detection(10, 10, 80, 60, class_id=7, name="truck")],
        [sig(1.0)],
        frame_index=0,
    )[0]
    second = tracker.update(
        [detection(12, 11, 82, 61, class_id=2, name="car")],
        [sig(1.0)],
        frame_index=1,
    )[0]

    assert second.track_id == first.track_id
    assert tracker.created_track_count == 1
