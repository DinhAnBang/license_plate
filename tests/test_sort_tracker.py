"""Required Phase T1 behavior tests for the SORT tracker."""

from __future__ import annotations

from core.kalman_box_tracker import KalmanBoxTracker
from core.sort_tracker import SortTracker
from core.tracker import calculate_iou


def det(x1: int, y1: int = 100, width: int = 100, height: int = 50) -> dict:
    return {"conf": 0.9, "box": [x1, y1, x1 + width, y1 + height]}


def ids(items: list[dict]) -> list[int]:
    return [int(item["track_id"]) for item in items]


def test_stationary_and_normal_motion() -> None:
    stationary = SortTracker(iou_threshold=0.3)
    assert ids(stationary.update([det(100)], 1)) == [1]
    assert ids(stationary.update([det(100)], 2)) == [1]
    assert ids(stationary.update([det(100)], 3)) == [1]

    moving = SortTracker(iou_threshold=0.3)
    assert ids(moving.update([det(100)], 1)) == [1]
    assert ids(moving.update([det(110)], 2)) == [1]
    assert ids(moving.update([det(120)], 3)) == [1]


def test_prediction_beats_previous_box_iou() -> None:
    boxes = [det(100), det(120), det(140), det(195)]
    assert calculate_iou(boxes[2]["box"], boxes[3]["box"]) < 0.3

    probe = KalmanBoxTracker(boxes[0], 1, 1)
    for frame, item in enumerate(boxes[1:3], start=2):
        assert probe.predict() is not None
        assert probe.update(item, frame)
    predicted = probe.predict()
    assert predicted is not None
    assert calculate_iou(predicted, boxes[3]["box"]) >= 0.3

    tracker = SortTracker(iou_threshold=0.3)
    assert [ids(tracker.update([item], frame))[0] for frame, item in enumerate(boxes, 1)] == [1, 1, 1, 1]


def test_unrelated_and_one_to_one() -> None:
    tracker = SortTracker(iou_threshold=0.3)
    assert ids(tracker.update([det(10)], 1)) == [1]
    assert ids(tracker.update([det(500)], 2)) == [2]

    two = SortTracker(iou_threshold=0.3)
    assert ids(two.update([det(0), det(300)], 1)) == [1, 2]
    reversed_detections = two.update([det(305), det(5)], 2)
    assert ids(reversed_detections) == [2, 1]

    split = SortTracker(iou_threshold=0.3)
    split.update([det(100)], 1)
    assigned = split.update([det(105), det(115)], 2)
    assert len(set(ids(assigned))) == 2
    assert 1 in ids(assigned) and len(split.active_tracks) == 2


def test_temporary_miss_and_expiry() -> None:
    tracker = SortTracker(iou_threshold=0.3, max_age=2)
    assert ids(tracker.update([det(100)], 1)) == [1]
    assert ids(tracker.update([det(110)], 2)) == [1]
    assert ids(tracker.update([det(120)], 3)) == [1]
    assert tracker.update([], 4) == []
    assert tracker.update([], 5) == []
    assert ids(tracker.update([det(150)], 6)) == [1]

    expiring = SortTracker(iou_threshold=0.3, max_age=2)
    expiring.update([det(100)], 1)
    expiring.update([], 2)
    expiring.update([], 3)
    expiring.update([], 4)
    assert 1 not in expiring.active_tracks
    assert ids(expiring.update([det(100)], 5)) == [2]
    assert expiring.tracks_removed_by_max_age == 1


def test_reset_and_invalid_detection() -> None:
    tracker = SortTracker()
    tracker.update([det(100)], 1)
    tracker.update([det(110)], 2)
    assert tracker.update([{"conf": 0.9, "box": [1, 1, 1, 2]}], 3) == []
    tracker.reset()
    assert ids(tracker.update([det(500)], 1)) == [1]
    assert tracker.created_tracks == 1


def main() -> int:
    test_stationary_and_normal_motion()
    test_prediction_beats_previous_box_iou()
    test_unrelated_and_one_to_one()
    test_temporary_miss_and_expiry()
    test_reset_and_invalid_detection()
    print("SORT tracker lifecycle and association tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
