"""Synthetic tests for the Phase 4 IoU tracker."""

from __future__ import annotations

from core.tracker import PlateTracker, calculate_iou


def main() -> int:
    assert abs(calculate_iou([0, 0, 10, 10], [0, 0, 10, 10]) - 1.0) < 1e-9
    assert calculate_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert calculate_iou([0, 0, 0, 10], [0, 0, 10, 10]) == 0.0

    tracker = PlateTracker(iou_threshold=0.3, max_missed=10)
    frame_1 = tracker.update(
        [{"conf": 0.9, "box": [100, 100, 200, 150]}],
        frame_index=1,
    )
    assert [item["track_id"] for item in frame_1] == [1]

    frame_2 = tracker.update(
        [
            {"conf": 0.91, "box": [105, 102, 205, 152]},
            {"conf": 0.88, "box": [500, 500, 600, 550]},
        ],
        frame_index=2,
    )
    assert [item["track_id"] for item in frame_2] == [1, 2]

    assert tracker.update([], frame_index=3) == []
    frame_4 = tracker.update(
        [{"conf": 0.92, "box": [110, 104, 210, 154]}],
        frame_index=4,
    )
    assert [item["track_id"] for item in frame_4] == [1]

    expiring_tracker = PlateTracker(iou_threshold=0.3, max_missed=2)
    expiring_tracker.update([{"conf": 0.9, "box": [10, 10, 50, 40]}], 1)
    expiring_tracker.update([], 2)
    expiring_tracker.update([], 3)
    expiring_tracker.update([], 4)
    assert 1 in [track.track_id for track in expiring_tracker.finished_tracks]
    new_track = expiring_tracker.update(
        [{"conf": 0.9, "box": [10, 10, 50, 40]}],
        5,
    )
    assert [item["track_id"] for item in new_track] == [2]
    expiring_tracker.finalize()

    one_to_one_tracker = PlateTracker(iou_threshold=0.3, max_missed=10)
    one_to_one_tracker.update(
        [
            {"conf": 0.9, "box": [0, 0, 100, 100]},
            {"conf": 0.9, "box": [10, 0, 110, 100]},
        ],
        frame_index=1,
    )
    shared_detection = one_to_one_tracker.update(
        [{"conf": 0.9, "box": [5, 0, 105, 100]}],
        frame_index=2,
    )
    assert len(shared_detection) == 1
    assert len(one_to_one_tracker.active_tracks) == 2
    assert sum(track.missed == 1 for track in one_to_one_tracker.active_tracks.values()) == 1

    print("Tracker synthetic tests: OK")
    print("Temporary miss preserved ID: 1")
    print("Expired track received new ID: 2")
    print("One-to-one matching: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
