"""Synthetic tests for BYTE two-stage association on SORT motion state."""

from __future__ import annotations

from core.byte_tracker import ByteTracker, split_detections


def det(x1: int, confidence: float, y1: int = 100) -> dict:
    return {"conf": confidence, "box": [x1, y1, x1 + 100, y1 + 50]}


def ids(items: list[dict]) -> list[int]:
    return [int(item["track_id"]) for item in items]


def test_threshold_split() -> None:
    detections = [det(index * 200, score) for index, score in enumerate((0.95, 0.70, 0.699, 0.30, 0.10, 0.099))]
    high, low, discarded = split_detections(detections, low_thresh=0.10, high_thresh=0.70)
    assert [item["conf"] for item in high] == [0.95, 0.70]
    assert [item["conf"] for item in low] == [0.699, 0.30, 0.10]
    assert [item["conf"] for item in discarded] == [0.099]
    for invalid in ((0.7, 0.7), (-0.1, 0.7), (0.1, 1.1)):
        try:
            ByteTracker(track_low_thresh=invalid[0], track_high_thresh=invalid[1])
        except ValueError:
            pass
        else:
            raise AssertionError("invalid BYTE thresholds must fail")


def test_high_match_and_low_recovery() -> None:
    tracker = ByteTracker()
    sequence = [
        det(100, 0.90),
        det(110, 0.85),
        det(120, 0.40),
        det(130, 0.35),
        det(140, 0.88),
    ]
    outputs = [tracker.update([item], frame) for frame, item in enumerate(sequence, 1)]
    assert [ids(items) for items in outputs] == [[1], [1], [1], [1], [1]]
    assert outputs[2][0]["association_stage"] == outputs[3][0]["association_stage"] == "low"
    assert outputs[2][0]["top_k_eligible"] is outputs[3][0]["top_k_eligible"] is False
    assert outputs[2][0]["box"] == sequence[2]["box"]
    assert tracker.high_matches == 2
    assert tracker.low_matches == tracker.low_score_recoveries == 2
    assert tracker.created_tracks == tracker.new_tracks_from_high == 1
    assert tracker.active_tracks[1].hits == 5


def test_low_never_creates_track_or_false_match() -> None:
    low_only = ByteTracker()
    assert low_only.update([det(100, 0.40)], 1) == []
    assert low_only.update([det(105, 0.30)], 2) == []
    assert low_only.created_tracks == 0 and not low_only.active_tracks
    assert low_only.unmatched_low_detections == 2

    tracker = ByteTracker()
    tracker.update([det(10, 0.90)], 1)
    assert tracker.update([det(500, 0.40)], 2) == []
    assert not tracker.active_tracks
    assert tracker.unconfirmed_removed == 1
    assert tracker.low_matches == 0 and tracker.unmatched_low_detections == 1


def test_high_priority_and_one_to_one() -> None:
    priority = ByteTracker()
    priority.update([det(100, 0.90)], 1)
    output = priority.update([det(105, 0.80), det(106, 0.40)], 2)
    assert ids(output) == [1]
    assert output[0]["association_stage"] == "unconfirmed_high"
    assert priority.high_matches == 1 and priority.low_matches == 0
    assert priority.unmatched_low_detections == 1

    tracker = ByteTracker()
    assert ids(tracker.update([det(0, 0.90), det(300, 0.91)], 1)) == [1, 2]
    assert ids(tracker.update([det(3, 0.90), det(303, 0.91)], 2)) == [1, 2]
    output = tracker.update([det(5, 0.80), det(305, 0.40)], 3)
    assert ids(output) == [1, 2]
    assert len(set(ids(output))) == 2
    assert [item["association_stage"] for item in output] == ["high", "low"]


def test_reset() -> None:
    tracker = ByteTracker()
    tracker.update([det(100, 0.9)], 1)
    tracker.update([det(105, 0.9)], 2)
    tracker.update([det(110, 0.4)], 3)
    assert tracker.low_score_recoveries == 1
    tracker.reset()
    assert not tracker.active_tracks and not tracker.finished_tracks
    assert tracker.low_score_recoveries == tracker.created_tracks == 0
    assert ids(tracker.update([det(500, 0.9)], 1)) == [1]


def main() -> int:
    test_threshold_split()
    test_high_match_and_low_recovery()
    test_low_never_creates_track_or_false_match()
    test_high_priority_and_one_to_one()
    test_reset()
    print("BYTE threshold, two-stage association, priority, and reset tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
