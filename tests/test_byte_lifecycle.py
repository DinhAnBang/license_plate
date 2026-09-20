"""Phase T3 BYTE lifecycle, confirmation, reactivation, and FPS tests."""

from __future__ import annotations

from core.byte_tracker import ByteTracker, TrackState


def det(x1: int, confidence: float) -> dict:
    return {"conf": confidence, "box": [x1, 100, x1 + 100, 150]}


def confirm_track(tracker: ByteTracker) -> None:
    first = tracker.update([det(100, 0.85)], 1)
    assert first[0]["track_id"] == 1
    assert tracker.unconfirmed_tracks[1].state is TrackState.NEW
    second = tracker.update([det(104, 0.82)], 2)
    assert second[0]["track_id"] == 1
    assert tracker.tracked_tracks[1].state is TrackState.TRACKED
    assert tracker.tracked_tracks[1].is_activated


def test_high_confirmation() -> None:
    tracker = ByteTracker()
    confirm_track(tracker)
    assert tracker.unconfirmed_confirmed == 1
    assert tracker.tracked_tracks[1].high_hits == 2
    assert tracker.tracked_tracks[1].low_hits == 0


def test_low_cannot_confirm_new() -> None:
    tracker = ByteTracker()
    tracker.update([det(100, 0.80)], 1)
    assert tracker.update([det(104, 0.40)], 2) == []
    assert tracker.update([det(108, 0.35)], 3) == []
    assert 1 in tracker.removed_tracks
    assert tracker.removed_tracks[1].state is TrackState.REMOVED
    assert not tracker.removed_tracks[1].is_activated
    assert tracker.unconfirmed_confirmed == 0
    assert tracker.unconfirmed_removed == 1
    assert tracker.low_matches == 0
    tracker.finalize()
    assert tracker.track_summaries() == []


def test_lost_and_reactivation_same_id() -> None:
    tracker = ByteTracker()
    confirm_track(tracker)
    assert tracker.update([], 3) == []
    assert tracker.lost_tracks[1].state is TrackState.LOST
    assert tracker.update([det(112, 0.88)], 4)[0]["track_id"] == 1
    assert tracker.tracked_tracks[1].state is TrackState.TRACKED
    assert tracker.reactivated_tracks == 1
    assert tracker.tracks_marked_lost == 1


def test_buffer_expiry_and_removed_cannot_reactivate() -> None:
    tracker = ByteTracker(track_buffer_frames_at_30fps=2)
    tracker.configure_frame_rate(30.0)
    confirm_track(tracker)
    tracker.update([], 3)
    tracker.update([], 4)
    output = tracker.update([det(104, 0.90)], 5)
    assert output[0]["track_id"] == 2
    assert tracker.removed_tracks[1].state is TrackState.REMOVED
    assert tracker.removed_after_buffer == 1
    assert 1 not in tracker.active_tracks


def test_fps_normalization() -> None:
    at_30 = ByteTracker(track_buffer_frames_at_30fps=10)
    at_60 = ByteTracker(track_buffer_frames_at_30fps=10)
    assert at_30.configure_frame_rate(30.0) == 10
    assert at_60.configure_frame_rate(60.0) == 20
    assert abs(at_30.effective_track_buffer / 30.0 - at_60.effective_track_buffer / 60.0) < 0.01


def test_low_recovery_for_confirmed_track() -> None:
    tracker = ByteTracker()
    outputs = [
        tracker.update([det(100, 0.90)], 1),
        tracker.update([det(104, 0.85)], 2),
        tracker.update([det(108, 0.40)], 3),
        tracker.update([det(112, 0.35)], 4),
        tracker.update([det(116, 0.88)], 5),
    ]
    assert [[item["track_id"] for item in output] for output in outputs] == [[1]] * 5
    assert outputs[2][0]["top_k_eligible"] is False
    assert outputs[3][0]["top_k_eligible"] is False
    track = tracker.tracked_tracks[1]
    assert (track.hits, track.high_hits, track.low_hits) == (5, 3, 2)


def test_reset_clears_all_state() -> None:
    tracker = ByteTracker(track_buffer_frames_at_30fps=10)
    tracker.configure_frame_rate(60.0)
    confirm_track(tracker)
    tracker.update([], 3)
    tracker.reset()
    assert not tracker.tracked_tracks
    assert not tracker.lost_tracks
    assert not tracker.unconfirmed_tracks
    assert not tracker.removed_tracks
    assert tracker.effective_track_buffer == 10
    assert tracker.created_tracks == tracker.reactivated_tracks == 0
    assert tracker.update([det(300, 0.9)], 1)[0]["track_id"] == 1


def test_finalize_preserves_object_and_stats() -> None:
    tracker = ByteTracker()
    confirm_track(tracker)
    tracker.update([det(108, 0.40)], 3)
    active_object = tracker.tracked_tracks[1]
    active_object_id = id(active_object)
    assert (active_object.hits, active_object.high_hits, active_object.low_hits) == (3, 2, 1)

    tracker.finalize()
    assert id(tracker.removed_tracks[1]) == active_object_id
    summary = tracker.track_summaries()[0]
    assert (summary["hits"], summary["high_hits"], summary["low_hits"]) == (3, 2, 1)

    tracker.reset()
    tracker.update([det(500, 0.9)], 1)
    assert tracker.unconfirmed_tracks[1].track_id == 1
    assert id(tracker.unconfirmed_tracks[1]) != active_object_id


def main() -> int:
    test_high_confirmation()
    test_low_cannot_confirm_new()
    test_lost_and_reactivation_same_id()
    test_buffer_expiry_and_removed_cannot_reactivate()
    test_fps_normalization()
    test_low_recovery_for_confirmed_track()
    test_reset_clears_all_state()
    test_finalize_preserves_object_and_stats()
    print("BYTE T3 lifecycle, low-confirmation guard, reactivation, and FPS tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
