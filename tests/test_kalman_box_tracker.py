"""Synthetic motion tests for SORT's seven-state Kalman box tracker."""

from __future__ import annotations

import numpy as np

from core.kalman_box_tracker import (
    KalmanBoxTracker,
    bbox_to_measurement,
    state_to_bbox,
)


def detection(box: list[int], conf: float = 0.9) -> dict:
    return {"box": box, "conf": conf}


def center_x(box: np.ndarray) -> float:
    return float((box[0] + box[2]) / 2.0)


def test_stationary() -> None:
    expected = np.asarray([100, 100, 200, 150], dtype=np.float64)
    tracker = KalmanBoxTracker(detection(expected.tolist()), 1, 1)
    for frame in range(2, 12):
        prediction = tracker.predict()
        assert prediction is not None
        assert tracker.update(detection(expected.tolist()), frame)
    state = tracker.get_state()
    assert state is not None
    assert np.max(np.abs(state - expected)) < 1.0


def test_motion_and_missing_measurements() -> None:
    tracker = KalmanBoxTracker(detection([100, 100, 200, 150]), 1, 1)
    for frame, x1 in ((2, 110), (3, 120)):
        assert tracker.predict() is not None
        assert tracker.update(detection([x1, 100, x1 + 100, 150]), frame)

    before = tracker.get_state()
    prediction_1 = tracker.predict()
    prediction_2 = tracker.predict()
    assert before is not None and prediction_1 is not None and prediction_2 is not None
    assert center_x(prediction_1) > center_x(before)
    assert center_x(prediction_2) > center_x(prediction_1)

    assert tracker.update(detection([150, 100, 250, 150]), 6)
    recovered = tracker.get_state()
    assert recovered is not None
    assert 140.0 < center_x(recovered) < 210.0


def test_invalid_boxes_are_safe() -> None:
    tracker = KalmanBoxTracker(detection([100, 100, 200, 150]), 1, 1)
    assert tracker.predict() is not None
    assert not tracker.update(detection([100, 100, 100, 150]), 2)
    assert not tracker.update(detection([0, 0, float("inf"), 10]), 2)
    assert bbox_to_measurement([0, 0, 0, 10]) is None
    assert bbox_to_measurement([0, 0, float("nan"), 10]) is None
    assert state_to_bbox(np.asarray([0, 0, -1, 1, 0, 0, 0])) is None


def main() -> int:
    test_stationary()
    test_motion_and_missing_measurements()
    test_invalid_boxes_are_safe()
    print("Kalman box tracker tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
