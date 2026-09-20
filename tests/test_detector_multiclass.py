"""Regression test for one-class and multi-class YOLO detection outputs."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from core.detector import PlateDetector, _LetterboxInfo


def run_case(class_scores: list[float]) -> dict:
    detector = PlateDetector.__new__(PlateDetector)
    detector.conf_threshold = 0.5
    detector.iou_threshold = 0.45
    detector._outputs = [SimpleNamespace(name="output0")]

    features = 4 + len(class_scores)
    output = np.zeros((1, features, 10), dtype=np.float32)
    output[0, :4, 0] = [50.0, 40.0, 20.0, 10.0]
    output[0, 4:, 0] = class_scores
    detections = detector._postprocess(
        [output],
        image_shape=(100, 100, 3),
        letterbox=_LetterboxInfo(scale=1.0, pad_left=0, pad_top=0),
    )
    assert len(detections) == 1
    return detections[0]


def main() -> int:
    one_class = run_case([0.8])
    two_class = run_case([0.001, 0.9])
    assert abs(one_class["conf"] - 0.8) < 1e-6
    assert abs(two_class["conf"] - 0.9) < 1e-6
    assert one_class["box"] == two_class["box"] == [40, 35, 60, 45]
    print("One-class and two-class detector output decoding: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
