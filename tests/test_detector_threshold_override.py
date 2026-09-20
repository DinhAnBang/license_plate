"""Detector threshold override and shared-NMS regression tests for BYTE mode."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from core.detector import PlateDetector, _LetterboxInfo


def main() -> int:
    detector = PlateDetector.__new__(PlateDetector)
    detector.conf_threshold = 0.7
    detector.iou_threshold = 0.45
    detector._outputs = [SimpleNamespace(name="output0")]

    output = np.zeros((1, 5, 10), dtype=np.float32)
    output[0, :4, 0] = [50, 50, 20, 10]
    output[0, 4, 0] = 0.80
    output[0, :4, 1] = [50, 50, 20, 10]
    output[0, 4, 1] = 0.60  # Must lose to the overlapping high box.
    output[0, :4, 2] = [80, 80, 10, 10]
    output[0, 4, 2] = 0.20
    info = _LetterboxInfo(scale=1.0, pad_left=0, pad_top=0)

    default = detector._postprocess([output], (100, 100, 3), info)
    low, stats = detector._postprocess_with_stats(
        [output], (100, 100, 3), info, conf_threshold=0.10
    )
    assert len(default) == 1 and abs(default[0]["conf"] - 0.80) < 1e-6
    assert len(low) == 2
    assert any(item["box"] == default[0]["box"] and item["conf"] == default[0]["conf"] for item in low)
    assert stats == {
        "raw_detector_candidates": 10,
        "detections_after_threshold": 3,
        "detections_after_nms": 2,
    }
    assert detector.conf_threshold == 0.7
    print("Detector per-call threshold and high-priority shared NMS: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
