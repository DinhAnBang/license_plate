"""Synthetic tests for Phase 5 quality metrics."""

from __future__ import annotations

import cv2
import numpy as np

from core.quality import PlateQualityEvaluator


def main() -> int:
    evaluator = PlateQualityEvaluator(reference_area=12_000.0)

    sharp = np.zeros((80, 240, 3), dtype=np.uint8)
    cv2.rectangle(sharp, (10, 10), (230, 70), (255, 255, 255), 2)
    cv2.putText(
        sharp,
        "59A-123.45",
        (25, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    blurred = cv2.GaussianBlur(sharp, (9, 9), 0)
    sharp_metrics = evaluator.evaluate(sharp, detection_confidence=0.9)
    blurred_metrics = evaluator.evaluate(blurred, detection_confidence=0.9)
    assert sharp_metrics["sharpness"] > blurred_metrics["sharpness"]
    assert sharp_metrics["sharpness_raw"] > blurred_metrics["sharpness_raw"]

    normal = np.full((80, 120, 3), 128, dtype=np.uint8)
    dark = np.full((80, 120, 3), 20, dtype=np.uint8)
    bright = np.full((80, 120, 3), 235, dtype=np.uint8)
    normal_score = evaluator.evaluate(normal, 0.9)["brightness"]
    dark_score = evaluator.evaluate(dark, 0.9)["brightness"]
    bright_score = evaluator.evaluate(bright, 0.9)["brightness"]
    assert normal_score > dark_score
    assert normal_score > bright_score

    large = np.full((80, 240, 3), 128, dtype=np.uint8)
    small = np.full((20, 60, 3), 128, dtype=np.uint8)
    assert evaluator.evaluate(large, 0.9)["size"] >= evaluator.evaluate(small, 0.9)["size"]

    try:
        evaluator.evaluate(np.empty((0, 0, 3), dtype=np.uint8), 0.9)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid crop should raise ValueError")

    print("Quality evaluator tests: OK")
    print(f"Sharp raw: {sharp_metrics['sharpness_raw']:.2f}")
    print(f"Blurred raw: {blurred_metrics['sharpness_raw']:.2f}")
    print(f"Brightness scores: normal={normal_score:.3f}, dark={dark_score:.3f}, bright={bright_score:.3f}")
    print(f"Size scores: large={evaluator.evaluate(large, 0.9)['size']:.3f}, small={evaluator.evaluate(small, 0.9)['size']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
