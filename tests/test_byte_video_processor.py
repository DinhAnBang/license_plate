"""VideoProcessor contract: low BYTE matches track but never enter Top-K/OCR."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from core.byte_tracker import ByteTracker
from core.quality import PlateQualityEvaluator
from core.video_processor import VideoProcessor


class FakeByteDetector:
    def __init__(self, confidences: list[float] | None = None) -> None:
        self.calls = 0
        self.confidences = confidences or [0.90, 0.85, 0.40, 0.35, 0.88]

    def detect_with_stats(self, frame: np.ndarray, *, conf_threshold: float | None = None):
        assert conf_threshold == 0.10
        confidence = self.confidences[self.calls]
        x1 = 20 + 5 * self.calls
        self.calls += 1
        detections = [{"conf": confidence, "box": [x1, 20, x1 + 80, 60]}]
        return detections, {
            "raw_detector_candidates": 10,
            "detections_after_threshold": 1,
            "detections_after_nms": 1,
        }


class FakeOCR:
    session_creation_count = 1

    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, crop: np.ndarray) -> dict:
        self.calls += 1
        return {"text": "59N304864", "confidence": 0.9}


def test_low_is_tracking_only() -> None:
    with tempfile.TemporaryDirectory(prefix="plate_byte_video_") as temp_dir:
        root = Path(temp_dir)
        source = root / "byte.mp4"
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 100))
        assert writer.isOpened()
        for _ in range(5):
            writer.write(np.full((100, 160, 3), 128, dtype=np.uint8))
        writer.release()

        detector = FakeByteDetector()
        ocr = FakeOCR()
        processor = VideoProcessor(
            detector,
            project_root=root,
            output_dir=root / "output",
            tracker_mode="byte",
            quality_evaluator=PlateQualityEvaluator(),
            ocr=ocr,
            top_k=3,
        )
        result = processor.process(source)
        assert isinstance(processor.tracker, ByteTracker)
        assert result["total_tracks"] == 1
        assert result["tracks"][0]["hits"] == 5
        assert result["quality_candidates"] == 3
        assert result["ocr_calls"] == ocr.calls == 3
        assert processor.tracker.low_score_recoveries == 2
        assert processor.tracker.high_detections == 3
        assert processor.tracker.low_detections == 2


def test_unconfirmed_track_is_not_final_or_ocr_event() -> None:
    with tempfile.TemporaryDirectory(prefix="plate_byte_unconfirmed_") as temp_dir:
        root = Path(temp_dir)
        source = root / "one_high_then_low.mp4"
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (160, 100))
        assert writer.isOpened()
        for _ in range(3):
            writer.write(np.full((100, 160, 3), 128, dtype=np.uint8))
        writer.release()

        detector = FakeByteDetector([0.80, 0.40, 0.35])
        ocr = FakeOCR()
        processor = VideoProcessor(
            detector,
            project_root=root,
            output_dir=root / "output",
            tracker_mode="byte",
            quality_evaluator=PlateQualityEvaluator(),
            ocr=ocr,
            top_k=3,
        )
        result = processor.process(source)
        assert result["total_tracks"] == 0
        assert result["best_crops"] == []
        assert result["ocr_calls"] == ocr.calls == 0
        assert processor.tracker.unconfirmed_removed == 1
        assert processor.tracker.low_matches == 0


def main() -> int:
    test_low_is_tracking_only()
    test_unconfirmed_track_is_not_final_or_ocr_event()
    print("BYTE low-score tracking-only VideoProcessor contract: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
