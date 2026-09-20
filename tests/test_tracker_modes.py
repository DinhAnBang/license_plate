"""VideoProcessor integration tests for legacy, SORT, and source defaults."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from core.quality import PlateQualityEvaluator
from core.sort_tracker import SortTracker
from core.tracker import PlateTracker
from core.video_processor import VideoProcessor


class FakeDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame: np.ndarray) -> list[dict]:
        self.calls += 1
        x1 = 20 + self.calls
        return [{"conf": 0.9, "box": [x1, 20, x1 + 80, 60]}]


class FakeOCR:
    session_creation_count = 1

    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, crop: np.ndarray) -> dict:
        self.calls += 1
        return {"text": "61B161519", "confidence": 0.9}


def _video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 100))
    assert writer.isOpened()
    for _ in range(4):
        writer.write(np.full((100, 160, 3), 128, dtype=np.uint8))
    writer.release()


def test_modes() -> None:
    with tempfile.TemporaryDirectory(prefix="plate_tracker_modes_") as temp_dir:
        root = Path(temp_dir)
        source = root / "video.mp4"
        _video(source)

        default_processor = VideoProcessor(FakeDetector(), project_root=root)
        assert default_processor.tracker_mode == "sort"
        assert isinstance(default_processor.tracker, SortTracker)

        legacy_detector = FakeDetector()
        legacy_ocr = FakeOCR()
        legacy = VideoProcessor(
            legacy_detector,
            project_root=root,
            output_dir=root / "legacy",
            tracker=PlateTracker(iou_threshold=0.25, max_missed=10),
            tracker_mode="legacy",
            quality_evaluator=PlateQualityEvaluator(),
            ocr=legacy_ocr,
        )
        legacy_result = legacy.process(source)
        assert isinstance(legacy.tracker, PlateTracker)
        assert legacy_result["total_tracks"] == 1
        assert legacy_detector.calls == 4 and legacy_ocr.calls == 3

        sort_detector = FakeDetector()
        sort_ocr = FakeOCR()
        sort = VideoProcessor(
            sort_detector,
            project_root=root,
            output_dir=root / "sort",
            tracker_mode="sort",
            quality_evaluator=PlateQualityEvaluator(),
            ocr=sort_ocr,
        )
        sort_result = sort.process(source)
        assert isinstance(sort.tracker, SortTracker)
        assert sort_result["total_tracks"] == 1
        assert sort_detector.calls == 4 and sort_ocr.calls == 3
        assert legacy_ocr.session_creation_count == sort_ocr.session_creation_count == 1

        sort.process(source)
        assert sort_result["tracks"][0]["track_id"] == 1
        assert sort.tracker.track_summaries()[0]["track_id"] == 1


def main() -> int:
    test_modes()
    print("VideoProcessor legacy/SORT mode integration: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
