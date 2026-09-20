"""T5 video filter and unchanged public JSON contract integration test."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from core.quality import PlateQualityEvaluator
from core.result_writer import VideoResultWriter
from core.sort_tracker import SortTracker
from core.video_processor import VideoProcessor


class ValidAndAdDetector:
    def detect(self, frame: np.ndarray) -> list[dict]:
        del frame
        return [
            {"conf": 0.9, "box": [10, 20, 90, 60]},
            {"conf": 0.9, "box": [130, 20, 159, 60]},
        ]


class ValidThenAdOCR:
    session_creation_count = 1

    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, crop: np.ndarray) -> dict:
        del crop
        self.calls += 1
        return (
            {"text": "30-F 051.48", "confidence": 0.9}
            if self.calls % 2 == 1
            else {"text": "CON PHONG", "confidence": 0.99}
        )


def make_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 100))
    assert writer.isOpened()
    for value in (80, 100, 120):
        writer.write(np.full((100, 160, 3), value, dtype=np.uint8))
    writer.release()


def test_t5_filters_invalid_event_without_public_schema_changes() -> None:
    with tempfile.TemporaryDirectory(prefix="plate_t5_video_") as temp_dir:
        root = Path(temp_dir)
        source = root / "t5.mp4"
        make_video(source)
        ocr = ValidThenAdOCR()
        processor = VideoProcessor(
            ValidAndAdDetector(),
            project_root=root,
            output_dir=root / "output",
            tracker=SortTracker(iou_threshold=0.25, max_age=0, min_hits=1),
            tracker_mode="sort",
            quality_evaluator=PlateQualityEvaluator(),
            result_writer=VideoResultWriter(project_root=root),
            ocr=ocr,
            top_k=1,
            t5_enabled=True,
        )
        result = processor.process(source)
        official = result["official_result"]
        assert official["count"] == len(official["plates"]) == 1
        assert official["plates"][0]["plate_text"] == "30F05148"
        assert set(official) == {
            "status", "request_id", "input_type", "output_video", "processing", "count", "plates"
        }
        assert set(official["plates"][0]) == {
            "plate_text", "detection_confidence", "ocr_confidence",
            "first_detected_frame", "last_detected_frame",
            "first_detected_time_sec", "last_detected_time_sec",
            "best_frame_index", "best_frame_time_sec", "crop_path",
        }
        assert result["stitching"]["metrics"]["t5_invalid_count"] == 1
        assert len(result["stitching"]["t5"]["invalid_results"]) == 1
        saved = json.loads((root / "output/json/t5.json").read_text(encoding="utf-8"))
        assert saved == official


if __name__ == "__main__":
    test_t5_filters_invalid_event_without_public_schema_changes()
    print("T5 video integration: OK")
