"""Synthetic T4 VideoProcessor integration and public JSON regression."""

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
from engine.output_manager import RequestOutputManager


class FragmentedDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame: np.ndarray) -> list[dict]:
        del frame
        self.calls += 1
        if self.calls in {1, 2}:
            return [{"conf": 0.80, "box": [20, 20, 100, 60]}]
        if self.calls in {4, 5}:
            return [{"conf": 0.90, "box": [25, 20, 105, 60]}]
        return []


class SamePlateOCR:
    session_creation_count = 1

    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, crop: np.ndarray) -> dict:
        assert crop.size > 0
        self.calls += 1
        return {"text": "59-N3 048.64", "confidence": 0.92}


def make_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 100)
    )
    assert writer.isOpened()
    for value in (80, 100, 120, 160, 200):
        writer.write(np.full((100, 160, 3), value, dtype=np.uint8))
    writer.release()


def run(root: Path, source: Path, request_id: str, enabled: bool) -> dict:
    output_manager = RequestOutputManager(root)
    paths = output_manager.paths_for(request_id, "video")
    output_manager.prepare(paths)
    processor = VideoProcessor(
        FragmentedDetector(),
        project_root=root,
        output_dir=root / "output",
        tracker=SortTracker(iou_threshold=0.25, max_age=0, min_hits=1),
        tracker_mode="sort",
        quality_evaluator=PlateQualityEvaluator(),
        result_writer=VideoResultWriter(project_root=root),
        ocr=SamePlateOCR(),
        top_k=2,
        stitching_enabled=enabled,
    )
    return processor.process(source, output_paths=paths)


def test_video_tracklets_become_one_event_and_one_crop() -> None:
    with tempfile.TemporaryDirectory(prefix="plate_t4_integration_") as temp_dir:
        root = Path(temp_dir)
        source = root / "fragmented.mp4"
        make_video(source)

        before = run(root, source, "before", False)
        assert before["total_tracks"] == 2
        assert len(before["best_crops"]) == 2
        assert before["stitching"]["metrics"]["number_of_merges"] == 0
        assert len(list((root / "output/requests/before/crops").glob("*.jpg"))) == 2

        after = run(root, source, "after", True)
        official = after["official_result"]
        assert after["total_tracks"] == 2
        assert len(after["best_crops"]) == 1
        assert after["best_crops"][0]["member_track_ids"] == [1, 2]
        assert after["stitching"]["metrics"]["number_of_merges"] == 1
        assert after["stitching"]["metrics"]["exact_ocr_merges"] == 1
        assert after["stitching"]["metrics"]["final_crop_count"] == 1
        crops = list((root / "output/requests/after/crops").glob("*.jpg"))
        assert len(crops) == 1

        assert official["count"] == len(official["plates"]) == 1
        assert official["plates"][0]["plate_text"] == "59N304864"
        assert official["plates"][0]["first_detected_frame"] == 1
        assert official["plates"][0]["last_detected_frame"] == 5
        assert set(official) == {
            "status",
            "request_id",
            "input_type",
            "output_video",
            "processing",
            "count",
            "plates",
        }
        assert set(official["plates"][0]) == {
            "plate_text",
            "detection_confidence",
            "ocr_confidence",
            "first_detected_frame",
            "last_detected_frame",
            "first_detected_time_sec",
            "last_detected_time_sec",
            "best_frame_index",
            "best_frame_time_sec",
            "crop_path",
        }
        saved = json.loads(
            (root / "output/requests/after/result.json").read_text(encoding="utf-8")
        )
        assert saved == official
        assert "member_track_ids" not in json.dumps(official)


def main() -> int:
    test_video_tracklets_become_one_event_and_one_crop()
    print("T4 synthetic VideoProcessor integration: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
