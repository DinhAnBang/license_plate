"""Synthetic tests for the official Phase 6 video JSON contract."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from core.result_writer import VideoResultWriter


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="plate_json_test_") as temp_dir:
        project_root = Path(temp_dir)
        crops_dir = project_root / "output" / "crops"
        crops_dir.mkdir(parents=True)
        crop_a = crops_dir / "test_track_0003.jpg"
        crop_b = crops_dir / "test_track_0007.jpg"
        cv2.imwrite(str(crop_a), np.full((20, 50, 3), 128, dtype=np.uint8))
        cv2.imwrite(str(crop_b), np.full((30, 70, 3), 128, dtype=np.uint8))

        writer = VideoResultWriter(
            output_dir=project_root / "output" / "json",
            project_root=project_root,
        )
        unsorted_plates = [
            {
                "track_id": np.int64(7),
                "first_frame": np.int64(20),
                "last_frame": np.int64(30),
                "hits": np.int64(5),
                "best_frame": np.int64(25),
                "conf": np.float32(0.88),
                "quality": np.float32(0.72),
                "box": [np.int64(10), np.int64(20), np.int64(80), np.int64(50)],
                "crop": "output/crops/test_track_0007.jpg",
                "text": "",
                "ocr_conf": np.float32(0.0),
            },
            {
                "track_id": np.int64(3),
                "first_frame": np.int64(2),
                "last_frame": np.int64(12),
                "hits": np.int64(4),
                "best_frame": np.int64(7),
                "conf": np.float32(0.91),
                "quality": np.float32(0.81),
                "box": [np.int64(1), np.int64(2), np.int64(60), np.int64(40)],
                "crop": "output/crops/test_track_0003.jpg",
                "text": "",
                "ocr_conf": np.float32(0.0),
            },
        ]

        json_path = writer.write(
            source_path="input/test.mp4",
            width=720,
            height=1280,
            fps=np.float32(30.0),
            frames=np.int64(228),
            plates=unsorted_plates,
        )
        saved_path = project_root / json_path
        data = json.loads(saved_path.read_text(encoding="utf-8"))
        assert data["count"] == len(data["plates"]) == 2
        assert [plate["first_detected_frame"] for plate in data["plates"]] == [2, 20]
        assert data["plates"][0]["best_frame_time_sec"] == 0.2
        assert all(type(plate["detection_confidence"]) is float for plate in data["plates"])
        assert all(plate["plate_text"] == "" and plate["ocr_confidence"] == 0.0 for plate in data["plates"])
        assert all(Path(plate["crop_path"]).is_absolute() for plate in data["plates"])
        assert set(data) == {
            "status", "request_id", "input_type", "output_video", "processing", "count", "plates"
        }

        empty = writer.build_result(
            source_path="input/no_plate.mp4",
            width=1280,
            height=720,
            fps=30.0,
            frames=300,
            plates=[],
        )
        assert empty["count"] == 0
        assert empty["plates"] == []
        assert empty["processing"] == {"total_ms": 0.0, "frames": 300, "average_frame_ms": 0.0}

    print("Video JSON contract tests: OK")
    print("Normal result: count=2, sorted by first frame")
    print("No-detection result: count=0, plates=[]")
    print("NumPy values converted to standard JSON numbers")
    print("1-based time calculation: best_frame 7 at 30 FPS = 0.2 s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
