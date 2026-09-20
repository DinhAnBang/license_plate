"""R1 production DTO field, path, time, and best-observation contracts."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from core.result_serializer import build_image_result, build_video_result


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="plate_r1_serializer_") as temp_dir:
        root = Path(temp_dir)
        crop = root / "crops" / "best.jpg"
        crop.parent.mkdir(parents=True)
        assert cv2.imwrite(str(crop), np.full((20, 60, 3), 128, dtype=np.uint8))
        annotated_image = root / "requests" / "img" / "annotated.jpg"
        annotated_video = root / "requests" / "vid" / "annotated.mp4"

        image = build_image_result(
            request_id="img_001",
            output_image=annotated_image,
            processing_total_ms=285.42,
            plates=[
                {
                    "text": "72A16231",
                    "conf": 0.84108,
                    "ocr_conf": 0.975158,
                    "crop": crop,
                }
            ],
            project_root=root,
        )
        assert set(image) == {
            "status", "request_id", "input_type", "output_image", "processing", "count", "plates"
        }
        assert set(image["plates"][0]) == {
            "plate_text", "detection_confidence", "ocr_confidence", "crop_path"
        }
        assert image["status"] == "success"
        assert image["count"] == len(image["plates"]) == 1
        assert Path(image["output_image"]).is_absolute()
        assert Path(image["plates"][0]["crop_path"]).is_absolute()

        video = build_video_result(
            request_id="video_001",
            output_video=annotated_video,
            fps=30.0,
            frames=120,
            processing_total_ms=1200.0,
            average_frame_ms=999.0,
            plates=[
                {
                    "text": "59N304864",
                    # This is the confidence of the selected best-frame
                    # observation, not the maximum track confidence.
                    "conf": 0.82,
                    "ocr_conf": 0.9681,
                    "first_frame": 100,
                    "last_frame": 104,
                    "best_frame": 101,
                    "crop": crop,
                }
            ],
            project_root=root,
        )
        plate = video["plates"][0]
        assert set(video) == {
            "status", "request_id", "input_type", "output_video", "processing", "count", "plates"
        }
        assert set(video["processing"]) == {"total_ms", "frames", "average_frame_ms"}
        assert set(plate) == {
            "plate_text", "detection_confidence", "ocr_confidence",
            "first_detected_frame", "last_detected_frame",
            "first_detected_time_sec", "last_detected_time_sec",
            "best_frame_index", "best_frame_time_sec", "crop_path",
        }
        assert video["count"] == len(video["plates"]) == 1
        assert plate["detection_confidence"] == 0.82
        assert plate["first_detected_frame"] == 100
        assert plate["last_detected_frame"] == 104
        assert plate["best_frame_index"] == 101
        assert plate["first_detected_time_sec"] == round(99 / 30, 6)
        assert plate["last_detected_time_sec"] == round(103 / 30, 6)
        assert plate["best_frame_time_sec"] == round(100 / 30, 6)
        assert video["processing"]["average_frame_ms"] == 10.0
        assert Path(video["output_video"]).is_absolute()

    print("R1 production result serializer contract: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
