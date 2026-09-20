"""Run the Phase T1 legacy-versus-SORT benchmark on the two reference videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from core.detector import PlateDetector
from core.config import (
    BENCHMARK_DETECTOR_CONF_THRESHOLD,
    BENCHMARK_LEGACY_IOU_THRESHOLD,
    DETECTOR_NMS_IOU_THRESHOLD,
    QUALITY_BRIGHTNESS_TARGET,
    QUALITY_BRIGHTNESS_WEIGHT,
    QUALITY_CONFIDENCE_WEIGHT,
    QUALITY_REFERENCE_AREA,
    QUALITY_SHARPNESS_REFERENCE,
    QUALITY_SHARPNESS_WEIGHT,
    QUALITY_SIZE_WEIGHT,
    RECOGNITION_TOP_K,
    SORT_IOU_THRESHOLD,
    SORT_MAX_AGE,
    SORT_MIN_HITS,
)
from core.ocr import MicroCharNetOCR
from core.quality import PlateQualityEvaluator
from core.sort_tracker import SortTracker
from core.tracker import PlateTracker
from core.video_processor import VideoProcessor


ROOT = Path(__file__).resolve().parents[1]


def quality_evaluator() -> PlateQualityEvaluator:
    return PlateQualityEvaluator(
        confidence_weight=QUALITY_CONFIDENCE_WEIGHT,
        sharpness_weight=QUALITY_SHARPNESS_WEIGHT,
        brightness_weight=QUALITY_BRIGHTNESS_WEIGHT,
        size_weight=QUALITY_SIZE_WEIGHT,
        sharpness_reference=QUALITY_SHARPNESS_REFERENCE,
        brightness_target=QUALITY_BRIGHTNESS_TARGET,
        reference_area=QUALITY_REFERENCE_AREA,
    )


def summarize(result: dict[str, Any], tracker: PlateTracker | SortTracker) -> dict[str, Any]:
    hits = [int(track["hits"]) for track in result["tracks"]]
    summary: dict[str, Any] = {
        "total_detections": int(result["total_detections"]),
        "created_tracks": len(result["tracks"]),
        "final_tracks": int(result["total_tracks"]),
        "average_hits_per_track": sum(hits) / len(hits) if hits else 0.0,
        "max_hits": max(hits, default=0),
        "processing_time_seconds": float(result["total_processing_time"]),
        "average_tracking_ms": float(result["avg_tracking_ms"]),
        "ocr_calls": int(result["ocr_calls"]),
        "tracks": result["tracks"],
        "plates": [
            {
                "track_id": item["track_id"],
                "first_frame": item["first_frame"],
                "last_frame": item["last_frame"],
                "hits": item["hits"],
                "best_frame": item["best_frame"],
                "text": item["text"],
                "raw_text": item["raw_text"],
                "ocr_conf": item["ocr_conf"],
                "crop": item["crop"],
            }
            for item in result["best_crops"]
        ],
    }
    if isinstance(tracker, SortTracker):
        summary["created_tracks"] = tracker.created_tracks
        summary.update(tracker.statistics())
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--videos",
        nargs="+",
        type=Path,
        default=[ROOT / "input/test1_60fps.mp4", ROOT / "input/test2_30fps.mp4"],
    )
    parser.add_argument("--output", type=Path, default=ROOT / "output/t1_benchmark")
    args = parser.parse_args()

    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    detector = PlateDetector(
        ROOT / "models/best.onnx",
        conf_threshold=BENCHMARK_DETECTOR_CONF_THRESHOLD,
        iou_threshold=DETECTOR_NMS_IOU_THRESHOLD,
        providers=["CPUExecutionProvider"],
    )
    ocr = MicroCharNetOCR(ROOT / "models/OCR/microcharnet.onnx")

    detector_probe = np.zeros(
        [dimension if isinstance(dimension, int) and dimension > 0 else 1 for dimension in detector.input_shape],
        dtype=np.float32,
    )
    detector.session.run(
        [item["name"] for item in detector.output_info],
        {detector.input_name: detector_probe},
    )
    ocr.session.run(
        [ocr.output_name],
        {ocr.input_name: np.zeros((1, 3, ocr.input_height, ocr.input_width), dtype=np.float32)},
    )

    report: dict[str, Any] = {
        "configuration": {
            "detector_conf_threshold": detector.conf_threshold,
            "detector_nms_iou_threshold": detector.iou_threshold,
            "legacy_iou_threshold": BENCHMARK_LEGACY_IOU_THRESHOLD,
            "legacy_max_missed": SORT_MAX_AGE,
            "sort_iou_threshold": SORT_IOU_THRESHOLD,
            "sort_max_age": SORT_MAX_AGE,
            "sort_min_hits": SORT_MIN_HITS,
            "top_k": RECOGNITION_TOP_K,
            # One detector object owns the single session used by all runs.
            "detector_sessions": 1,
            "ocr_sessions": ocr.session_creation_count,
        },
        "videos": {},
    }

    for video_value in args.videos:
        video = video_value if video_value.is_absolute() else (ROOT / video_value)
        video = video.resolve()
        if not video.is_file():
            raise FileNotFoundError(video)
        video_report: dict[str, Any] = {}
        for mode in ("legacy", "sort"):
            tracker: PlateTracker | SortTracker
            if mode == "legacy":
                tracker = PlateTracker(
                    iou_threshold=BENCHMARK_LEGACY_IOU_THRESHOLD,
                    max_missed=SORT_MAX_AGE,
                )
            else:
                tracker = SortTracker(
                    iou_threshold=SORT_IOU_THRESHOLD,
                    max_age=SORT_MAX_AGE,
                    min_hits=SORT_MIN_HITS,
                )
            processor = VideoProcessor(
                detector,
                output_dir=output_root / video.stem / mode,
                project_root=ROOT,
                tracker=tracker,
                tracker_mode=mode,
                quality_evaluator=quality_evaluator(),
                ocr=ocr,
                top_k=RECOGNITION_TOP_K,
                write_json=False,
                stitching_enabled=False,
            )
            detector.conf_threshold = BENCHMARK_DETECTOR_CONF_THRESHOLD
            result = processor.process(video)
            video_report[mode] = summarize(result, tracker)
            print(
                f"{video.name} {mode}: detections={result['total_detections']} "
                f"tracks={result['total_tracks']} ocr={result['ocr_calls']} "
                f"time={result['total_processing_time']:.2f}s"
            )
        report["videos"][video.name] = video_report

    report_path = output_root / "benchmark_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Benchmark report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
