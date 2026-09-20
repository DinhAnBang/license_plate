"""Phase T2 benchmark for legacy, SORT, and BYTE source modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from core.byte_tracker import ByteTracker
from core.config import (
    BENCHMARK_DETECTOR_CONF_THRESHOLD,
    DETECTOR_NMS_IOU_THRESHOLD,
    BYTE_HIGH_MATCH_IOU_THRESHOLD,
    BYTE_HIGH_THRESHOLD,
    BYTE_LOW_MATCH_IOU_THRESHOLD,
    BYTE_LOW_THRESHOLD,
    BYTE_TRACK_BUFFER_FRAMES_AT_30FPS,
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
from core.detector import PlateDetector
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


def summarize(
    result: dict[str, Any], tracker: PlateTracker | SortTracker | ByteTracker, mode: str
) -> dict[str, Any]:
    hits = [int(track["hits"]) for track in result["tracks"]]
    high_count = int(result["total_detections"])
    summary: dict[str, Any] = {
        "total_frames": int(result["processed_frames"]),
        "processing_time_seconds": float(result["total_processing_time"]),
        "average_tracking_ms": float(result["avg_tracking_ms"]),
        "high_score_detections": high_count,
        "created_tracks": len(result["tracks"]),
        "final_tracks": int(result["total_tracks"]),
        "average_hits_per_track": sum(hits) / len(hits) if hits else 0.0,
        "max_hits": max(hits, default=0),
        "ocr_calls": int(result["ocr_calls"]),
        "short_high_tracks_le_2_hits": sum(value <= 2 for value in hits),
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
    if isinstance(tracker, ByteTracker):
        summary["high_score_detections"] = tracker.high_detections
        summary["low_score_detections"] = tracker.low_detections
        summary["unmatched_low_dropped"] = tracker.unmatched_low_detections
    elif mode != "byte":
        summary["low_score_detections"] = 0
    return summary


def build_trace(
    video_name: str,
    summary: dict[str, Any],
    tracker: ByteTracker,
) -> dict[str, Any]:
    target_plates = [
        plate
        for plate in summary["plates"]
        if "59N3" in plate["text"] or "59N3" in plate["raw_text"]
    ]
    target_ids = {int(plate["track_id"]) for plate in target_plates}
    overlapping_ids = {
        int(track["track_id"])
        for track in summary["tracks"]
        if int(track["last_frame"]) >= 900 and int(track["first_frame"]) <= 1000
    }
    if not target_ids:
        target_ids = overlapping_ids
    rows = [
        row
        for row in tracker.trace_records
        if 880 <= int(row["frame_index"]) <= 1010
        and int(row["track_id"]) in (target_ids | overlapping_ids)
    ]
    return {
        "video": video_name,
        "target": "59-N3 048.64",
        "target_ids_from_ocr": sorted(target_ids),
        "all_track_ids_overlapping_frames_900_1000": sorted(overlapping_ids),
        "fields": {
            "classification": ["HIGH", "LOW", "NONE"],
            "association_result": [
                "HIGH_MATCH",
                "LOW_RECOVERY",
                "UNMATCHED",
                "NO_DETECTION",
                "NEW_TRACK",
            ],
        },
        "trace": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--videos",
        nargs="+",
        type=Path,
        default=[ROOT / "input/test1_60fps.mp4", ROOT / "input/test2_30fps.mp4"],
    )
    parser.add_argument("--output", type=Path, default=ROOT / "output/t2_benchmark")
    args = parser.parse_args()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "traces").mkdir(parents=True, exist_ok=True)

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
            "track_high_thresh": BYTE_HIGH_THRESHOLD,
            "track_low_thresh": BYTE_LOW_THRESHOLD,
            "detector_nms_iou_threshold": DETECTOR_NMS_IOU_THRESHOLD,
            "sort_iou_threshold": SORT_IOU_THRESHOLD,
            "byte_high_match_iou_threshold": BYTE_HIGH_MATCH_IOU_THRESHOLD,
            "byte_low_match_iou_threshold": BYTE_LOW_MATCH_IOU_THRESHOLD,
            "max_age": BYTE_TRACK_BUFFER_FRAMES_AT_30FPS,
            "min_hits": SORT_MIN_HITS,
            "top_k": RECOGNITION_TOP_K,
            "detector_sessions": 1,
            "ocr_sessions": ocr.session_creation_count,
        },
        "videos": {},
    }

    for video_value in args.videos:
        video = (video_value if video_value.is_absolute() else ROOT / video_value).resolve()
        if not video.is_file():
            raise FileNotFoundError(video)
        video_report: dict[str, Any] = {}
        byte_tracker: ByteTracker | None = None
        for mode in ("legacy", "sort", "byte"):
            if mode == "legacy":
                tracker: PlateTracker | SortTracker | ByteTracker = PlateTracker(
                    iou_threshold=SORT_IOU_THRESHOLD, max_missed=SORT_MAX_AGE
                )
            elif mode == "sort":
                tracker = SortTracker(
                    iou_threshold=SORT_IOU_THRESHOLD,
                    max_age=SORT_MAX_AGE,
                    min_hits=SORT_MIN_HITS,
                )
            else:
                tracker = ByteTracker(
                    track_high_thresh=BYTE_HIGH_THRESHOLD,
                    track_low_thresh=BYTE_LOW_THRESHOLD,
                    high_match_iou_threshold=BYTE_HIGH_MATCH_IOU_THRESHOLD,
                    low_match_iou_threshold=BYTE_LOW_MATCH_IOU_THRESHOLD,
                    max_age=BYTE_TRACK_BUFFER_FRAMES_AT_30FPS,
                    min_hits=SORT_MIN_HITS,
                    trace_enabled=video.name == "test2_30fps.mp4",
                )
                byte_tracker = tracker
            detector.conf_threshold = BENCHMARK_DETECTOR_CONF_THRESHOLD
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
            result = processor.process(video)
            summary = summarize(result, tracker, mode)
            video_report[mode] = summary
            print(
                f"{video.name} {mode}: high={summary['high_score_detections']} "
                f"low={summary['low_score_detections']} tracks={summary['final_tracks']} "
                f"ocr={summary['ocr_calls']} time={summary['processing_time_seconds']:.2f}s"
            )

        high_counts = {
            mode: int(video_report[mode]["high_score_detections"])
            for mode in ("legacy", "sort", "byte")
        }
        if len(set(high_counts.values())) != 1:
            raise RuntimeError(f"High-score detection regression for {video.name}: {high_counts}")
        report["videos"][video.name] = video_report

        if video.name == "test2_30fps.mp4" and byte_tracker is not None:
            trace = build_trace(video.name, video_report["byte"], byte_tracker)
            trace_path = output_root / "traces/plate_59N304864.json"
            trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = output_root / "benchmark_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Benchmark report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
