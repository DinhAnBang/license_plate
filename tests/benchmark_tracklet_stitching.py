"""T4 before/after benchmark for post-tracking plate-event stitching."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from core.byte_tracker import ByteTracker
from core.config import (
    BENCHMARK_DETECTOR_CONF_THRESHOLD,
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
    STITCHING_ENABLED,
    STITCH_MAX_CENTER_DISTANCE_RATIO,
    STITCH_MAX_EDIT_DISTANCE,
    STITCH_MAX_GAP_SEC,
    STITCH_MIN_FUZZY_TEXT_LENGTH,
)
from core.detector import PlateDetector
from core.ocr import MicroCharNetOCR
from core.quality import PlateQualityEvaluator
from core.result_writer import VideoResultWriter
from core.sort_tracker import SortTracker
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


def make_tracker(mode: str) -> SortTracker | ByteTracker:
    if mode == "sort":
        return SortTracker(
            iou_threshold=SORT_IOU_THRESHOLD,
            max_age=SORT_MAX_AGE,
            min_hits=SORT_MIN_HITS,
        )
    if mode == "byte":
        return ByteTracker()
    raise ValueError(mode)


def duplicate_texts(best_crops: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Return OCR texts that still occur in more than one final event."""

    grouped: dict[str, list[int]] = {}
    for item in best_crops:
        text = str(item["text"])
        if text:
            grouped.setdefault(text, []).append(int(item["event_id"]))
    return {
        text: event_ids
        for text, event_ids in grouped.items()
        if len(event_ids) > 1
    }


def summarize(result: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    stitching = result["stitching"]
    metrics = dict(stitching["metrics"])
    crop_dir = output_dir / "crops"
    final_crops = sorted(str(path) for path in crop_dir.glob("*.jpg"))
    reason_counts = Counter(
        str(item["reason"])
        for item in stitching["decisions"]
        if "reason" in item
    )
    official = result["official_result"]
    return {
        **metrics,
        "final_crop_count": len(final_crops),
        "ocr_calls": int(result["ocr_calls"]),
        "processing_time_seconds": float(result["total_processing_time"]),
        "tracking_ms_per_frame": float(result["avg_tracking_ms"]),
        "public_count": int(official["count"]),
        "public_schema": {
            "top_level": list(official),
            "plate": list(official["plates"][0]) if official["plates"] else [],
        },
        "events": stitching["events"],
        "decisions": stitching["decisions"],
        "decision_reason_counts": dict(reason_counts),
        "remaining_exact_text_duplicates": duplicate_texts(result["best_crops"]),
        "fuzzy_merges": [
            item
            for item in stitching["decisions"]
            if item.get("reason") == "MERGED_FUZZY_OCR_DISTANCE_1"
        ],
        "suspected_wrong_merges": [
            item
            for item in stitching["decisions"]
            if item.get("reason") == "MERGED_FUZZY_OCR_DISTANCE_1"
        ],
        "con_phong_events": [
            item
            for item in stitching["events"]
            if "CONPHONG" in str(item["canonical_plate_text"])
        ],
        "final_crops": final_crops,
        "production_json_example": official,
    }


def target_trace(after: dict[str, Any], text: str) -> dict[str, Any]:
    events = [
        item
        for item in after["events"]
        if text in str(item["canonical_plate_text"])
    ]
    decisions = [
        item
        for item in after["decisions"]
        if text
        in {
            str(item.get("event_ocr", "")),
            str(item.get("latest_track_ocr", "")),
            str(item.get("tracklet_ocr", "")),
        }
    ]
    return {"events": events, "decisions": decisions}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--videos",
        nargs="+",
        type=Path,
        default=[ROOT / "input/test1_60fps.mp4", ROOT / "input/test2_30fps.mp4"],
    )
    parser.add_argument("--output", type=Path, default=ROOT / "output/t4_benchmark")
    parser.add_argument("--modes", nargs="+", choices=("sort", "byte"), default=("sort", "byte"))
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
    detector.session.run(
        [item["name"] for item in detector.output_info],
        {
            detector.input_name: np.zeros(
                [
                    value if isinstance(value, int) and value > 0 else 1
                    for value in detector.input_shape
                ],
                dtype=np.float32,
            )
        },
    )
    ocr.session.run(
        [ocr.output_name],
        {ocr.input_name: np.zeros((1, 3, ocr.input_height, ocr.input_width), dtype=np.float32)},
    )

    report: dict[str, Any] = {
        "phase": "T4",
        "configuration": {
            "stitching_enabled_default": STITCHING_ENABLED,
            "max_gap_sec": STITCH_MAX_GAP_SEC,
            "max_edit_distance": STITCH_MAX_EDIT_DISTANCE,
            "min_fuzzy_text_length": STITCH_MIN_FUZZY_TEXT_LENGTH,
            "max_center_distance_ratio": STITCH_MAX_CENTER_DISTANCE_RATIO,
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
        for mode in args.modes:
            mode_report: dict[str, Any] = {}
            for label, enabled in (("before", False), ("after", True)):
                run_output = output_root / video.stem / mode / label
                detector.conf_threshold = BENCHMARK_DETECTOR_CONF_THRESHOLD
                processor = VideoProcessor(
                    detector,
                    output_dir=run_output,
                    project_root=ROOT,
                    tracker=make_tracker(mode),
                    tracker_mode=mode,
                    quality_evaluator=quality_evaluator(),
                    result_writer=VideoResultWriter(
                        output_dir=run_output / "json", project_root=ROOT
                    ),
                    ocr=ocr,
                    top_k=RECOGNITION_TOP_K,
                    write_json=False,
                    stitching_enabled=enabled,
                )
                result = processor.process(video)
                mode_report[label] = summarize(result, run_output)
                print(
                    f"{video.name} {mode} {label}: "
                    f"raw={result['total_tracks']} "
                    f"events={len(result['best_crops'])} "
                    f"ocr={result['ocr_calls']} "
                    f"time={result['total_processing_time']:.2f}s"
                )

            before = mode_report["before"]
            after = mode_report["after"]
            if before["raw_track_count"] != after["raw_track_count"]:
                raise RuntimeError(f"Raw tracker output changed for {video.name} {mode}")
            if before["ocr_calls"] != after["ocr_calls"]:
                raise RuntimeError(f"OCR calls changed for {video.name} {mode}")
            mode_report["comparison"] = {
                "raw_tracks_unchanged": True,
                "ocr_calls_unchanged": True,
                "event_reduction": before["final_plate_event_count"]
                - after["final_plate_event_count"],
                "processing_delta_seconds": after["processing_time_seconds"]
                - before["processing_time_seconds"],
                "stitching_overhead_ms": after["stitching_ms"],
            }
            mode_report["targets"] = {
                "61C2879": target_trace(after, "61C2879"),
                "59N304864": target_trace(after, "59N304864"),
            }
            video_report[mode] = mode_report
        report["videos"][video.name] = video_report

    report_path = output_root / "benchmark_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"T4 benchmark report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
