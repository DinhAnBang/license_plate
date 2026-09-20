"""T5 before/after benchmark for Vietnam validation and final deduplication."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from core.config import (
    BENCHMARK_DETECTOR_CONF_THRESHOLD,
    DETECTOR_NMS_IOU_THRESHOLD,
    DUPLICATE_CENTER_DISTANCE_THRESHOLD,
    DUPLICATE_MEAN_IOU_THRESHOLD,
    DUPLICATE_MIN_SHARED_FRAMES,
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


def summarize(result: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    stitching = result["stitching"]
    metrics = dict(stitching["metrics"])
    t5 = stitching.get("t5", {})
    decisions = list(stitching.get("decisions", []))
    reason_counts = Counter(str(item["reason"]) for item in decisions if "reason" in item)
    official = result["official_result"]
    return {
        "raw_tracks": int(result["total_tracks"]),
        "plate_events_before_t5": int(metrics.get("final_plate_event_count", 0)),
        "valid": int(metrics.get("t5_valid_count", 0)),
        "uncertain": int(metrics.get("t5_uncertain_count", 0)),
        "invalid": int(metrics.get("t5_invalid_count", 0)),
        "final_events": int(len(result["best_crops"])),
        "final_crops": len(list((output_dir / "crops").glob("*.jpg"))),
        "duplicate_merges": int(metrics.get("t5_duplicate_merges", 0)),
        "sequential_duplicate_merges": int(metrics.get("sequential_duplicate_merges", 0)),
        "overlapping_duplicate_merges": int(metrics.get("overlapping_duplicate_merges", 0)),
        "ocr_calls": int(result["ocr_calls"]),
        "validator_ms": float(metrics.get("validator_ms", 0.0)),
        "dedup_ms": float(metrics.get("duplicate_ms", 0.0)),
        "total_t5_ms": float(metrics.get("total_t5_ms", 0.0)),
        "total_processing_ms": float(result["total_processing_time"]) * 1000.0,
        "processing_time_seconds": float(result["total_processing_time"]),
        "public_count": int(official["count"]),
        "decision_reason_counts": dict(reason_counts),
        "invalid_results": list(t5.get("invalid_results", [])),
        "duplicate_decisions": list(t5.get("duplicate_decisions", [])),
        "public_schema": {
            "top_level": list(official),
            "plate": list(official["plates"][0]) if official["plates"] else [],
        },
        "output_video": official["output_video"],
        "final_crop_paths": [str(path) for path in sorted((output_dir / "crops").glob("*.jpg"))],
    }


def run_video(
    detector: PlateDetector,
    ocr: MicroCharNetOCR,
    video: Path,
    output_dir: Path,
    *,
    t5_enabled: bool,
) -> dict[str, Any]:
    detector.conf_threshold = BENCHMARK_DETECTOR_CONF_THRESHOLD
    processor = VideoProcessor(
        detector,
        output_dir=output_dir,
        project_root=ROOT,
        tracker=SortTracker(
            iou_threshold=SORT_IOU_THRESHOLD,
            max_age=SORT_MAX_AGE,
            min_hits=SORT_MIN_HITS,
        ),
        tracker_mode="sort",
        quality_evaluator=quality_evaluator(),
        result_writer=VideoResultWriter(output_dir=output_dir / "json", project_root=ROOT),
        ocr=ocr,
        top_k=RECOGNITION_TOP_K,
        write_json=False,
        stitching_enabled=STITCHING_ENABLED,
        t5_enabled=t5_enabled,
    )
    result = processor.process(video)
    # Keep T4 and T5 merge counters distinct in the internal benchmark data.
    t5_metrics = result["stitching"]["metrics"]
    t5_metrics["t5_duplicate_merges"] = int(
        result["stitching"].get("t5", {}).get("duplicate_metrics", {}).get("number_of_merges", 0)
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--videos",
        nargs="+",
        type=Path,
        default=[ROOT / "input/test1_60fps.mp4", ROOT / "input/test2_30fps.mp4"],
    )
    parser.add_argument("--output", type=Path, default=ROOT / "output/t5_benchmark")
    parser.add_argument(
        "--report-dir", type=Path, default=ROOT / "benchmark/reports"
    )
    args = parser.parse_args()

    output_root = args.output.resolve()
    report_dir = args.report_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

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
                [value if isinstance(value, int) and value > 0 else 1 for value in detector.input_shape],
                dtype=np.float32,
            )
        },
    )
    ocr.session.run(
        [ocr.output_name],
        {ocr.input_name: np.zeros((1, 3, ocr.input_height, ocr.input_width), dtype=np.float32)},
    )

    report: dict[str, Any] = {
        "phase": "T5",
        "configuration": {
            "baseline": "T4 with T5 disabled",
            "after": "T4 plus Vietnam validator and final deduplication",
            "stitching_enabled": STITCHING_ENABLED,
            "duplicate_min_shared_frames": DUPLICATE_MIN_SHARED_FRAMES,
            "duplicate_mean_iou_threshold": DUPLICATE_MEAN_IOU_THRESHOLD,
            "duplicate_center_distance_threshold": DUPLICATE_CENTER_DISTANCE_THRESHOLD,
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
        for label, enabled in (("before", False), ("after", True)):
            run_output = output_root / video.stem / label
            result = run_video(detector, ocr, video, run_output, t5_enabled=enabled)
            summary = summarize(result, run_output)
            video_report[label] = summary
            print(
                f"{video.name} {label}: raw={summary['raw_tracks']} "
                f"events_before_t5={summary['plate_events_before_t5']} "
                f"valid={summary['valid']} uncertain={summary['uncertain']} "
                f"invalid={summary['invalid']} final={summary['final_events']} "
                f"merges={summary['duplicate_merges']} ocr={summary['ocr_calls']} "
                f"time={summary['processing_time_seconds']:.2f}s"
            )

        before = video_report["before"]
        after = video_report["after"]
        if before["raw_tracks"] != after["raw_tracks"]:
            raise RuntimeError(f"Raw tracker output changed for {video.name}")
        if before["ocr_calls"] != after["ocr_calls"]:
            raise RuntimeError(f"OCR calls changed for {video.name}")
        video_report["comparison"] = {
            "raw_tracks_unchanged": True,
            "ocr_calls_unchanged": True,
            "baseline_events": before["plate_events_before_t5"],
            "invalid_removed": after["invalid"],
            "duplicate_merges": after["duplicate_merges"],
            "other_event_delta": (
                before["plate_events_before_t5"]
                - after["invalid"]
                - after["duplicate_merges"]
                - after["final_events"]
            ),
            "processing_delta_seconds": after["processing_time_seconds"] - before["processing_time_seconds"],
        }
        report["videos"][video.name] = video_report

    report_path = report_dir / "t5_benchmark_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    invalid_path = report_dir / "t5_invalid_results.json"
    invalid_path.write_text(
        json.dumps(
            {
                video: data["after"]["invalid_results"]
                for video, data in report["videos"].items()
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    merge_path = report_dir / "t5_duplicate_merges.json"
    merge_path.write_text(
        json.dumps(
            {
                video: data["after"]["duplicate_decisions"]
                for video, data in report["videos"].items()
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"T5 benchmark report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
