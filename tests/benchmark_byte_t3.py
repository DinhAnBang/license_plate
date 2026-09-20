"""Phase T3 source benchmark for SORT and lifecycle-aware BYTE tracking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from core.byte_tracker import ByteTracker
from core.config import (
    BENCHMARK_DETECTOR_CONF_THRESHOLD,
    BYTE_HIGH_MATCH_IOU_THRESHOLD,
    BYTE_HIGH_THRESHOLD,
    BYTE_LOW_MATCH_IOU_THRESHOLD,
    BYTE_LOW_THRESHOLD,
    BYTE_REFERENCE_FPS,
    BYTE_TRACK_BUFFER_FRAMES_AT_30FPS,
    BYTE_UNCONFIRMED_MATCH_IOU_THRESHOLD,
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
from core.detector import PlateDetector
from core.ocr import MicroCharNetOCR
from core.quality import PlateQualityEvaluator
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


def plate_by_track(result: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(item["track_id"]): item for item in result["best_crops"]}


def summarize_sort(result: dict[str, Any], tracker: SortTracker) -> dict[str, Any]:
    hits = [int(track["hits"]) for track in result["tracks"]]
    return {
        "frames": int(result["processed_frames"]),
        "high_detections": int(result["total_detections"]),
        "final_tracks": int(result["total_tracks"]),
        "average_total_hits": sum(hits) / len(hits) if hits else 0.0,
        "max_total_hits": max(hits, default=0),
        "ocr_calls": int(result["ocr_calls"]),
        "tracking_ms_per_frame": float(result["avg_tracking_ms"]),
        "processing_time_seconds": float(result["total_processing_time"]),
        "statistics": tracker.statistics(),
        "tracks": result["tracks"],
    }


def qualified_track_key(video_name: str, track_id: int) -> str:
    """Return the globally unambiguous benchmark identity for one track."""

    return f"{video_name}#track_{track_id}"


def summarize_byte(
    result: dict[str, Any], tracker: ByteTracker, video_name: str
) -> dict[str, Any]:
    plates = plate_by_track(result)
    diagnostics: list[dict[str, Any]] = []
    for track in result["tracks"]:
        track_id = int(track["track_id"])
        plate = plates.get(track_id, {})
        diagnostics.append(
            {
                "video": video_name,
                "track_key": qualified_track_key(video_name, track_id),
                "track_id": track_id,
                "state": track["state"],
                "first_frame": int(track["first_frame"]),
                "last_frame": int(track["last_frame"]),
                "total_hits": int(track["hits"]),
                "high_hits": int(track["high_hits"]),
                "low_hits": int(track["low_hits"]),
                "high_ratio": float(track["high_ratio"]),
                "ocr_text": str(plate.get("text", "")),
                "raw_text": str(plate.get("raw_text", "")),
                "ocr_confidence": float(plate.get("ocr_conf", 0.0)),
                "crop": plate.get("crop"),
            }
        )
    total_hits = [item["total_hits"] for item in diagnostics]
    high_hits = [item["high_hits"] for item in diagnostics]
    low_hits = [item["low_hits"] for item in diagnostics]
    stats = tracker.statistics()
    return {
        "frames": int(result["processed_frames"]),
        "high_detections": tracker.high_detections,
        "low_detections": tracker.low_detections,
        "new_tracks_created": tracker.created_tracks,
        "unconfirmed_tracks_created": tracker.unconfirmed_tracks_created,
        "unconfirmed_confirmed": tracker.unconfirmed_confirmed,
        "unconfirmed_removed": tracker.unconfirmed_removed,
        "high_matches": tracker.high_matches,
        "low_recoveries": tracker.low_score_recoveries,
        "tracks_marked_lost": tracker.tracks_marked_lost,
        "tracks_reactivated": tracker.reactivated_tracks,
        "tracks_removed_after_buffer": tracker.removed_after_buffer,
        "confirmed_final_tracks": int(result["total_tracks"]),
        "average_total_hits": sum(total_hits) / len(total_hits) if total_hits else 0.0,
        "average_high_hits": sum(high_hits) / len(high_hits) if high_hits else 0.0,
        "average_low_hits": sum(low_hits) / len(low_hits) if low_hits else 0.0,
        "max_total_hits": max(total_hits, default=0),
        "ocr_calls": int(result["ocr_calls"]),
        "tracking_ms_per_frame": float(result["avg_tracking_ms"]),
        "processing_time_seconds": float(result["total_processing_time"]),
        "effective_track_buffer": tracker.effective_track_buffer,
        "statistics": stats,
        "tracks": diagnostics,
        "rejected_unconfirmed_tracks": [
            {
                "video": video_name,
                "track_key": qualified_track_key(video_name, int(item["track_id"])),
                **item,
            }
            for item in tracker.all_track_summaries()
            if not item["is_activated"]
        ],
        "suspicious_low_high_ratio_tracks": [
            item for item in diagnostics if item["total_hits"] >= 10 and item["high_ratio"] < 0.20
        ],
    }


def trace_diagnostics(
    video_name: str,
    summary: dict[str, Any],
    tracker: ByteTracker,
) -> dict[str, Any]:
    target_ids = {
        int(track["track_id"])
        for track in summary["tracks"]
        if "59N3" in track["ocr_text"] or "59N3" in track["raw_text"]
    }
    if video_name == "test2_30fps.mp4" and not target_ids:
        target_ids = {
            int(track["track_id"])
            for track in summary["tracks"]
            if track["first_frame"] <= 965 and track["last_frame"] >= 938
        }

    sign_candidates: list[dict[str, Any]] = []
    sign_ids: set[int] = set()
    if video_name == "test1_60fps.mp4":
        sign_candidates = [
            track
            for track in summary["tracks"]
            if track["first_frame"] <= 60 and track["last_frame"] >= 240
        ]
        if sign_candidates:
            chosen = max(sign_candidates, key=lambda item: item["total_hits"])
            sign_ids.add(int(chosen["track_id"]))

    selected_ids = target_ids | sign_ids
    return {
        "video": video_name,
        "target_59N304864_track_ids": sorted(target_ids),
        "con_phong_candidate_track_ids": sorted(sign_ids),
        "con_phong_candidates": sign_candidates,
        "trace": [
            row for row in tracker.trace_records if int(row["track_id"]) in selected_ids
        ],
    }


def load_t2_reference() -> dict[str, Any] | None:
    path = ROOT / "output/t2_benchmark/benchmark_report.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def qualify_existing_report_tracks(report: dict[str, Any]) -> None:
    """Backfill per-video track identity when resuming a prior T3 report."""

    report["track_identity_scope"] = "track_id is unique only within one video run"
    for video_name, video_report in report.get("videos", {}).items():
        byte_summary = video_report.get("byte_t3")
        if not isinstance(byte_summary, dict):
            continue
        for collection_name in (
            "tracks",
            "rejected_unconfirmed_tracks",
            "suspicious_low_high_ratio_tracks",
        ):
            for item in byte_summary.get(collection_name, []):
                track_id = int(item["track_id"])
                item["video"] = video_name
                item["track_key"] = qualified_track_key(video_name, track_id)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--videos",
        nargs="+",
        type=Path,
        default=[ROOT / "input/test1_60fps.mp4", ROOT / "input/test2_30fps.mp4"],
    )
    parser.add_argument("--output", type=Path, default=ROOT / "output/t3_benchmark")
    parser.add_argument(
        "--byte-only",
        action="store_true",
        help="rerun only BYTE T3 and preserve existing SORT/reference results",
    )
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
    detector.session.run(
        [item["name"] for item in detector.output_info],
        {
            detector.input_name: np.zeros(
                [
                    dimension if isinstance(dimension, int) and dimension > 0 else 1
                    for dimension in detector.input_shape
                ],
                dtype=np.float32,
            )
        },
    )
    ocr.session.run(
        [ocr.output_name],
        {ocr.input_name: np.zeros((1, 3, ocr.input_height, ocr.input_width), dtype=np.float32)},
    )

    report_path = output_root / "benchmark_report.json"
    if args.byte_only and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {
            "phase": "T3",
            "track_identity_scope": "track_id is unique only within one video run",
            "configuration": {
                "track_high_thresh": BYTE_HIGH_THRESHOLD,
                "track_low_thresh": BYTE_LOW_THRESHOLD,
                "detector_nms_iou_threshold": DETECTOR_NMS_IOU_THRESHOLD,
                "sort_iou_threshold": SORT_IOU_THRESHOLD,
                "byte_high_match_iou_threshold": BYTE_HIGH_MATCH_IOU_THRESHOLD,
                "byte_low_match_iou_threshold": BYTE_LOW_MATCH_IOU_THRESHOLD,
                "byte_unconfirmed_match_iou_threshold": BYTE_UNCONFIRMED_MATCH_IOU_THRESHOLD,
                "byte_track_buffer_at_30fps": BYTE_TRACK_BUFFER_FRAMES_AT_30FPS,
                "byte_reference_fps": BYTE_REFERENCE_FPS,
                "sort_max_age": SORT_MAX_AGE,
                "top_k": RECOGNITION_TOP_K,
                "detector_sessions": 1,
                "ocr_sessions": ocr.session_creation_count,
            },
            "byte_t2_reference": load_t2_reference(),
            "videos": {},
        }
    qualify_existing_report_tracks(report)

    for video_value in args.videos:
        video = (video_value if video_value.is_absolute() else ROOT / video_value).resolve()
        if not video.is_file():
            raise FileNotFoundError(video)
        video_report: dict[str, Any] = report["videos"].get(video.name, {})

        if not args.byte_only:
            detector.conf_threshold = BENCHMARK_DETECTOR_CONF_THRESHOLD
            sort_tracker = SortTracker(
                iou_threshold=SORT_IOU_THRESHOLD,
                max_age=SORT_MAX_AGE,
                min_hits=SORT_MIN_HITS,
            )
            sort_result = VideoProcessor(
                detector,
                output_dir=output_root / video.stem / "sort",
                project_root=ROOT,
                tracker=sort_tracker,
                tracker_mode="sort",
                quality_evaluator=quality_evaluator(),
                ocr=ocr,
                top_k=RECOGNITION_TOP_K,
                write_json=False,
                stitching_enabled=False,
            ).process(video)
            video_report["sort"] = summarize_sort(sort_result, sort_tracker)

        detector.conf_threshold = BENCHMARK_DETECTOR_CONF_THRESHOLD
        byte_tracker = ByteTracker(trace_enabled=True)
        byte_result = VideoProcessor(
            detector,
            output_dir=output_root / video.stem / "byte_t3",
            project_root=ROOT,
            tracker=byte_tracker,
            tracker_mode="byte",
            quality_evaluator=quality_evaluator(),
            ocr=ocr,
            top_k=RECOGNITION_TOP_K,
            write_json=False,
            stitching_enabled=False,
        ).process(video)
        byte_summary = summarize_byte(byte_result, byte_tracker, video.name)
        video_report["byte_t3"] = byte_summary

        if (
            "sort" in video_report
            and video_report["sort"]["high_detections"] != byte_summary["high_detections"]
        ):
            raise RuntimeError(
                f"High detection count changed for {video.name}: "
                f"SORT={video_report['sort']['high_detections']} "
                f"BYTE={byte_summary['high_detections']}"
            )

        trace = trace_diagnostics(video.name, byte_summary, byte_tracker)
        (output_root / "traces" / f"{video.stem}.json").write_text(
            json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report["videos"][video.name] = video_report
        sort_text = (
            f"SORT{' preserved' if args.byte_only else ''} "
            f"tracks={video_report['sort']['final_tracks']} | "
            if "sort" in video_report
            else ""
        )
        print(
            f"{video.name}: {sort_text}"
            f"BYTE T3 high={byte_summary['high_detections']} low={byte_summary['low_detections']} "
            f"confirmed={byte_summary['confirmed_final_tracks']} "
            f"unconfirmed_removed={byte_summary['unconfirmed_removed']} "
            f"reactivated={byte_summary['tracks_reactivated']}"
        )

    qualify_existing_report_tracks(report)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Benchmark report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
