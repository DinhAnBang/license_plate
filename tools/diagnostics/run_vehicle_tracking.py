"""Run V2.1 vehicle detection and ByteTrack-style tracking on a complete video."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from src.tracking import ByteTracker, TrackedVehicle
from src.vehicle_detector import VehicleDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input video")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/vehicle/yolo26n.onnx"),
        help="Vehicle ONNX model",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--track-low-threshold", type=float, default=0.10)
    parser.add_argument("--track-high-threshold", type=float, default=0.25)
    parser.add_argument("--new-track-threshold", type=float, default=0.25)
    parser.add_argument("--match-cost-threshold", type=float, default=0.80)
    parser.add_argument(
        "--second-match-cost-threshold", type=float, default=0.50
    )
    parser.add_argument(
        "--unconfirmed-match-cost-threshold", type=float, default=0.80
    )
    parser.add_argument("--track-buffer-seconds", type=float, default=1.0)
    parser.add_argument("--min-confirmed-hits", type=int, default=2)
    parser.add_argument("--duplicate-iou-threshold", type=float, default=0.85)
    parser.add_argument(
        "--no-cross-class-dedup",
        action="store_true",
        help="Disable V2.1 conservative cross-class detector deduplication",
    )
    parser.add_argument(
        "--cross-class-duplicate-iou-threshold", type=float, default=0.90
    )
    parser.add_argument(
        "--cross-class-duplicate-area-ratio-threshold", type=float, default=0.80
    )
    parser.add_argument(
        "--cross-class-duplicate-center-distance-threshold",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--active-duplicate-suppression",
        action="store_true",
        help="Enable persistent TRACKED-to-TRACKED duplicate suppression",
    )
    parser.add_argument("--active-duplicate-iou-threshold", type=float, default=0.90)
    parser.add_argument("--active-duplicate-min-frames", type=int, default=3)
    parser.add_argument(
        "--no-fuse-score",
        action="store_true",
        help="Disable HIGH-detection score fusion",
    )
    parser.add_argument("--preview", action="store_true")
    parser.add_argument(
        "--debug", action="store_true", help="Print ByteTrack association counts"
    )
    parser.add_argument(
        "--debug-json",
        action="store_true",
        help="Include per-frame active tracks in the JSON output",
    )
    parser.add_argument(
        "--debug-class",
        action="store_true",
        help="Annotate the current detector class beside the voted track class",
    )
    parser.add_argument(
        "--output-suffix",
        default="_tracking_v21",
        help="Suffix for tracking video and JSON outputs",
    )
    return parser.parse_args()


def track_color(track_id: int) -> tuple[int, int, int]:
    """Return a deterministic, bright BGR colour for a track ID."""

    return (
        64 + (track_id * 47) % 192,
        64 + (track_id * 89) % 192,
        64 + (track_id * 137) % 192,
    )


def draw_tracks(
    frame: np.ndarray,
    tracks: list[TrackedVehicle],
    debug_class: bool = False,
) -> np.ndarray:
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    for track in tracks:
        x1, y1, x2, y2 = track.bbox
        x1 = int(np.clip(x1, 0, width - 1))
        y1 = int(np.clip(y1, 0, height - 1))
        x2 = int(np.clip(x2, 0, width))
        y2 = int(np.clip(y2, 0, height))
        if x2 <= x1 or y2 <= y1:
            continue

        color = track_color(track.track_id)
        label = f"ID {track.track_id} | {track.class_name} | {track.confidence:.2f}"
        if debug_class and track.current_detection_class_name is not None:
            label += f" | det={track.current_detection_class_name}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        label_top = max(0, y1 - text_height - baseline - 6)
        cv2.rectangle(
            annotated,
            (x1, label_top),
            (min(width, x1 + text_width + 8), y1),
            color,
            -1,
        )
        text_y = max(text_height, y1 - baseline - 3)
        cv2.putText(
            annotated,
            label,
            (x1 + 4, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated


def track_to_json(track: TrackedVehicle) -> dict[str, object]:
    return {
        "track_id": track.track_id,
        "class_id": track.class_id,
        "class_name": track.class_name,
        "confidence": round(track.confidence, 4),
        "bbox_xyxy": list(track.bbox),
        "current_detection_class_id": track.current_detection_class_id,
        "current_detection_class_name": track.current_detection_class_name,
    }


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input video not found: {args.input}")

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {args.input}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError(f"Video has invalid FPS: {fps}")
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError(f"Video has invalid dimensions: {width}x{height}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_output = args.output_dir / f"{args.input.stem}{args.output_suffix}.mp4"
    json_output = args.output_dir / f"{args.input.stem}{args.output_suffix}.json"
    writer = cv2.VideoWriter(
        str(video_output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise OSError(f"Could not create output video: {video_output}")

    # ByteTrack needs LOW detections, so the detector threshold is deliberately
    # lower than the HIGH/new-track thresholds.
    detector = VehicleDetector(
        model_path=args.model,
        confidence_threshold=args.track_low_threshold,
    )
    tracker = ByteTracker(
        fps=fps,
        track_high_threshold=args.track_high_threshold,
        track_low_threshold=args.track_low_threshold,
        new_track_threshold=args.new_track_threshold,
        match_cost_threshold=args.match_cost_threshold,
        second_match_cost_threshold=args.second_match_cost_threshold,
        unconfirmed_match_cost_threshold=args.unconfirmed_match_cost_threshold,
        track_buffer_seconds=args.track_buffer_seconds,
        min_confirmed_hits=args.min_confirmed_hits,
        duplicate_iou_threshold=args.duplicate_iou_threshold,
        fuse_score=not args.no_fuse_score,
        debug=args.debug,
        cross_class_dedup_enabled=not args.no_cross_class_dedup,
        cross_class_duplicate_iou_threshold=args.cross_class_duplicate_iou_threshold,
        cross_class_duplicate_area_ratio_threshold=(
            args.cross_class_duplicate_area_ratio_threshold
        ),
        cross_class_duplicate_center_distance_threshold=(
            args.cross_class_duplicate_center_distance_threshold
        ),
        active_duplicate_suppression_enabled=args.active_duplicate_suppression,
        active_duplicate_iou_threshold=args.active_duplicate_iou_threshold,
        active_duplicate_min_frames=args.active_duplicate_min_frames,
    )

    frame_index = 0
    detector_seconds = 0.0
    tracker_seconds = 0.0
    confirmed_tracks_total = 0
    rejected_tentative_total = 0
    suppressed_duplicates_total = 0
    debug_frames: list[dict[str, object]] = []
    started = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            detector_started = time.perf_counter()
            detections = detector.detect(frame)
            detector_seconds += time.perf_counter() - detector_started

            tracker_started = time.perf_counter()
            tracks = tracker.update(detections, frame_index)
            tracker_seconds += time.perf_counter() - tracker_started
            confirmed_tracks_total += int(
                tracker.last_debug_stats["newly_confirmed"]
            )
            rejected_tentative_total += int(
                tracker.last_debug_stats["removed_tentative"]
            )
            suppressed_duplicates_total += int(
                tracker.last_debug_stats["duplicates_removed"]
            )

            writer.write(draw_tracks(frame, tracks, args.debug_class))
            if args.debug_json:
                debug_frames.append(
                    {
                        "frame_index": frame_index,
                        "tracks": [track_to_json(track) for track in tracks],
                        "tracker_debug": dict(tracker.last_debug_stats),
                    }
                )

            processed = frame_index + 1
            if processed == 1 or processed % 30 == 0:
                print(
                    f"Frame {processed}: detections={len(detections)}, "
                    f"active_tracks={len(tracks)}",
                    flush=True,
                )

            if args.preview:
                cv2.imshow(
                    "Vehicle tracking - press q to stop",
                    draw_tracks(frame, tracks, args.debug_class),
                )
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    frame_index += 1
                    break
            frame_index += 1
    finally:
        elapsed_seconds = time.perf_counter() - started
        capture.release()
        writer.release()
        if args.preview:
            cv2.destroyAllWindows()

    frame_count = frame_index
    average_detector_ms = (
        detector_seconds * 1000.0 / frame_count if frame_count else 0.0
    )
    average_tracker_ms = (
        tracker_seconds * 1000.0 / frame_count if frame_count else 0.0
    )
    processing_fps = frame_count / elapsed_seconds if elapsed_seconds else 0.0
    payload: dict[str, object] = {
        "status": "ok",
        "video": {
            "fps": round(fps, 4),
            "frames": frame_count,
            "width": width,
            "height": height,
        },
        "tracking_config": {
            "track_low_threshold": args.track_low_threshold,
            "track_high_threshold": args.track_high_threshold,
            "new_track_threshold": args.new_track_threshold,
            "match_cost_threshold": args.match_cost_threshold,
            "second_match_cost_threshold": args.second_match_cost_threshold,
            "unconfirmed_match_cost_threshold": (
                args.unconfirmed_match_cost_threshold
            ),
            "track_buffer_seconds": args.track_buffer_seconds,
            "max_lost_frames": tracker.max_lost_frames,
            "min_confirmed_hits": args.min_confirmed_hits,
            "duplicate_iou_threshold": args.duplicate_iou_threshold,
            "fuse_score": tracker.fuse_score,
            "hard_class_gate": "off",
            "temporal_class_voting": "on",
            "cross_class_dedup_enabled": tracker.cross_class_dedup_enabled,
            "cross_class_duplicate_iou_threshold": (
                tracker.cross_class_duplicate_iou_threshold
            ),
            "cross_class_duplicate_area_ratio_threshold": (
                tracker.cross_class_duplicate_area_ratio_threshold
            ),
            "cross_class_duplicate_center_distance_threshold": (
                tracker.cross_class_duplicate_center_distance_threshold
            ),
            "active_duplicate_suppression_enabled": (
                tracker.active_duplicate_suppression_enabled
            ),
            "active_duplicate_iou_threshold": tracker.active_duplicate_iou_threshold,
            "active_duplicate_min_frames": tracker.active_duplicate_min_frames,
        },
        "performance": {
            "elapsed_seconds": round(elapsed_seconds, 4),
            "processing_fps": round(processing_fps, 3),
            "average_detector_ms": round(average_detector_ms, 3),
            "average_tracker_ms": round(average_tracker_ms, 3),
            "average_cross_class_dedup_ms": round(
                tracker.cross_class_dedup_seconds * 1000.0 / frame_count
                if frame_count
                else 0.0,
                3,
            ),
            "average_active_duplicate_suppression_ms": round(
                tracker.active_duplicate_suppression_seconds * 1000.0 / frame_count
                if frame_count
                else 0.0,
                3,
            ),
        },
        "unique_track_count": tracker.unique_track_count,
        "tracking_statistics": {
            "confirmed_tracks": confirmed_tracks_total,
            "rejected_tentative_tracks": rejected_tentative_total,
            "suppressed_duplicate_tracks": suppressed_duplicates_total,
            "active_duplicate_tracks_removed": tracker.diagnostics[
                "active_duplicate_tracks_removed"
            ],
            "unconfirmed_at_end": len(tracker.unconfirmed_tracks),
        },
        "v21_diagnostics": tracker.diagnostics,
        "tracks": [
            {
                "track_id": track.track_id,
                "class_id": track.class_id,
                "class_name": track.class_name,
                "first_frame": track.first_frame,
                "last_frame": track.last_frame,
                "hits": track.hits,
                "state": track.state.name.lower(),
            }
            for track in tracker.all_tracks
        ],
    }
    if args.debug_json:
        payload["frames"] = debug_frames
    json_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Processed frames: {frame_count}")
    print(f"Unique track IDs: {tracker.unique_track_count}")
    print(f"Rejected tentative tracks: {rejected_tentative_total}")
    print(f"Suppressed duplicate tracks: {suppressed_duplicates_total}")
    print(f"V2.1 diagnostics: {tracker.diagnostics}")
    print(f"Processing FPS: {processing_fps:.2f}")
    print(f"Average detector time: {average_detector_ms:.2f} ms/frame")
    print(f"Average tracker time: {average_tracker_ms:.2f} ms/frame")
    print(
        "Average cross-class dedup time: "
        f"{tracker.cross_class_dedup_seconds * 1000.0 / frame_count if frame_count else 0.0:.3f} ms/frame"
    )
    print(
        "Average active duplicate suppression time: "
        f"{tracker.active_duplicate_suppression_seconds * 1000.0 / frame_count if frame_count else 0.0:.3f} ms/frame"
    )
    print(f"Saved video: {video_output}")
    print(f"Saved JSON: {json_output}")


if __name__ == "__main__":
    main()
