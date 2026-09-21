"""Run the V3 vehicle-to-plate pipeline on an image or complete video."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.plate_detector import (
    PlateDetector,
    TrackedPlateCandidate,
    crop_vehicle_roi,
    detect_tracked_plates,
    local_bbox_to_global,
    select_best_plate,
)
from src.plate_ownership import resolve_plate_ownership_detailed
from src.plate_ownership_temporal import (
    TemporalPlateOwnershipConfig,
    TemporalPlateOwnershipResolver,
)
from src.plate_quality import crop_plate_from_frame, score_plate_quality
from src.plate_buffer import BufferedPlateCandidate
from src.microcharnet_ocr import (
    MicroCharNetOCR,
    run_ocr_on_image_candidates,
    write_ocr_json,
)
from src.tracking import ByteTracker, TrackedVehicle
from src.vehicle_detector import VehicleDetector


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--vehicle-model",
        type=Path,
        default=Path("models/vehicle/yolo26n.onnx"),
    )
    parser.add_argument(
        "--plate-model", type=Path, default=Path("models/plate/best.onnx")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--vehicle-confidence",
        type=float,
        default=None,
        help="Default: 0.25 for image, 0.10 for video tracking",
    )
    parser.add_argument("--plate-confidence", type=float, default=0.25)
    parser.add_argument("--plate-iou", type=float, default=0.45)
    parser.add_argument("--plate-duplicate-iou-threshold", type=float, default=0.80)
    parser.add_argument("--no-temporal-ownership", action="store_true")
    parser.add_argument("--ownership-history-size", type=int, default=20)
    parser.add_argument("--ownership-min-history-samples", type=int, default=5)
    parser.add_argument("--ownership-conflict-iou-threshold", type=float, default=0.70)
    parser.add_argument("--ownership-overlap-over-smaller-threshold", type=float, default=0.85)
    parser.add_argument("--ownership-min-history-update-margin", type=float, default=0.10)
    parser.add_argument("--track-high-threshold", type=float, default=0.25)
    parser.add_argument("--new-track-threshold", type=float, default=0.25)
    parser.add_argument("--match-cost-threshold", type=float, default=0.80)
    parser.add_argument("--second-match-cost-threshold", type=float, default=0.50)
    parser.add_argument("--unconfirmed-match-cost-threshold", type=float, default=0.80)
    parser.add_argument("--track-buffer-seconds", type=float, default=1.0)
    parser.add_argument("--min-confirmed-hits", type=int, default=2)
    parser.add_argument("--duplicate-iou-threshold", type=float, default=0.85)
    parser.add_argument("--no-fuse-score", action="store_true")
    parser.add_argument("--save-crops", action="store_true")
    parser.add_argument(
        "--run-ocr", action="store_true", help="Run V5 OCR once per resolved image plate"
    )
    parser.add_argument(
        "--ocr-model", type=Path, default=Path("models/OCR/microcharnet.onnx")
    )
    parser.add_argument("--ocr-confidence", type=float, default=0.25)
    parser.add_argument("--ocr-iou", type=float, default=0.70)
    parser.add_argument(
        "--ocr-preprocess-mode",
        choices=("letterbox", "direct_resize"),
        default="letterbox",
        help="letterbox is the Ultralytics-compatible hypothesis; direct_resize is experimental",
    )
    parser.add_argument("--debug-ocr", action="store_true")
    parser.add_argument("--full-json", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-ownership", action="store_true")
    return parser.parse_args()


def create_temporal_resolver(args: argparse.Namespace) -> TemporalPlateOwnershipResolver | None:
    if args.no_temporal_ownership:
        return None
    return TemporalPlateOwnershipResolver(
        TemporalPlateOwnershipConfig(
            history_size=args.ownership_history_size,
            min_history_samples=args.ownership_min_history_samples,
            definite_duplicate_iou_threshold=args.plate_duplicate_iou_threshold,
            ownership_conflict_iou_threshold=args.ownership_conflict_iou_threshold,
            overlap_over_smaller_threshold=args.ownership_overlap_over_smaller_threshold,
            min_history_update_margin=args.ownership_min_history_update_margin,
        )
    )


def track_color(track_id: int) -> tuple[int, int, int]:
    return (
        64 + (track_id * 47) % 192,
        64 + (track_id * 89) % 192,
        64 + (track_id * 137) % 192,
    )


def _draw_label(
    image: np.ndarray,
    label: str,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = int(np.clip(x1, 0, width - 1))
    y1 = int(np.clip(y1, 0, height - 1))
    x2 = int(np.clip(x2, 0, width))
    y2 = int(np.clip(y2, 0, height))
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1
    )
    label_top = max(0, y1 - text_height - baseline - 5)
    cv2.rectangle(
        image,
        (x1, label_top),
        (min(width, x1 + text_width + 7), y1),
        color,
        -1,
    )
    cv2.putText(
        image,
        label,
        (x1 + 3, max(text_height, y1 - baseline - 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_video_results(
    frame: np.ndarray,
    tracks: list[TrackedVehicle],
    plates: list[TrackedPlateCandidate],
) -> np.ndarray:
    annotated = frame.copy()
    for track in tracks:
        _draw_label(
            annotated,
            f"ID {track.track_id} | {track.class_name} | {track.confidence:.2f}",
            track.bbox,
            track_color(track.track_id),
        )
    for plate in plates:
        _draw_label(
            annotated,
            f"{plate.plate_class_name} | {plate.plate_confidence:.2f}",
            plate.plate_bbox,
            (0, 0, 255),
            thickness=2,
        )
    return annotated


def save_plate_crop(
    frame: np.ndarray,
    plate: TrackedPlateCandidate,
    crop_directory: Path,
) -> None:
    x1, y1, x2, y2 = plate.plate_bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return
    crop_directory.mkdir(parents=True, exist_ok=True)
    filename = (
        f"track_{plate.track_id}_frame_{plate.frame_index}_"
        f"conf_{plate.plate_confidence:.2f}.jpg"
    )
    cv2.imwrite(str(crop_directory / filename), crop)


def create_detectors(
    args: argparse.Namespace, vehicle_confidence: float
) -> tuple[VehicleDetector, PlateDetector]:
    vehicle_detector = VehicleDetector(
        model_path=args.vehicle_model,
        confidence_threshold=vehicle_confidence,
    )
    plate_detector = PlateDetector(
        model_path=args.plate_model,
        confidence_threshold=args.plate_confidence,
        iou_threshold=args.plate_iou,
        debug=args.debug,
    )
    return vehicle_detector, plate_detector


def process_image(args: argparse.Namespace) -> None:
    image = cv2.imread(str(args.input))
    if image is None:
        raise ValueError(f"Could not read image: {args.input}")
    frame_height, frame_width = image.shape[:2]
    vehicle_confidence = (
        0.25 if args.vehicle_confidence is None else args.vehicle_confidence
    )
    vehicle_detector, plate_detector = create_detectors(
        args, vehicle_confidence
    )

    vehicle_started = time.perf_counter()
    vehicles = vehicle_detector.detect(image)
    vehicle_seconds = time.perf_counter() - vehicle_started
    annotated = image.copy()
    vehicle_payloads: list[dict[str, object]] = []
    total_plate_candidates = 0
    raw_candidates: list[TrackedPlateCandidate] = []
    local_bboxes: dict[int, tuple[int, int, int, int]] = {}

    for vehicle_index, vehicle in enumerate(vehicles):
        _draw_label(
            annotated,
            f"{vehicle.class_name} | {vehicle.confidence:.2f}",
            vehicle.bbox,
            track_color(vehicle_index + 1),
        )
        payload: dict[str, object] = {
            "vehicle_index": vehicle_index,
            "class_id": vehicle.class_id,
            "class_name": vehicle.class_name,
            "confidence": round(vehicle.confidence, 4),
            "vehicle_bbox_xyxy": list(vehicle.bbox),
            "plate": None,
        }
        cropped = crop_vehicle_roi(image, vehicle.bbox)
        if cropped is not None:
            vehicle_roi, vehicle_bbox = cropped
            plates = plate_detector.detect(vehicle_roi)
            total_plate_candidates += len(plates)
            best_plate = select_best_plate(plates)
            if best_plate is not None:
                global_bbox = local_bbox_to_global(
                    best_plate.bbox,
                    vehicle_bbox,
                    frame_width,
                    frame_height,
                )
                raw_candidates.append(
                    TrackedPlateCandidate(
                        frame_index=0,
                        track_id=vehicle_index,
                        vehicle_class_id=vehicle.class_id,
                        vehicle_class_name=vehicle.class_name,
                        vehicle_confidence=vehicle.confidence,
                        vehicle_bbox=vehicle_bbox,
                        plate_class_id=best_plate.class_id,
                        plate_class_name=best_plate.class_name,
                        plate_confidence=best_plate.confidence,
                        plate_bbox=global_bbox,
                    )
                )
                local_bboxes[vehicle_index] = best_plate.bbox
        vehicle_payloads.append(payload)

    temporal_resolver = create_temporal_resolver(args)
    if temporal_resolver is None:
        resolution = resolve_plate_ownership_detailed(
            raw_candidates,
            duplicate_iou_threshold=args.plate_duplicate_iou_threshold,
            debug=args.debug_ownership,
        )
        resolved_candidates = list(resolution.candidates)
        ownership_stats = resolution.stats
    else:
        temporal_resolution = temporal_resolver.resolve(0, raw_candidates)
        resolved_candidates = list(temporal_resolution.candidates)
        ownership_stats = temporal_resolution.stats
    for candidate in resolved_candidates:
        vehicle_payloads[candidate.track_id]["plate"] = {
            "class_id": candidate.plate_class_id,
            "class_name": candidate.plate_class_name,
            "confidence": round(candidate.plate_confidence, 4),
            "bbox_xyxy": list(candidate.plate_bbox),
            "local_bbox_xyxy": list(local_bboxes[candidate.track_id]),
        }
        _draw_label(
            annotated,
            f"{candidate.plate_class_name} | {candidate.plate_confidence:.2f}",
            candidate.plate_bbox,
            (0, 0, 255),
        )
        if args.save_crops:
            save_plate_crop(
                image,
                candidate,
                args.output_dir / "debug_plate_crops",
            )

    vehicles_with_plate = len(resolved_candidates)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_output = args.output_dir / f"{args.input.stem}_vehicle_plate{args.input.suffix}"
    json_output = args.output_dir / f"{args.input.stem}_vehicle_plate.json"
    ocr_json_output: Path | None = None
    ocr_engine = None
    ocr_candidates = ()
    if args.run_ocr:
        image_candidates: list[BufferedPlateCandidate] = []
        for candidate in resolved_candidates:
            crop = crop_plate_from_frame(image, candidate.plate_bbox)
            if crop is None:
                continue
            image_candidates.append(
                BufferedPlateCandidate(
                    frame_index=0,
                    track_id=candidate.track_id,
                    plate_class_id=candidate.plate_class_id,
                    plate_class_name=candidate.plate_class_name,
                    plate_confidence=candidate.plate_confidence,
                    bbox=candidate.plate_bbox,
                    quality=score_plate_quality(crop, candidate.plate_confidence),
                    crop=crop,
                )
            )
        ocr_engine = MicroCharNetOCR(
            args.ocr_model,
            conf_threshold=args.ocr_confidence,
            iou_threshold=args.ocr_iou,
            preprocess_mode=args.ocr_preprocess_mode,
        )
        ocr_candidates = run_ocr_on_image_candidates(
            image_candidates, ocr_engine, debug=args.debug_ocr
        )
        ocr_json_output = args.output_dir / f"{args.input.stem}_ocr_candidates.json"
        write_ocr_json(ocr_json_output, ocr_candidates, ocr_engine)
    if not cv2.imwrite(str(image_output), annotated):
        raise OSError(f"Could not write image: {image_output}")
    plate_calls = plate_detector.detect_call_count
    payload = {
        "status": "ok",
        "input_type": "image",
        "vehicle_count": len(vehicles),
        "vehicles_with_plate": vehicles_with_plate,
        "plate_candidates": total_plate_candidates,
        "ownership": {
            "raw_plate_candidates": ownership_stats.raw_candidates,
            "duplicate_groups": (
                ownership_stats.duplicate_groups
                if hasattr(ownership_stats, "duplicate_groups")
                else ownership_stats.conflict_groups
            ),
            "removed_candidates": (
                ownership_stats.removed_duplicate_candidates
                if hasattr(ownership_stats, "removed_duplicate_candidates")
                else ownership_stats.removed_conflict_candidates
            ),
            "final_plate_results": ownership_stats.final_results,
            "mode": "v3.2_temporal" if temporal_resolver is not None else "v3.1_legacy",
        },
        "performance": {
            "vehicle_detector_ms": round(vehicle_seconds * 1000.0, 3),
            "plate_ms_per_vehicle_roi": round(
                plate_detector.total_processing_seconds * 1000.0 / plate_calls,
                3,
            )
            if plate_calls
            else 0.0,
        },
        "vehicles": vehicle_payloads,
    }
    if ocr_json_output is not None:
        payload["v5_ocr_json"] = str(ocr_json_output)
    json_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Vehicles: {len(vehicles)}")
    print(f"Vehicles with plate: {vehicles_with_plate}")
    print(f"Post-NMS plate candidates: {total_plate_candidates}")
    print(
        "Ownership: "
        f"raw={ownership_stats.raw_candidates}, "
        f"groups={getattr(ownership_stats, 'duplicate_groups', getattr(ownership_stats, 'conflict_groups', 0))}, "
        f"removed={getattr(ownership_stats, 'removed_duplicate_candidates', getattr(ownership_stats, 'removed_conflict_candidates', 0))}, "
        f"final={ownership_stats.final_results}"
    )
    print(f"Saved image: {image_output}")
    print(f"Saved JSON: {json_output}")
    if ocr_engine is not None and ocr_json_output is not None:
        print(f"V5 image OCR calls: {ocr_engine.inference_count}")
        print(f"V5 image OCR session init count: {ocr_engine.session_init_count}")
        print(f"Saved V5 OCR JSON: {ocr_json_output}")

    if args.preview:
        cv2.imshow("Vehicle + plate", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(args: argparse.Namespace) -> None:
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {args.input}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not np.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError(f"Invalid video metadata: {width}x{height} at {fps} FPS")

    vehicle_confidence = (
        0.10 if args.vehicle_confidence is None else args.vehicle_confidence
    )
    vehicle_detector, plate_detector = create_detectors(
        args, vehicle_confidence
    )
    tracker = ByteTracker(
        fps=fps,
        track_low_threshold=vehicle_confidence,
        track_high_threshold=args.track_high_threshold,
        new_track_threshold=args.new_track_threshold,
        match_cost_threshold=args.match_cost_threshold,
        second_match_cost_threshold=args.second_match_cost_threshold,
        unconfirmed_match_cost_threshold=args.unconfirmed_match_cost_threshold,
        track_buffer_seconds=args.track_buffer_seconds,
        min_confirmed_hits=args.min_confirmed_hits,
        duplicate_iou_threshold=args.duplicate_iou_threshold,
        fuse_score=not args.no_fuse_score,
        debug=args.debug,
    )
    temporal_resolver = create_temporal_resolver(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_output = args.output_dir / f"{args.input.stem}_vehicle_plate.mp4"
    json_output = args.output_dir / f"{args.input.stem}_vehicle_plate.json"
    writer = cv2.VideoWriter(
        str(video_output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise OSError(f"Could not create video: {video_output}")

    frame_index = 0
    vehicle_seconds = 0.0
    tracking_seconds = 0.0
    ownership_seconds = 0.0
    confirmed_vehicle_observations = 0
    frames_with_plate = 0
    plate_detection_count = 0
    raw_plate_candidate_count = 0
    ownership_duplicate_groups = 0
    ownership_removed_candidates = 0
    ownership_invalid_candidates = 0
    plate_frames: dict[int, list[dict[str, object]]] = defaultdict(list)
    crop_directory = args.output_dir / "debug_plate_crops"
    started = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            stage_started = time.perf_counter()
            detections = vehicle_detector.detect(frame)
            vehicle_seconds += time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            tracks = tracker.update(detections, frame_index)
            tracking_seconds += time.perf_counter() - stage_started
            confirmed_vehicle_observations += len(tracks)

            raw_candidates = detect_tracked_plates(
                frame, tracks, plate_detector, frame_index
            )
            raw_plate_candidate_count += len(raw_candidates)
            ownership_started = time.perf_counter()
            if temporal_resolver is None:
                resolution = resolve_plate_ownership_detailed(
                    raw_candidates,
                    duplicate_iou_threshold=args.plate_duplicate_iou_threshold,
                    debug=args.debug_ownership,
                )
                tracked_plates = list(resolution.candidates)
                duplicate_groups = resolution.stats.duplicate_groups
                removed_candidates = resolution.stats.removed_duplicate_candidates
                invalid_candidates = resolution.stats.invalid_candidates
            else:
                temporal_resolution = temporal_resolver.resolve(
                    frame_index, raw_candidates
                )
                tracked_plates = list(temporal_resolution.candidates)
                duplicate_groups = temporal_resolution.stats.conflict_groups
                removed_candidates = temporal_resolution.stats.removed_conflict_candidates
                invalid_candidates = temporal_resolution.stats.invalid_candidates
            ownership_seconds += time.perf_counter() - ownership_started
            ownership_duplicate_groups += duplicate_groups
            ownership_removed_candidates += removed_candidates
            ownership_invalid_candidates += invalid_candidates
            plate_detection_count += len(tracked_plates)
            if tracked_plates:
                frames_with_plate += 1
            for plate in tracked_plates:
                plate_frames[plate.track_id].append(
                    {
                        "frame_index": frame_index,
                        "class_id": plate.plate_class_id,
                        "class_name": plate.plate_class_name,
                        "confidence": round(plate.plate_confidence, 4),
                        "bbox_xyxy": list(plate.plate_bbox),
                    }
                )
                if args.save_crops:
                    save_plate_crop(frame, plate, crop_directory)
            if temporal_resolver is not None:
                temporal_resolver.cleanup(
                    track.track_id
                    for track in tracker.all_tracks
                    if track.state.name != "REMOVED"
                )

            annotated = draw_video_results(frame, tracks, tracked_plates)
            writer.write(annotated)
            processed = frame_index + 1
            if processed == 1 or processed % 30 == 0:
                print(
                    f"Frame {processed}: confirmed={len(tracks)}, "
                    f"plates={len(tracked_plates)}/{len(raw_candidates)}",
                    flush=True,
                )
            if args.preview:
                cv2.imshow("Vehicle + plate - press q to stop", annotated)
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
    plate_calls = plate_detector.detect_call_count
    track_payloads: list[dict[str, object]] = []
    for track in tracker.all_tracks:
        frames = plate_frames.get(track.track_id, [])
        track_payload: dict[str, object] = {
            "track_id": track.track_id,
            "class_id": track.class_id,
            "class_name": track.class_name,
            "first_frame": track.first_frame,
            "last_frame": track.last_frame,
            "hits": track.hits,
            "plate_frame_count": len(frames),
            "best_plate_confidence": (
                max(frame["confidence"] for frame in frames) if frames else None
            ),
        }
        if args.full_json:
            track_payload["plate_frames"] = frames
        track_payloads.append(track_payload)

    average = lambda seconds: seconds * 1000.0 / frame_count if frame_count else 0.0
    plate_ms_per_roi = (
        plate_detector.total_processing_seconds * 1000.0 / plate_calls
        if plate_calls
        else 0.0
    )
    payload = {
        "status": "ok",
        "video": {
            "fps": round(fps, 4),
            "frames": frame_count,
            "width": width,
            "height": height,
        },
        "summary": {
            "confirmed_tracks": tracker.unique_track_count,
            "frames_with_plate": frames_with_plate,
            "plate_detections": plate_detection_count,
            "raw_plate_candidates": raw_plate_candidate_count,
            "ownership_duplicate_groups": ownership_duplicate_groups,
            "ownership_removed_candidates": ownership_removed_candidates,
            "ownership_invalid_candidates": ownership_invalid_candidates,
            "final_plate_results": plate_detection_count,
            "confirmed_vehicle_observations": confirmed_vehicle_observations,
            "vehicle_rois_processed": plate_calls,
            "ownership_mode": "v3.2_temporal"
            if temporal_resolver is not None
            else "v3.1_legacy",
            "temporal_history_updates": (
                temporal_resolver.total_history_updates
                if temporal_resolver is not None
                else 0
            ),
        },
        "performance": {
            "vehicle_inference_ms_per_frame": round(average(vehicle_seconds), 3),
            "tracking_ms_per_frame": round(average(tracking_seconds), 3),
            "plate_ms_per_vehicle_roi": round(plate_ms_per_roi, 3),
            "plate_onnx_ms_per_vehicle_roi": round(
                plate_detector.total_inference_seconds * 1000.0 / plate_calls,
                3,
            )
            if plate_calls
            else 0.0,
            "plate_total_ms_per_frame": round(
                average(plate_detector.total_processing_seconds), 3
            ),
            "ownership_resolver_ms_per_frame": round(
                average(ownership_seconds), 6
            ),
            "v32_relative_geometry_ms_per_frame": round(
                temporal_resolver.total_relative_geometry_seconds * 1000.0 / frame_count
                if temporal_resolver is not None and frame_count
                else 0.0,
                6,
            ),
            "v32_history_scoring_ms_per_frame": round(
                temporal_resolver.total_history_scoring_seconds * 1000.0 / frame_count
                if temporal_resolver is not None and frame_count
                else 0.0,
                6,
            ),
            "average_confirmed_vehicles_per_frame": round(
                confirmed_vehicle_observations / frame_count
                if frame_count
                else 0.0,
                3,
            ),
            "elapsed_seconds": round(elapsed_seconds, 4),
            "end_to_end_fps": round(
                frame_count / elapsed_seconds if elapsed_seconds else 0.0, 3
            ),
        },
        "tracks": track_payloads,
    }
    json_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Processed frames: {frame_count}")
    print(f"Confirmed tracks: {tracker.unique_track_count}")
    print(f"Frames with plate: {frames_with_plate}")
    print(f"Raw plate candidates: {raw_plate_candidate_count}")
    print(f"Ownership duplicate groups: {ownership_duplicate_groups}")
    print(
        "Ownership mode: "
        f"{'V3.2 temporal' if temporal_resolver is not None else 'V3.1 legacy'}"
    )
    print(f"Ownership removed candidates: {ownership_removed_candidates}")
    print(f"Final plate results: {plate_detection_count}")
    print(f"Ownership resolver: {average(ownership_seconds):.6f} ms/frame")
    print(f"Plate processing: {plate_ms_per_roi:.2f} ms/vehicle ROI")
    print(
        "Average confirmed vehicles: "
        f"{confirmed_vehicle_observations / frame_count if frame_count else 0.0:.2f}/frame"
    )
    print(f"End-to-end FPS: {frame_count / elapsed_seconds if elapsed_seconds else 0.0:.2f}")
    print(f"Saved video: {video_output}")
    print(f"Saved JSON: {json_output}")


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input not found: {args.input}")
    extension = args.input.suffix.lower()
    if extension in IMAGE_EXTENSIONS:
        process_image(args)
    elif extension in VIDEO_EXTENSIONS:
        process_video(args)
    else:
        raise ValueError(f"Unsupported input extension: {extension or '<none>'}")


if __name__ == "__main__":
    main()
