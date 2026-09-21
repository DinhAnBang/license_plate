"""Run V1 -> V3.1 -> V4 plate quality and temporal Top-K on a video."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from src.plate_buffer import (
    BufferedPlateCandidate,
    PlateBufferConfig,
    PlateBufferManager,
)
from src.plate_detector import PlateDetector, TrackedPlateCandidate, detect_tracked_plates
from src.plate_ownership import resolve_plate_ownership_detailed
from src.plate_ownership_temporal import (
    TemporalPlateOwnershipConfig,
    TemporalPlateOwnershipResolution,
    TemporalPlateOwnershipResolver,
)
from src.plate_quality import (
    PlateQualityConfig,
    PlateQualityMetrics,
    crop_plate_from_frame,
    score_plate_quality,
)
from src.microcharnet_ocr import (
    MicroCharNetOCR,
    build_ocr_json,
    run_ocr_on_topk,
    write_ocr_json,
)
from src.ocr_fusion import (
    OCRFusionConfig,
    build_fused_json,
    format_fusion_debug,
    fuse_candidates,
)
from src.tracking import ByteTracker, TrackedVehicle
from src.vehicle_detector import VehicleDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional bounded run for V4/V5 diagnostics; default processes the full video",
    )
    parser.add_argument("--vehicle-model", type=Path, default=Path("models/vehicle/yolo26n.onnx"))
    parser.add_argument("--plate-model", type=Path, default=Path("models/plate/best.onnx"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--vehicle-confidence", type=float, default=0.10)
    parser.add_argument("--plate-confidence", type=float, default=0.25)
    parser.add_argument("--plate-iou", type=float, default=0.45)
    parser.add_argument("--plate-duplicate-iou-threshold", type=float, default=0.80)
    parser.add_argument(
        "--no-temporal-ownership",
        action="store_true",
        help="Use the V3.1 resolver instead of V3.2 temporal ownership",
    )
    parser.add_argument("--ownership-history-size", type=int, default=20)
    parser.add_argument("--ownership-min-history-samples", type=int, default=5)
    parser.add_argument(
        "--ownership-conflict-iou-threshold", type=float, default=0.70
    )
    parser.add_argument(
        "--ownership-overlap-over-smaller-threshold", type=float, default=0.85
    )
    parser.add_argument("--ownership-min-history-update-margin", type=float, default=0.10)
    parser.add_argument(
        "--compare-legacy-ownership",
        action="store_true",
        help="Run V3.1 on the same raw candidates for an A/B report",
    )
    parser.add_argument("--debug-ownership-temporal", action="store_true")
    parser.add_argument("--track-high-threshold", type=float, default=0.25)
    parser.add_argument("--new-track-threshold", type=float, default=0.25)
    parser.add_argument("--match-cost-threshold", type=float, default=0.80)
    parser.add_argument("--second-match-cost-threshold", type=float, default=0.50)
    parser.add_argument("--unconfirmed-match-cost-threshold", type=float, default=0.80)
    parser.add_argument("--track-buffer-seconds", type=float, default=1.0)
    parser.add_argument("--min-confirmed-hits", type=int, default=2)
    parser.add_argument("--duplicate-iou-threshold", type=float, default=0.85)
    parser.add_argument("--no-cross-class-dedup", action="store_true")
    parser.add_argument(
        "--cross-class-duplicate-iou-threshold", type=float, default=0.90
    )
    parser.add_argument(
        "--cross-class-duplicate-area-ratio-threshold", type=float, default=0.80
    )
    parser.add_argument("--active-duplicate-suppression", action="store_true")
    parser.add_argument("--active-duplicate-iou-threshold", type=float, default=0.90)
    parser.add_argument("--active-duplicate-min-frames", type=int, default=3)
    parser.add_argument("--no-fuse-score", action="store_true")
    parser.add_argument("--confidence-weight", type=float, default=0.30)
    parser.add_argument("--sharpness-weight", type=float, default=0.40)
    parser.add_argument("--size-weight", type=float, default=0.20)
    parser.add_argument("--exposure-weight", type=float, default=0.10)
    parser.add_argument("--sharpness-reference", type=float, default=200.0)
    parser.add_argument("--target-plate-height", type=int, default=48)
    parser.add_argument("--normalized-sharpness-height", type=int, default=64)
    parser.add_argument("--dark-clip-value", type=int, default=3)
    parser.add_argument("--bright-clip-value", type=int, default=252)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-frame-gap", type=int, default=2)
    parser.add_argument("--min-quality-score", type=float, default=0.0)
    parser.add_argument("--save-topk-crops", action="store_true")
    parser.add_argument(
        "--run-ocr",
        action="store_true",
        help="Run V5 MicroCharNet OCR only on final retained Top-K crops",
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
    parser.add_argument(
        "--debug-ocr",
        action="store_true",
        help="Print one V5 diagnostic line per retained crop",
    )
    parser.add_argument(
        "--run-ocr-fusion",
        action="store_true",
        help="Run V6 temporal fusion after V5 OCR and write *_ocr_fused.json",
    )
    parser.add_argument(
        "--exact-consensus-threshold",
        type=float,
        default=0.65,
        help="V6 weighted exact-vote threshold before sequence alignment",
    )
    parser.add_argument(
        "--include-plate-confidence",
        action="store_true",
        help="Optional V6 weight factor; default OFF because V4 already uses it",
    )
    parser.add_argument(
        "--debug-ocr-fusion",
        action="store_true",
        help="Print V6 candidate votes, medoid/alignment, and selected result",
    )
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-ownership", action="store_true")
    return parser.parse_args()


def _draw_label(
    image: np.ndarray,
    label: str,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
) -> None:
    image_height, image_width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = int(np.clip(x1, 0, image_width - 1))
    y1 = int(np.clip(y1, 0, image_height - 1))
    x2 = int(np.clip(x2, 0, image_width))
    y2 = int(np.clip(y2, 0, image_height))
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1
    )
    top = max(0, y1 - text_height - baseline - 5)
    cv2.rectangle(image, (x1, top), (min(image_width, x1 + text_width + 7), y1), color, -1)
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


def _track_color(track_id: int) -> tuple[int, int, int]:
    return (
        64 + (track_id * 47) % 192,
        64 + (track_id * 89) % 192,
        64 + (track_id * 137) % 192,
    )


def draw_results(
    frame: np.ndarray,
    tracks: list[TrackedVehicle],
    plates: list[TrackedPlateCandidate],
    qualities: dict[int, PlateQualityMetrics],
) -> np.ndarray:
    annotated = frame.copy()
    for track in tracks:
        _draw_label(
            annotated,
            f"ID {track.track_id} | {track.class_name}",
            track.bbox,
            _track_color(track.track_id),
        )
    for plate in plates:
        quality = qualities.get(plate.track_id)
        suffix = f" | Q {quality.total_score:.2f}" if quality is not None else " | invalid crop"
        _draw_label(
            annotated,
            f"{plate.plate_class_name} | {plate.plate_confidence:.2f}{suffix}",
            plate.plate_bbox,
            (0, 0, 255),
        )
    return annotated


def _distribution(values: list[float], percentiles: tuple[int, ...]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    result: dict[str, float | int] = {"count": int(array.size), "min": round(float(array.min()), 6)}
    for percentile in percentiles:
        key = "median" if percentile == 50 else f"p{percentile}"
        result[key] = round(float(np.percentile(array, percentile)), 6)
    result["max"] = round(float(array.max()), 6)
    return result


def _quality_payload(metrics: PlateQualityMetrics) -> dict[str, float | int]:
    return {
        "plate_confidence": round(metrics.plate_confidence, 6),
        "width": metrics.width,
        "height": metrics.height,
        "sharpness_raw": round(metrics.sharpness_raw, 6),
        "sharpness_score": round(metrics.sharpness_score, 6),
        "size_score": round(metrics.size_score, 6),
        "exposure_score": round(metrics.exposure_score, 6),
        "total_score": round(metrics.total_score, 6),
    }


def _candidate_payload(rank: int, candidate: BufferedPlateCandidate) -> dict[str, object]:
    return {
        "rank": rank,
        "frame_index": candidate.frame_index,
        "track_id": candidate.track_id,
        "plate_class_id": candidate.plate_class_id,
        "plate_class_name": candidate.plate_class_name,
        "plate_confidence": round(candidate.plate_confidence, 6),
        "bbox_xyxy": list(candidate.bbox),
        "quality": _quality_payload(candidate.quality),
    }


def _raw_plate_candidate_payload(candidate: TrackedPlateCandidate) -> dict[str, object]:
    return {
        "track_id": candidate.track_id,
        "vehicle_bbox": list(candidate.vehicle_bbox),
        "plate_bbox": list(candidate.plate_bbox),
        "plate_class_id": candidate.plate_class_id,
        "plate_class_name": candidate.plate_class_name,
        "plate_confidence": round(candidate.plate_confidence, 6),
        "vehicle_confidence": round(candidate.vehicle_confidence, 6),
    }


def _temporal_resolution_payload(
    resolution: TemporalPlateOwnershipResolution,
) -> dict[str, object]:
    return {
        "stats": asdict(resolution.stats),
        "selected_track_ids": [candidate.track_id for candidate in resolution.candidates],
        "candidates": [asdict(diagnostic) for diagnostic in resolution.diagnostics],
    }


def save_final_crops(manager: PlateBufferManager, output_dir: Path) -> int:
    saved = 0
    root = output_dir / "topk"
    for summary in manager.finalize_all():
        candidates = manager.get_top_candidates(summary.track_id)
        if not candidates:
            continue
        track_dir = root / f"track_{summary.track_id}"
        track_dir.mkdir(parents=True, exist_ok=True)
        for rank, candidate in enumerate(candidates, 1):
            filename = (
                f"rank_{rank}_frame_{candidate.frame_index}_"
                f"{candidate.plate_class_name}_conf_{candidate.plate_confidence:.3f}_"
                f"q_{candidate.quality.total_score:.3f}.jpg"
            )
            if not cv2.imwrite(str(track_dir / filename), candidate.crop):
                raise OSError(f"Could not save Top-K crop: {track_dir / filename}")
            saved += 1
    return saved


def process_video(args: argparse.Namespace) -> tuple[Path, Path]:
    if not args.input.is_file():
        raise FileNotFoundError(f"Input not found: {args.input}")
    if args.run_ocr_fusion and not args.run_ocr:
        raise ValueError("--run-ocr-fusion requires --run-ocr")
    quality_config = PlateQualityConfig(
        confidence_weight=args.confidence_weight,
        sharpness_weight=args.sharpness_weight,
        size_weight=args.size_weight,
        exposure_weight=args.exposure_weight,
        sharpness_reference=args.sharpness_reference,
        target_plate_height=args.target_plate_height,
        normalized_sharpness_height=args.normalized_sharpness_height,
        dark_clip_value=args.dark_clip_value,
        bright_clip_value=args.bright_clip_value,
    )
    buffer_config = PlateBufferConfig(
        top_k=args.top_k,
        min_frame_gap=args.min_frame_gap,
        min_quality_score=args.min_quality_score,
    )
    manager = PlateBufferManager(buffer_config)
    temporal_config = TemporalPlateOwnershipConfig(
        history_size=args.ownership_history_size,
        min_history_samples=args.ownership_min_history_samples,
        definite_duplicate_iou_threshold=args.plate_duplicate_iou_threshold,
        ownership_conflict_iou_threshold=args.ownership_conflict_iou_threshold,
        overlap_over_smaller_threshold=args.ownership_overlap_over_smaller_threshold,
        min_history_update_margin=args.ownership_min_history_update_margin,
    )
    temporal_resolver = (
        None
        if args.no_temporal_ownership
        else TemporalPlateOwnershipResolver(temporal_config)
    )

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {args.input}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not np.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError(f"Invalid video metadata: {width}x{height} at {fps} FPS")

    vehicle_detector = VehicleDetector(args.vehicle_model, args.vehicle_confidence)
    plate_detector = PlateDetector(args.plate_model, args.plate_confidence, args.plate_iou, args.debug)
    tracker = ByteTracker(
        fps=fps,
        track_low_threshold=args.vehicle_confidence,
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
        cross_class_dedup_enabled=not args.no_cross_class_dedup,
        cross_class_duplicate_iou_threshold=args.cross_class_duplicate_iou_threshold,
        cross_class_duplicate_area_ratio_threshold=(
            args.cross_class_duplicate_area_ratio_threshold
        ),
        active_duplicate_suppression_enabled=args.active_duplicate_suppression,
        active_duplicate_iou_threshold=args.active_duplicate_iou_threshold,
        active_duplicate_min_frames=args.active_duplicate_min_frames,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_output = args.output_dir / f"{args.input.stem}_plate_quality.mp4"
    json_output = args.output_dir / f"{args.input.stem}_plate_quality.json"
    writer = cv2.VideoWriter(str(video_output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise OSError(f"Could not create video: {video_output}")

    frame_index = 0
    resolved_events = 0
    frames_with_plate = 0
    raw_plate_candidates = 0
    ownership_groups = 0
    ownership_removed = 0
    ownership_invalid = 0
    vehicle_observations = 0
    vehicle_seconds = 0.0
    tracking_seconds = 0.0
    ownership_seconds = 0.0
    legacy_compare_seconds = 0.0
    quality_seconds = 0.0
    buffer_seconds = 0.0
    v4_seconds = 0.0
    sharpness_values: list[float] = []
    height_values: list[float] = []
    quality_values: list[float] = []
    legacy_resolved_events = 0
    legacy_ownership_groups = 0
    legacy_ownership_removed = 0
    legacy_changed_frames = 0
    temporal_conflict_groups = 0
    temporal_ambiguous_groups = 0
    temporal_history_updates = 0
    temporal_debug_frames: list[dict[str, object]] = []
    started = time.perf_counter()
    try:
        while True:
            if args.max_frames is not None and frame_index >= args.max_frames:
                break
            ok, frame = capture.read()
            if not ok:
                break
            stage = time.perf_counter()
            detections = vehicle_detector.detect(frame)
            vehicle_seconds += time.perf_counter() - stage
            stage = time.perf_counter()
            tracks = tracker.update(detections, frame_index)
            tracking_seconds += time.perf_counter() - stage
            vehicle_observations += len(tracks)
            for track in tracks:
                manager.mark_vehicle_seen(track.track_id, frame_index)

            raw = detect_tracked_plates(frame, tracks, plate_detector, frame_index)
            raw_plate_candidates += len(raw)
            stage = time.perf_counter()
            if temporal_resolver is None:
                resolution = resolve_plate_ownership_detailed(
                    raw,
                    duplicate_iou_threshold=args.plate_duplicate_iou_threshold,
                    debug=args.debug_ownership,
                )
                resolved = list(resolution.candidates)
                ownership_group_count = resolution.stats.duplicate_groups
                ownership_removed_count = resolution.stats.removed_duplicate_candidates
                temporal_resolution = None
            else:
                temporal_resolution = temporal_resolver.resolve(frame_index, raw)
                resolved = list(temporal_resolution.candidates)
                ownership_group_count = temporal_resolution.stats.conflict_groups
                ownership_removed_count = (
                    temporal_resolution.stats.removed_conflict_candidates
                )
                temporal_conflict_groups += temporal_resolution.stats.conflict_groups
                temporal_ambiguous_groups += (
                    temporal_resolution.stats.ambiguous_conflict_groups
                )
                temporal_history_updates += temporal_resolution.stats.history_updates
            ownership_seconds += time.perf_counter() - stage
            resolved_events += len(resolved)
            ownership_groups += ownership_group_count
            ownership_removed += ownership_removed_count
            ownership_invalid += (
                temporal_resolution.stats.invalid_candidates
                if temporal_resolution is not None
                else resolution.stats.invalid_candidates
            )

            legacy_resolution = None
            if args.compare_legacy_ownership and temporal_resolver is not None:
                legacy_started = time.perf_counter()
                legacy_resolution = resolve_plate_ownership_detailed(
                    raw,
                    duplicate_iou_threshold=args.plate_duplicate_iou_threshold,
                    debug=False,
                )
                legacy_compare_seconds += time.perf_counter() - legacy_started
                legacy_resolved_events += len(legacy_resolution.candidates)
                legacy_ownership_groups += legacy_resolution.stats.duplicate_groups
                legacy_ownership_removed += (
                    legacy_resolution.stats.removed_duplicate_candidates
                )
                if {
                    candidate.track_id for candidate in legacy_resolution.candidates
                } != {candidate.track_id for candidate in resolved}:
                    legacy_changed_frames += 1

            if temporal_resolution is not None and args.debug_ownership_temporal:
                temporal_debug_frames.append(
                    {
                        "frame_index": frame_index,
                        "raw_candidates": [
                            _raw_plate_candidate_payload(candidate) for candidate in raw
                        ],
                        "temporal": _temporal_resolution_payload(temporal_resolution),
                        "legacy": (
                            {
                                "selected_track_ids": [
                                    candidate.track_id
                                    for candidate in legacy_resolution.candidates
                                ],
                                "duplicate_groups": legacy_resolution.stats.duplicate_groups,
                            }
                            if legacy_resolution is not None
                            else None
                        ),
                    }
                )
            if resolved:
                frames_with_plate += 1

            current_qualities: dict[int, PlateQualityMetrics] = {}
            v4_started = time.perf_counter()
            for plate in resolved:
                crop = crop_plate_from_frame(frame, plate.plate_bbox)
                if crop is None:
                    buffer_started = time.perf_counter()
                    manager.record_invalid_plate(
                        plate.track_id, plate.frame_index, plate.plate_class_name
                    )
                    buffer_seconds += time.perf_counter() - buffer_started
                    continue
                quality_started = time.perf_counter()
                quality = score_plate_quality(crop, plate.plate_confidence, quality_config)
                quality_seconds += time.perf_counter() - quality_started
                current_qualities[plate.track_id] = quality
                sharpness_values.append(quality.sharpness_raw)
                height_values.append(float(quality.height))
                quality_values.append(quality.total_score)
                buffered = BufferedPlateCandidate(
                    frame_index=plate.frame_index,
                    track_id=plate.track_id,
                    plate_class_id=plate.plate_class_id,
                    plate_class_name=plate.plate_class_name,
                    plate_confidence=plate.plate_confidence,
                    bbox=plate.plate_bbox,
                    quality=quality,
                    crop=crop,
                )
                buffer_started = time.perf_counter()
                manager.add_plate(buffered)
                buffer_seconds += time.perf_counter() - buffer_started
            v4_seconds += time.perf_counter() - v4_started
            if temporal_resolver is not None:
                temporal_resolver.cleanup(
                    track.track_id
                    for track in tracker.all_tracks
                    if track.state.name != "REMOVED"
                )

            writer.write(draw_results(frame, tracks, resolved, current_qualities))
            processed = frame_index + 1
            if processed == 1 or processed % 30 == 0:
                print(
                    f"Frame {processed}: confirmed={len(tracks)}, plates={len(resolved)}/{len(raw)}",
                    flush=True,
                )
            if args.preview:
                cv2.imshow("V4 plate quality - press q to stop", draw_results(frame, tracks, resolved, current_qualities))
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

    summaries = manager.finalize_all()
    tracks_with_plate = sum(summary.plate_observation_count > 0 for summary in summaries)
    saved_crops = save_final_crops(manager, args.output_dir) if args.save_topk_crops else 0
    ocr_json_output: Path | None = None
    ocr_candidates = ()
    ocr_engine = None
    v5_ocr_payload: dict[str, object] | None = None
    v6_fusion_output: Path | None = None
    v6_fusion_report = None
    if args.run_ocr:
        ocr_engine = MicroCharNetOCR(
            model_path=args.ocr_model,
            conf_threshold=args.ocr_confidence,
            iou_threshold=args.ocr_iou,
            preprocess_mode=args.ocr_preprocess_mode,
        )
        ocr_candidates = run_ocr_on_topk(
            manager,
            ocr_engine,
            debug=args.debug_ocr,
        )
        ocr_json_output = args.output_dir / f"{args.input.stem}_ocr_candidates.json"
        write_ocr_json(ocr_json_output, ocr_candidates, ocr_engine)
        v5_ocr_payload = build_ocr_json(ocr_candidates, ocr_engine)
    if args.run_ocr_fusion:
        if v5_ocr_payload is None:
            raise RuntimeError("V6 requires a V5 OCR payload")
        v6_fusion_report = fuse_candidates(
            ocr_candidates,
            track_ids=(summary.track_id for summary in summaries),
            config=OCRFusionConfig(
                exact_consensus_threshold=args.exact_consensus_threshold,
                include_plate_confidence=args.include_plate_confidence,
            ),
        )
        v6_fusion_output = args.output_dir / f"{args.input.stem}_ocr_fused.json"
        v6_fusion_output.write_text(
            json.dumps(
                build_fused_json(v5_ocr_payload, v6_fusion_report),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if args.debug_ocr_fusion:
            print(format_fusion_debug(v6_fusion_report, ocr_candidates))
    track_objects = {track.track_id: track for track in tracker.all_tracks}
    track_payloads: list[dict[str, object]] = []
    for summary in summaries:
        tracked = track_objects.get(summary.track_id)
        track_payloads.append(
            {
                "track_id": summary.track_id,
                "vehicle_class_id": tracked.class_id if tracked is not None else None,
                "vehicle_class_name": tracked.class_name if tracked is not None else None,
                "vehicle_observation_count": summary.vehicle_observation_count,
                "plate_observation_count": summary.plate_observation_count,
                "valid_plate_crop_count": summary.valid_plate_crop_count,
                "invalid_plate_crop_count": summary.invalid_plate_crop_count,
                "plate_detection_ratio": round(summary.plate_detection_ratio, 6),
                "top_k_count": summary.top_k_count,
                "first_plate_frame": summary.first_plate_frame,
                "last_plate_frame": summary.last_plate_frame,
                "vuong_count": summary.vuong_count,
                "dai_count": summary.dai_count,
                "layout_votes": {name: round(value, 6) for name, value in summary.layout_votes.items()},
                "dominant_plate_class": summary.dominant_plate_class,
                "temporal_history_samples": (
                    len(temporal_resolver.history.get(summary.track_id, ()))
                    if temporal_resolver is not None
                    else 0
                ),
                "top_candidates": [
                    _candidate_payload(rank, candidate)
                    for rank, candidate in enumerate(manager.get_top_candidates(summary.track_id), 1)
                ],
            }
        )

    frame_count = frame_index
    valid_quality_events = len(quality_values)
    average_ms = lambda seconds: seconds * 1000.0 / frame_count if frame_count else 0.0
    event_ms = lambda seconds, count: seconds * 1000.0 / count if count else 0.0
    payload = {
        "status": "ok",
        "video": {"fps": round(fps, 6), "frames": frame_count, "width": width, "height": height},
        "config": {
            "quality": asdict(quality_config),
            "buffer": asdict(buffer_config),
            "ownership_duplicate_iou_threshold": args.plate_duplicate_iou_threshold,
            "ownership_mode": "v3.2_temporal"
            if temporal_resolver is not None
            else "v3.1_legacy",
            "temporal_ownership": asdict(temporal_config),
        },
        "summary": {
            "confirmed_tracks": tracker.unique_track_count,
            "confirmed_vehicle_observations": vehicle_observations,
            "raw_plate_candidates": raw_plate_candidates,
            "ownership_duplicate_groups": ownership_groups,
            "ownership_removed_candidates": ownership_removed,
            "ownership_invalid_candidates": ownership_invalid,
            "ownership_conflict_groups": temporal_conflict_groups,
            "ownership_ambiguous_conflict_groups": temporal_ambiguous_groups,
            "temporal_history_updates": temporal_history_updates,
            "legacy_resolved_plate_events": legacy_resolved_events,
            "legacy_ownership_duplicate_groups": legacy_ownership_groups,
            "legacy_ownership_removed_candidates": legacy_ownership_removed,
            "legacy_changed_owner_frames": legacy_changed_frames,
            "resolved_plate_events": resolved_events,
            "valid_plate_crops": valid_quality_events,
            "invalid_plate_crops": sum(item.invalid_plate_crop_count for item in summaries),
            "frames_with_plate": frames_with_plate,
            "tracks_with_plate": tracks_with_plate,
            "retained_topk_candidates": manager.total_retained_candidates,
            "saved_topk_crops": saved_crops,
            "v21_diagnostics": tracker.diagnostics,
            "v21_config": {
                "hard_class_gate": "off",
                "temporal_class_voting": "on",
                "cross_class_dedup_enabled": tracker.cross_class_dedup_enabled,
                "cross_class_duplicate_iou_threshold": (
                    tracker.cross_class_duplicate_iou_threshold
                ),
                "cross_class_duplicate_area_ratio_threshold": (
                    tracker.cross_class_duplicate_area_ratio_threshold
                ),
                "active_duplicate_suppression_enabled": (
                    tracker.active_duplicate_suppression_enabled
                ),
                "active_duplicate_iou_threshold": tracker.active_duplicate_iou_threshold,
                "active_duplicate_min_frames": tracker.active_duplicate_min_frames,
            },
        },
        "distributions": {
            "sharpness_raw": _distribution(sharpness_values, (10, 25, 50, 75, 90)),
            "plate_height": _distribution(height_values, (10, 50, 90)),
            "total_quality": _distribution(quality_values, (10, 50, 90)),
        },
        "performance": {
            "vehicle_ms_per_frame": round(average_ms(vehicle_seconds), 6),
            "tracking_ms_per_frame": round(average_ms(tracking_seconds), 6),
            "cross_class_dedup_ms_per_frame": round(
                tracker.cross_class_dedup_seconds * 1000.0 / frame_count
                if frame_count
                else 0.0,
                6,
            ),
            "active_duplicate_suppression_ms_per_frame": round(
                tracker.active_duplicate_suppression_seconds * 1000.0 / frame_count
                if frame_count
                else 0.0,
                6,
            ),
            "ownership_ms_per_frame": round(average_ms(ownership_seconds), 6),
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
            "legacy_compare_ms_per_frame": round(
                legacy_compare_seconds * 1000.0 / frame_count if frame_count else 0.0,
                6,
            ),
            "plate_ms_per_vehicle_roi": round(event_ms(plate_detector.total_processing_seconds, plate_detector.detect_call_count), 6),
            "ownership_ms_per_frame": round(average_ms(ownership_seconds), 6),
            "quality_ms_per_valid_event": round(event_ms(quality_seconds, valid_quality_events), 6),
            "buffer_ms_per_resolved_event": round(event_ms(buffer_seconds, resolved_events), 6),
            "v4_total_ms_per_frame": round(average_ms(v4_seconds), 6),
            "elapsed_seconds": round(elapsed_seconds, 6),
            "end_to_end_fps": round(frame_count / elapsed_seconds if elapsed_seconds else 0.0, 6),
        },
        "tracks": track_payloads,
    }
    if temporal_debug_frames:
        payload["ownership_temporal_frames"] = temporal_debug_frames
    if ocr_json_output is not None:
        payload["v5_ocr_json"] = str(ocr_json_output)
    if v6_fusion_output is not None:
        payload["v6_ocr_fused_json"] = str(v6_fusion_output)
    json_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Processed frames: {frame_count}")
    print(f"Confirmed tracks: {tracker.unique_track_count}")
    print(f"V2.1 diagnostics: {tracker.diagnostics}")
    print(
        "Ownership mode: "
        f"{'V3.2 temporal' if temporal_resolver is not None else 'V3.1 legacy'}"
    )
    print(f"Ownership conflict groups: {ownership_groups}")
    if args.compare_legacy_ownership and temporal_resolver is not None:
        print(f"Legacy resolved plate events: {legacy_resolved_events}")
        print(f"Temporal owner-changed frames: {legacy_changed_frames}")
    print(f"Resolved plate events: {resolved_events}")
    print(f"Tracks with plate: {tracks_with_plate}")
    print(f"Retained Top-K candidates: {manager.total_retained_candidates}")
    print(f"Quality: {event_ms(quality_seconds, valid_quality_events):.4f} ms/valid event")
    print(f"Buffer: {event_ms(buffer_seconds, resolved_events):.4f} ms/resolved event")
    print(f"Ownership: {average_ms(ownership_seconds):.4f} ms/frame")
    if temporal_resolver is not None:
        print(
            "Relative geometry: "
            f"{temporal_resolver.total_relative_geometry_seconds * 1000.0 / frame_count if frame_count else 0.0:.4f} ms/frame"
        )
        print(
            "History scoring: "
            f"{temporal_resolver.total_history_scoring_seconds * 1000.0 / frame_count if frame_count else 0.0:.4f} ms/frame"
        )
    print(f"V4: {average_ms(v4_seconds):.4f} ms/frame")
    print(f"End-to-end FPS: {frame_count / elapsed_seconds if elapsed_seconds else 0.0:.3f}")
    if args.save_topk_crops:
        print(f"Saved final Top-K crops: {saved_crops} under {args.output_dir / 'topk'}")
    if ocr_engine is not None and ocr_json_output is not None:
        print(f"V5 OCR retained Top-K candidates: {len(ocr_candidates)}")
        print(f"V5 OCR inference calls: {ocr_engine.inference_count}")
        print(f"V5 OCR session init count: {ocr_engine.session_init_count}")
        print(
            "V5 OCR timing ms/crop: "
            f"preprocess={ocr_engine.timing_totals['preprocess_ms_per_crop']:.3f}, "
            f"inference={ocr_engine.timing_totals['inference_ms_per_crop']:.3f}, "
            f"decode={ocr_engine.timing_totals['decode_ms_per_crop']:.3f}, "
            f"total={ocr_engine.timing_totals['total_ms_per_crop']:.3f}"
        )
        print(f"Saved V5 OCR JSON: {ocr_json_output}")
    if v6_fusion_report is not None and v6_fusion_output is not None:
        print(
            "V6 fusion: "
            f"tracks={v6_fusion_report.tracks_total}, "
            f"with_ocr={v6_fusion_report.tracks_with_ocr}, "
            f"exact={v6_fusion_report.tracks_with_exact_consensus}, "
            f"alignment={v6_fusion_report.tracks_using_alignment}, "
            f"single={v6_fusion_report.single_candidate_tracks}, "
            f"no_valid={v6_fusion_report.no_valid_ocr_tracks}"
        )
        print(f"V6 fusion time: {v6_fusion_report.total_ms:.3f} ms")
        print(f"Saved V6 OCR JSON: {v6_fusion_output}")
    print(f"Saved video: {video_output}")
    print(f"Saved JSON: {json_output}")
    return video_output, json_output


if __name__ == "__main__":
    process_video(parse_args())
