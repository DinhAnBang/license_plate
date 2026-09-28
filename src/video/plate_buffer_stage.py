"""Module 6: score plate crops and save rebuildable Top-K evidence."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import cv2

from ..config import PlateConfig, VehicleConfig
from ..plate_detector import PlateDetector
from ..vehicle_detector import VehicleDetector
from .plate_buffer import (
    VideoBufferedPlate,
    VideoPlateBuffer,
    VideoPlateBufferConfig,
)
from .plate_quality import (
    VideoPlateQualityConfig,
    VideoPlateQualityMetrics,
    crop_plate_from_frame,
    score_plate_quality,
)
from .plate_stage import (
    VideoPlateDecision,
    _draw_observation,
    detect_frame_plate_candidates,
    resolve_frame_plate_ownership,
)
from .tracking import VehicleTrackingConfig
from .tracking_runtime import TrackedVideoObservation, iter_tracked_video_frames
from .vehicle_stage import _providers_for_device
from .vehicle_validation import VehicleValidationConfig, rejection_reason_counts


def _quality_json(metrics: VideoPlateQualityMetrics | None) -> dict[str, object] | None:
    if metrics is None:
        return None
    return {
        "width": metrics.width,
        "height": metrics.height,
        "plate_confidence": round(metrics.plate_confidence, 6),
        "sharpness_raw": round(metrics.sharpness_raw, 6),
        "sharpness_score": round(metrics.sharpness_score, 6),
        "size_score": round(metrics.size_score, 6),
        "exposure_score": round(metrics.exposure_score, 6),
        "total_score": round(metrics.total_score, 6),
    }


def _candidate_json(
    decision: VideoPlateDecision,
    metrics: VideoPlateQualityMetrics | None,
    *,
    crop_valid: bool,
    buffer_status: str,
    buffer_rejection_reason: str | None,
) -> dict[str, object]:
    candidate = decision.candidate
    return {
        "candidate_index": candidate.candidate_index,
        "track_id": candidate.track_id,
        "vehicle_class_name": candidate.vehicle_class_name,
        "vehicle_confidence": round(candidate.vehicle_confidence, 6),
        "vehicle_bbox_xyxy": list(candidate.vehicle_bbox),
        "plate_class_id": candidate.plate_class_id,
        "plate_class_name": candidate.plate_class_name,
        "plate_confidence": round(candidate.plate_confidence, 6),
        "plate_bbox_xyxy": list(candidate.plate_bbox),
        "plate_bbox_before_clamp_xyxy": list(candidate.plate_bbox_before_clamp or candidate.plate_bbox),
        "coordinate_clamped": candidate.coordinate_clamped,
        "ownership_status": decision.ownership_status,
        "ownership_score": round(decision.ownership_score, 6),
        "conflict_group": decision.conflict_group,
        "crop_valid": crop_valid,
        "quality": _quality_json(metrics),
        "buffer_status": buffer_status,
        "buffer_rejection_reason": buffer_rejection_reason,
        "stages": [
            "DETECTED",
            "OWNERSHIP_ACCEPTED" if decision.ownership_status == "selected" else "OWNERSHIP_REJECTED",
            "CROP_VALID" if crop_valid else "CROP_INVALID",
            "QUALITY_ACCEPTED" if buffer_status == "eligible" else "QUALITY_REJECTED",
        ],
    }


def _write_final_crops(
    output_dir: Path,
    buffer: VideoPlateBuffer,
) -> list[dict[str, object]]:
    tracks: list[dict[str, object]] = []
    crops_root = output_dir / "crops"
    crops_root.mkdir(parents=True, exist_ok=True)
    for track_id in buffer.track_ids:
        selected = buffer.selected(track_id)
        track_dir = crops_root / f"track_{track_id}"
        track_dir.mkdir(parents=True, exist_ok=True)
        crop_records: list[dict[str, object]] = []
        for rank, item in enumerate(selected, start=1):
            frame_index = item.candidate.frame_index
            crop_path = track_dir / f"top{rank}_frame{frame_index:06d}.jpg"
            if not cv2.imwrite(str(crop_path), item.crop):
                raise OSError(f"Could not write plate crop: {crop_path}")
            crop_records.append(
                {
                    "rank": rank,
                    "frame_index": frame_index,
                    "candidate_index": item.candidate.candidate_index,
                    "plate_bbox_xyxy": list(item.candidate.plate_bbox),
                    "plate_class_name": item.candidate.plate_class_name,
                    "plate_confidence": round(item.candidate.plate_confidence, 6),
                    "quality": _quality_json(item.quality),
                    "path": str(crop_path),
                }
            )
        tracks.append(
            {
                "track_id": track_id,
                "all_evidence_count": len(buffer.all_candidates(track_id)),
                "eligible_evidence_count": buffer.eligible_count(track_id),
                "selected_count": len(selected),
                "top_k": crop_records,
            }
        )
    return tracks


def run_video_plate_buffer(
    source: str | Path,
    output_dir: str | Path,
    *,
    vehicle_model_path: str | Path | None = None,
    plate_model_path: str | Path | None = None,
    vehicle_confidence: float | None = None,
    vehicle_iou: float | None = None,
    plate_confidence: float | None = None,
    plate_iou: float | None = None,
    device: str = "auto",
    vehicle_detector: VehicleDetector | None = None,
    plate_detector: PlateDetector | None = None,
    validation_config: VehicleValidationConfig | None = None,
    tracking_config: VehicleTrackingConfig | None = None,
    quality_config: VideoPlateQualityConfig | None = None,
    buffer_config: VideoPlateBufferConfig | None = None,
    save_annotated: bool = True,
    max_frames: int | None = None,
) -> dict[str, object]:
    """Run vehicle/plate detection and save final Top-K crops per track."""

    source_path = Path(source)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / f"{source_path.stem}_plate_topk.json"
    annotated_path = output_path / f"{source_path.stem}_plate_topk_annotated.mp4"

    vehicle_defaults = VehicleConfig()
    plate_defaults = PlateConfig()
    if vehicle_detector is None:
        vehicle_detector = VehicleDetector(
            Path(vehicle_model_path) if vehicle_model_path is not None else vehicle_defaults.model,
            confidence_threshold=(vehicle_defaults.confidence if vehicle_confidence is None else vehicle_confidence),
            iou_threshold=vehicle_defaults.iou if vehicle_iou is None else vehicle_iou,
            providers=_providers_for_device(device),
        )
    if plate_detector is None:
        plate_detector = PlateDetector(
            Path(plate_model_path) if plate_model_path is not None else plate_defaults.model,
            confidence_threshold=(plate_defaults.confidence if plate_confidence is None else plate_confidence),
            iou_threshold=plate_defaults.iou if plate_iou is None else plate_iou,
            providers=_providers_for_device(device),
        )

    selected_quality = quality_config or VideoPlateQualityConfig()
    buffer = VideoPlateBuffer(buffer_config)
    records: list[dict[str, object]] = []
    rejection_reasons: Counter[str] = Counter()
    raw_vehicle_detections = 0
    accepted_vehicle_detections = 0
    rejected_vehicle_detections = 0
    raw_plate_candidates = 0
    ownership_rejected_candidates = 0
    quality_rejected_candidates = 0
    valid_crops = 0
    frames_read = 0
    frames_with_candidates = 0
    writer: cv2.VideoWriter | None = None
    started = cv2.getTickCount()
    last_observation: TrackedVideoObservation | None = None

    try:
        for observation in iter_tracked_video_frames(
            source_path,
            detector=vehicle_detector,
            validation_config=validation_config,
            tracking_config=tracking_config,
            max_frames=max_frames,
        ):
            last_observation = observation
            if writer is None and save_annotated:
                writer = cv2.VideoWriter(
                    str(annotated_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    observation.fps,
                    (observation.width, observation.height),
                )
                if not writer.isOpened():
                    raise OSError(f"Could not create annotated video: {annotated_path}")

            decisions = resolve_frame_plate_ownership(
                detect_frame_plate_candidates(observation, plate_detector)
            )
            frame_candidates: list[dict[str, object]] = []
            for decision in decisions:
                crop = crop_plate_from_frame(
                    observation.image, decision.candidate.plate_bbox,
                )
                metrics = (
                    score_plate_quality(
                        crop, decision.candidate.plate_confidence, selected_quality,
                    )
                    if crop is not None
                    else None
                )
                if crop is not None:
                    valid_crops += 1
                buffer_status = "not_buffered"
                buffer_rejection_reason: str | None = None
                if decision.ownership_status == "selected" and crop is not None and metrics is not None:
                    buffered = VideoBufferedPlate(decision.candidate, metrics, crop)
                    buffer_rejection_reason = buffer.rejection_reason(buffered)
                    buffer.add(buffered)
                    if buffer_rejection_reason is None:
                        buffer_status = "eligible"
                    else:
                        buffer_status = "rejected_quality"
                        quality_rejected_candidates += 1
                elif decision.ownership_status == "selected" and crop is None:
                    buffer_rejection_reason = "invalid_crop"
                elif decision.ownership_status != "selected":
                    buffer_rejection_reason = decision.ownership_status
                if decision.ownership_status != "selected":
                    ownership_rejected_candidates += 1
                frame_candidates.append(
                    _candidate_json(
                        decision,
                        metrics,
                        crop_valid=crop is not None,
                        buffer_status=buffer_status,
                        buffer_rejection_reason=buffer_rejection_reason,
                    )
                )

            raw_vehicle_detections += len(observation.validated)
            accepted_vehicle_detections += len(observation.accepted)
            rejected_vehicle_detections += len(observation.rejected)
            rejection_reasons.update(rejection_reason_counts(observation.validated))
            raw_plate_candidates += len(decisions)
            frames_with_candidates += bool(decisions)
            records.append(
                {
                    "frame_index": observation.frame_index,
                    "timestamp_seconds": round(observation.timestamp_seconds, 6),
                    "vehicles": [
                        {
                            "track_id": item.track_id,
                            "class_name": item.detection.class_name,
                            "confidence": round(float(item.detection.confidence), 6),
                            "bbox_xyxy": list(item.detection.bbox),
                        }
                        for item in observation.tracked
                    ],
                    "plate_candidates": frame_candidates,
                    "validation": {
                        "raw_count": len(observation.validated),
                        "accepted_count": len(observation.accepted),
                        "rejected_count": len(observation.rejected),
                        "rejection_reasons": rejection_reason_counts(observation.validated),
                    },
                }
            )
            if writer is not None:
                annotated = observation.image.copy()
                _draw_observation(annotated, observation, decisions)
                writer.write(annotated)
            frames_read += 1
    finally:
        if writer is not None:
            writer.release()

    if last_observation is None or frames_read == 0:
        raise ValueError(f"Video contains no decodable frames: {source_path}")

    final_tracks = _write_final_crops(output_path, buffer)
    retained = {
        (item.candidate.track_id, item.candidate.frame_index, item.candidate.candidate_index)
        for track_id in buffer.track_ids for item in buffer.selected(track_id)
    }
    retained_frames = {
        track_id: [item.candidate.frame_index for item in buffer.selected(track_id)]
        for track_id in buffer.track_ids
    }
    for frame in records:
        for candidate in frame["plate_candidates"]:
            key = (candidate["track_id"], frame["frame_index"], candidate["candidate_index"])
            is_retained = key in retained
            candidate["topk_status"] = "retained" if is_retained else "not_retained"
            candidate["stages"].append("TOPK_RETAINED" if is_retained else "TOPK_NOT_RETAINED")
            if is_retained:
                candidate["final_rejection_reason"] = None
            elif candidate["buffer_rejection_reason"] is not None:
                candidate["final_rejection_reason"] = candidate["buffer_rejection_reason"]
            elif candidate["buffer_status"] == "eligible":
                gap = buffer.config.min_frame_gap
                candidate["final_rejection_reason"] = (
                    "temporal_gap" if any(
                        abs(frame["frame_index"] - index) < gap
                        for index in retained_frames.get(candidate["track_id"], ())
                    ) else "lower_rank"
                )
            else:
                candidate["final_rejection_reason"] = "not_buffered"
    elapsed_seconds = (cv2.getTickCount() - started) / cv2.getTickFrequency()
    selected_validation = validation_config or VehicleValidationConfig()
    payload: dict[str, object] = {
        "status": "ok",
        "input": {
            "path": str(source_path),
            "type": "video",
            "fps": round(last_observation.fps, 6),
            "width": last_observation.width,
            "height": last_observation.height,
        },
        "summary": {
            "frames_read": frames_read,
            "frames_with_plate_candidates": int(frames_with_candidates),
            "raw_vehicle_detections": raw_vehicle_detections,
            "accepted_vehicle_detections": accepted_vehicle_detections,
            "rejected_vehicle_detections": rejected_vehicle_detections,
            "raw_plate_candidates": raw_plate_candidates,
            "valid_plate_crops": valid_crops,
            "ownership_rejected_candidates": ownership_rejected_candidates,
            "quality_rejected_candidates": quality_rejected_candidates,
            "unique_track_ids_created": last_observation.created_track_count,
            "reidentified_count": last_observation.reidentified_count,
            "track_count_with_evidence": len(final_tracks),
            "top_k": (buffer_config or VideoPlateBufferConfig()).top_k,
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
        },
        "tracks": final_tracks,
        "frames": records,
        "performance": {
            "total_seconds": round(elapsed_seconds, 6),
            "ms_per_frame": round(elapsed_seconds * 1000.0 / frames_read, 6),
            "vehicle_detector_calls": int(getattr(vehicle_detector, "detect_call_count", frames_read)),
            "plate_detector_calls": int(getattr(plate_detector, "detect_call_count", 0)),
            "vehicle_validation_min_confidence": selected_validation.min_confidence,
            "tracking_max_lost_seconds": (
                tracking_config or VehicleTrackingConfig()
            ).max_lost_seconds,
            "plate_detector_confidence": float(
                getattr(
                    plate_detector,
                    "confidence_threshold",
                    plate_confidence if plate_confidence is not None else plate_defaults.confidence,
                )
            ),
        },
    }
    if save_annotated:
        payload["annotated_path"] = str(annotated_path)
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    payload["output_path"] = str(json_path)
    return payload


__all__ = ["run_video_plate_buffer"]
