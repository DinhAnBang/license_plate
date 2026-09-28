"""Module 4: stable vehicle IDs over validated video detections."""

from __future__ import annotations

import json
import csv
from collections import Counter
from pathlib import Path
from typing import Sequence

import cv2

from ..config import VehicleConfig
from ..vehicle_detector import VehicleDetector
from .tracking import TrackedVehicle, VehicleTrackingConfig
from .tracking_runtime import TrackedVideoObservation, iter_tracked_video_frames
from .vehicle_stage import _providers_for_device
from .vehicle_validation import (
    ValidatedVehicleDetection,
    VehicleValidationConfig,
    rejection_reason_counts,
)


def _draw_tracked_vehicles(frame, tracked: Sequence[TrackedVehicle]) -> None:
    for item in tracked:
        x1, y1, x2, y2 = item.detection.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        label = (
            f"ID {item.track_id} {item.detection.class_name} "
            f"{item.detection.confidence:.2f}"
        )
        cv2.putText(
            frame,
            label,
            (max(0, x1), max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 0),
            2,
            cv2.LINE_AA,
        )


def _detection_to_json(
    vehicle_index: int,
    item: ValidatedVehicleDetection,
) -> dict[str, object]:
    detection = item.detection
    return {
        "vehicle_index": vehicle_index,
        "class_id": detection.class_id,
        "class_name": detection.class_name,
        "confidence": round(float(detection.confidence), 6),
        "bbox_xyxy": list(detection.bbox),
        "accepted": item.accepted,
        "rejection_reasons": list(item.reasons),
    }


def _tracked_to_json(
    item: TrackedVehicle,
    *,
    vehicle_index: int,
) -> dict[str, object]:
    return {
        "vehicle_index": vehicle_index,
        "track_id": item.track_id,
        "class_id": item.detection.class_id,
        "class_name": item.detection.class_name,
        "confidence": round(float(item.detection.confidence), 6),
        "bbox_xyxy": list(item.detection.bbox),
        "track_status": item.status,
        "track_age_frames": item.age_frames,
        "track_hits": item.hits,
        "reidentified": item.reidentified,
    }


def run_video_vehicle_tracking(
    source: str | Path,
    output_dir: str | Path,
    *,
    model_path: str | Path | None = None,
    confidence: float | None = None,
    iou: float | None = None,
    device: str = "auto",
    detector: VehicleDetector | None = None,
    validation_config: VehicleValidationConfig | None = None,
    tracking_config: VehicleTrackingConfig | None = None,
    save_annotated: bool = True,
    max_frames: int | None = None,
) -> dict[str, object]:
    """Detect, validate, and assign stable IDs to vehicles in a video."""

    source_path = Path(source)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / f"{source_path.stem}_tracking.json"
    annotated_path = output_path / f"{source_path.stem}_tracking_annotated.mp4"
    timeline_path = output_path / f"{source_path.stem}_tracking_timeline.csv"

    defaults = VehicleConfig()
    selected_validation = validation_config or VehicleValidationConfig()
    if detector is None:
        selected_model = Path(model_path) if model_path is not None else defaults.model
        selected_confidence = defaults.confidence if confidence is None else confidence
        selected_iou = defaults.iou if iou is None else iou
        detector = VehicleDetector(
            selected_model,
            confidence_threshold=selected_confidence,
            iou_threshold=selected_iou,
            providers=_providers_for_device(device),
        )

    records: list[dict[str, object]] = []
    class_counts: Counter[str] = Counter()
    frames_read = 0
    raw_frames_with_detections = 0
    frames_with_detections = 0
    raw_detections_count = 0
    accepted_detections_count = 0
    rejected_detections_count = 0
    rejection_reasons: Counter[str] = Counter()
    writer: cv2.VideoWriter | None = None
    started = cv2.getTickCount()
    last_observation: TrackedVideoObservation | None = None

    try:
        for observation in iter_tracked_video_frames(
            source_path,
            detector=detector,
            validation_config=selected_validation,
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
            validated = observation.validated
            accepted = observation.accepted
            rejected = observation.rejected
            tracked = observation.tracked
            raw_frames_with_detections += bool(validated)
            frames_with_detections += bool(accepted)
            raw_detections_count += len(validated)
            accepted_detections_count += len(accepted)
            rejected_detections_count += len(rejected)
            rejection_reasons.update(rejection_reason_counts(validated))
            class_counts.update(item.detection.class_name for item in accepted)
            raw_json = [_detection_to_json(index, item) for index, item in enumerate(validated)]
            accepted_json: list[dict[str, object]] = []
            accepted_index = 0
            for index, item in enumerate(validated):
                if item.accepted:
                    accepted_json.append(
                        _tracked_to_json(tracked[accepted_index], vehicle_index=index)
                    )
                    accepted_index += 1
            records.append(
                {
                    "frame_index": observation.frame_index,
                    "timestamp_seconds": round(observation.timestamp_seconds, 6),
                    "raw_detections": raw_json,
                    "accepted_detections": accepted_json,
                    "detections": accepted_json,
                    "validation": {
                        "raw_count": len(validated),
                        "accepted_count": len(accepted),
                        "rejected_count": len(rejected),
                        "rejection_reasons": rejection_reason_counts(validated),
                    },
                    "tracking": {
                        "active_track_ids": list(observation.active_track_ids),
                        "lost_track_ids": list(observation.lost_track_ids),
                        "removed_track_ids": list(observation.removed_track_ids),
                        "reidentified_count": observation.reidentified_count,
                        "frame_stats": observation.tracking_frame_stats,
                    },
                }
            )
            if writer is not None:
                annotated = observation.image.copy()
                _draw_tracked_vehicles(annotated, tracked)
                writer.write(annotated)
            frames_read += 1
    finally:
        if writer is not None:
            writer.release()

    if frames_read == 0:
        raise ValueError(f"Video contains no decodable frames: {source_path}")
    assert last_observation is not None

    elapsed_seconds = (cv2.getTickCount() - started) / cv2.getTickFrequency()
    payload: dict[str, object] = {
        "status": "ok",
        "input": {
            "path": str(source_path),
            "type": "video",
            "fps": round(last_observation.fps, 6),
            "frame_count_metadata": last_observation.frame_count_metadata,
            "width": last_observation.width,
            "height": last_observation.height,
        },
        "summary": {
            "frames_read": frames_read,
            "raw_frames_with_detections": int(raw_frames_with_detections),
            "frames_with_detections": int(frames_with_detections),
            "raw_vehicle_detections": raw_detections_count,
            "accepted_vehicle_detections": accepted_detections_count,
            "rejected_vehicle_detections": rejected_detections_count,
            "unique_track_ids_created": last_observation.created_track_count,
            "reidentified_count": last_observation.reidentified_count,
            "active_track_ids_at_end": list(last_observation.active_track_ids),
            "tracking_counters": last_observation.tracking_counters,
            "class_counts": dict(sorted(class_counts.items())),
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
        },
        "frames": records,
        "track_timeline": [
            last_observation.track_history[id_]
            for id_ in sorted(last_observation.track_history)
        ],
        "performance": {
            "total_seconds": round(elapsed_seconds, 6),
            "ms_per_frame": round(elapsed_seconds * 1000.0 / frames_read, 6),
            "detector_calls": int(getattr(detector, "detect_call_count", frames_read)),
            "model_path": str(getattr(detector, "model_path", model_path or "injected")),
            "detector_confidence": float(
                getattr(
                    detector,
                    "confidence_threshold",
                    confidence if confidence is not None else defaults.confidence,
                )
            ),
            "validation_min_confidence": selected_validation.min_confidence,
            "tracking_low_confidence": (
                tracking_config or VehicleTrackingConfig()
            ).low_confidence,
        },
    }
    if save_annotated:
        payload["annotated_path"] = str(annotated_path)
    with timeline_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer_csv = csv.DictWriter(
            handle,
            fieldnames=("track_id", "first_frame", "last_frame", "hits", "events", "removal_reason"),
        )
        writer_csv.writeheader()
        for item in payload["track_timeline"]:
            writer_csv.writerow({
                "track_id": item["track_id"],
                "first_frame": item["first_frame"],
                "last_frame": item["last_frame"],
                "hits": item["hits"],
                "events": "; ".join(
                    f"{event['frame_index']}:{event['status']}"
                    for event in item["events"]
                ),
                "removal_reason": item["removal_reason"] or "",
            })
    payload["timeline_path"] = str(timeline_path)
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    payload["output_path"] = str(json_path)
    return payload


__all__ = ["run_video_vehicle_tracking"]
