"""Module 5: plate candidates inside tracked video vehicles."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from ..config import PlateConfig, VehicleConfig
from ..plate_detector import PlateDetector
from ..vehicle_detector import VehicleDetector
from .tracking import VehicleTrackingConfig
from .tracking_runtime import TrackedVideoObservation, iter_tracked_video_frames
from .vehicle_stage import _providers_for_device
from .vehicle_validation import VehicleValidationConfig, rejection_reason_counts


BBox = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class VideoPlateCandidate:
    """One plate candidate in frame coordinates and its tracked vehicle."""

    frame_index: int
    candidate_index: int
    track_id: int
    vehicle_class_id: int
    vehicle_class_name: str
    vehicle_confidence: float
    vehicle_bbox: BBox
    plate_class_id: int
    plate_class_name: str
    plate_confidence: float
    plate_bbox: BBox
    plate_bbox_before_clamp: BBox | None = None
    coordinate_clamped: bool = False


@dataclass(frozen=True, slots=True)
class VideoPlateDecision:
    candidate: VideoPlateCandidate
    ownership_status: str
    conflict_group: int | None
    ownership_score: float


def crop_vehicle_roi(
    frame: np.ndarray,
    bbox: BBox,
) -> tuple[np.ndarray, BBox] | None:
    """Return a clamped, non-empty vehicle ROI and its frame bbox."""

    if not isinstance(frame, np.ndarray) or frame.ndim != 3:
        raise ValueError("frame must be an HxWxC NumPy array")
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = (int(value) for value in bbox)
    x1 = max(0, min(frame_width, x1))
    y1 = max(0, min(frame_height, y1))
    x2 = max(0, min(frame_width, x2))
    y2 = max(0, min(frame_height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame[y1:y2, x1:x2]
    return (roi, (x1, y1, x2, y2)) if roi.size else None


def _local_bbox_to_global(local_bbox: BBox, parent_bbox: BBox, frame: np.ndarray) -> BBox:
    """Map a plate bbox from vehicle-ROI coordinates to frame coordinates.

    The plate model is expected to return coordinates inside the ROI, but a
    malformed prediction must not escape that ROI. Clamp in local
    coordinates first, then translate. Clamping only to the full frame would
    allow a plate prediction to be assigned outside its vehicle bbox.
    """

    frame_height, frame_width = frame.shape[:2]
    parent_x1, parent_y1, parent_x2, parent_y2 = parent_bbox
    parent_x1 = max(0, min(frame_width, int(parent_x1)))
    parent_y1 = max(0, min(frame_height, int(parent_y1)))
    parent_x2 = max(parent_x1, min(frame_width, int(parent_x2)))
    parent_y2 = max(parent_y1, min(frame_height, int(parent_y2)))
    parent_width = parent_x2 - parent_x1
    parent_height = parent_y2 - parent_y1
    local_x1 = max(0, min(parent_width, int(local_bbox[0])))
    local_y1 = max(0, min(parent_height, int(local_bbox[1])))
    local_x2 = max(0, min(parent_width, int(local_bbox[2])))
    local_y2 = max(0, min(parent_height, int(local_bbox[3])))
    return (
        parent_x1 + local_x1,
        parent_y1 + local_y1,
        parent_x1 + local_x2,
        parent_y1 + local_y2,
    )


def detect_frame_plate_candidates(
    observation: TrackedVideoObservation,
    plate_detector: PlateDetector,
) -> tuple[VideoPlateCandidate, ...]:
    """Run the plate model on every accepted tracked vehicle ROI."""

    candidates: list[VideoPlateCandidate] = []
    for tracked in observation.tracked:
        cropped = crop_vehicle_roi(observation.image, tracked.detection.bbox)
        if cropped is None:
            continue
        vehicle_roi, vehicle_bbox = cropped
        for plate in plate_detector.detect(vehicle_roi):
            raw_bbox = (
                vehicle_bbox[0] + int(plate.bbox[0]),
                vehicle_bbox[1] + int(plate.bbox[1]),
                vehicle_bbox[0] + int(plate.bbox[2]),
                vehicle_bbox[1] + int(plate.bbox[3]),
            )
            plate_bbox = _local_bbox_to_global(
                plate.bbox, vehicle_bbox, observation.image,
            )
            candidates.append(
                VideoPlateCandidate(
                    frame_index=observation.frame_index,
                    candidate_index=len(candidates),
                    track_id=tracked.track_id,
                    vehicle_class_id=tracked.detection.class_id,
                    vehicle_class_name=tracked.detection.class_name,
                    vehicle_confidence=float(tracked.detection.confidence),
                    vehicle_bbox=vehicle_bbox,
                    plate_class_id=plate.class_id,
                    plate_class_name=plate.class_name,
                    plate_confidence=float(plate.confidence),
                    plate_bbox=plate_bbox,
                    plate_bbox_before_clamp=raw_bbox,
                    coordinate_clamped=raw_bbox != plate_bbox,
                )
            )
    return tuple(candidates)


def _bbox_iou(first: BBox, second: BBox) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0, min(ay2, by2) - max(ay1, by1)
    )
    first_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    second_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _overlap_over_smaller(first: BBox, second: BBox) -> float:
    intersection = max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0, min(first[3], second[3]) - max(first[1], second[1])
    )
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    smaller = min(first_area, second_area)
    return intersection / smaller if smaller else 0.0


def _ownership_score(candidate: VideoPlateCandidate) -> float:
    vehicle_width = max(1, candidate.vehicle_bbox[2] - candidate.vehicle_bbox[0])
    vehicle_height = max(1, candidate.vehicle_bbox[3] - candidate.vehicle_bbox[1])
    vehicle_center_x = (candidate.vehicle_bbox[0] + candidate.vehicle_bbox[2]) / 2.0
    plate_center_x = (candidate.plate_bbox[0] + candidate.plate_bbox[2]) / 2.0
    horizontal_offset = abs(plate_center_x - vehicle_center_x) / max(
        1.0, vehicle_width / 2.0
    )
    # A plate found near the horizontal centre of a tight vehicle ROI is
    # generally more trustworthy than the same plate found at the edge of a
    # much larger neighbouring ROI. This is important when a close vehicle's
    # detector box contains another vehicle entirely.
    center_alignment = max(0.0, 1.0 - min(1.0, horizontal_offset))
    plate_area = max(0, candidate.plate_bbox[2] - candidate.plate_bbox[0]) * max(
        0, candidate.plate_bbox[3] - candidate.plate_bbox[1]
    )
    vehicle_area = vehicle_width * vehicle_height
    tightness = min(1.0, (plate_area / vehicle_area) / 0.12)
    return float(
        0.45 * candidate.plate_confidence
        + 0.20 * candidate.vehicle_confidence
        + 0.25 * center_alignment
        + 0.10 * tightness
    )


def resolve_frame_plate_ownership(
    candidates: Sequence[VideoPlateCandidate],
    *,
    conflict_iou_threshold: float = 0.70,
    overlap_over_smaller_threshold: float = 0.85,
) -> tuple[VideoPlateDecision, ...]:
    """Mark cross-track plate conflicts without discarding raw candidates.

    Preserve every raw candidate in the decisions, but accept at most one
    plate per track and frame after resolving cross-track conflicts.
    """

    if not 0.0 < conflict_iou_threshold <= 1.0:
        raise ValueError("conflict_iou_threshold must be in (0, 1]")
    if not 0.0 < overlap_over_smaller_threshold <= 1.0:
        raise ValueError("overlap_over_smaller_threshold must be in (0, 1]")

    if not candidates:
        return ()
    parents = list(range(len(candidates)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    valid_indices = {
        index for index, candidate in enumerate(candidates)
        if not candidate.coordinate_clamped
        and candidate.plate_bbox[2] > candidate.plate_bbox[0]
        and candidate.plate_bbox[3] > candidate.plate_bbox[1]
        and candidate.vehicle_bbox[0] <= candidate.plate_bbox[0]
        and candidate.vehicle_bbox[1] <= candidate.plate_bbox[1]
        and candidate.plate_bbox[2] <= candidate.vehicle_bbox[2]
        and candidate.plate_bbox[3] <= candidate.vehicle_bbox[3]
    }
    for first in range(len(candidates)):
        for second in range(first + 1, len(candidates)):
            if first not in valid_indices or second not in valid_indices:
                continue
            if candidates[first].track_id == candidates[second].track_id:
                continue
            if (
                _bbox_iou(candidates[first].plate_bbox, candidates[second].plate_bbox)
                >= conflict_iou_threshold
                or _overlap_over_smaller(
                    candidates[first].plate_bbox, candidates[second].plate_bbox
                )
                >= overlap_over_smaller_threshold
            ):
                union(first, second)

    groups: dict[int, list[int]] = {}
    for index in range(len(candidates)):
        groups.setdefault(find(index), []).append(index)

    decisions: list[VideoPlateDecision | None] = [None] * len(candidates)
    conflict_group = 0
    for group in groups.values():
        if group[0] not in valid_indices:
            index = group[0]
            decisions[index] = VideoPlateDecision(
                candidates[index], "rejected_outside_vehicle", None,
                _ownership_score(candidates[index]),
            )
            continue
        track_ids = {candidates[index].track_id for index in group}
        if len(track_ids) == 1:
            for index in group:
                decisions[index] = VideoPlateDecision(
                    candidate=candidates[index],
                    ownership_status="selected",
                    conflict_group=None,
                    ownership_score=_ownership_score(candidates[index]),
                )
            continue

        best_index = max(
            group,
            key=lambda index: (
                _ownership_score(candidates[index]),
                candidates[index].plate_confidence,
                candidates[index].vehicle_confidence,
                -candidates[index].track_id,
            ),
        )
        owner_track_id = candidates[best_index].track_id
        for index in group:
            candidate = candidates[index]
            decisions[index] = VideoPlateDecision(
                candidate=candidate,
                ownership_status=(
                    "selected" if candidate.track_id == owner_track_id else "rejected_conflict"
                ),
                conflict_group=conflict_group,
                ownership_score=_ownership_score(candidate),
            )
        conflict_group += 1

    selected_by_track: dict[int, list[int]] = {}
    for index, item in enumerate(decisions):
        if item is not None and item.ownership_status == "selected":
            selected_by_track.setdefault(item.candidate.track_id, []).append(index)
    for indices in selected_by_track.values():
        best = max(
            indices,
            key=lambda index: (
                decisions[index].ownership_score,
                candidates[index].plate_confidence,
                -candidates[index].candidate_index,
            ),
        )
        for index in indices:
            if index != best:
                item = decisions[index]
                decisions[index] = VideoPlateDecision(
                    item.candidate, "rejected_same_track", item.conflict_group,
                    item.ownership_score,
                )
    return tuple(item for item in decisions if item is not None)


def _draw_observation(
    frame: np.ndarray,
    observation: TrackedVideoObservation,
    decisions: Sequence[VideoPlateDecision],
) -> None:
    for item in observation.tracked:
        x1, y1, x2, y2 = item.detection.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        label = f"ID {item.track_id} {item.detection.class_name} {item.detection.confidence:.2f}"
        cv2.putText(
            frame, label, (max(0, x1), max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2, cv2.LINE_AA,
        )
    for item in decisions:
        if item.ownership_status != "selected":
            continue
        x1, y1, x2, y2 = item.candidate.plate_bbox
        color = (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = (
            f"P{item.candidate.candidate_index} ID{item.candidate.track_id} "
            f"{item.candidate.plate_class_name} {item.candidate.plate_confidence:.2f}"
        )
        cv2.putText(
            frame, label, (max(0, x1), max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA,
        )


def _vehicle_json(observation: TrackedVideoObservation) -> list[dict[str, object]]:
    return [
        {
            "vehicle_index": index,
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
        for index, item in enumerate(observation.tracked)
    ]


def _plate_json(decision: VideoPlateDecision) -> dict[str, object]:
    candidate = decision.candidate
    return {
        "candidate_index": candidate.candidate_index,
        "track_id": candidate.track_id,
        "vehicle_class_id": candidate.vehicle_class_id,
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
        "conflict_group": decision.conflict_group,
        "ownership_score": round(decision.ownership_score, 6),
    }


def run_video_plate_detection(
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
    save_annotated: bool = True,
    max_frames: int | None = None,
) -> dict[str, object]:
    """Detect vehicles, track them, and inspect all plate candidates."""

    source_path = Path(source)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / f"{source_path.stem}_plate_detection.json"
    annotated_path = output_path / f"{source_path.stem}_plate_detection_annotated.mp4"

    vehicle_defaults = VehicleConfig()
    plate_defaults = PlateConfig()
    if vehicle_detector is None:
        vehicle_detector = VehicleDetector(
            Path(vehicle_model_path) if vehicle_model_path is not None else vehicle_defaults.model,
            confidence_threshold=(
                vehicle_defaults.confidence
                if vehicle_confidence is None
                else vehicle_confidence
            ),
            iou_threshold=vehicle_defaults.iou if vehicle_iou is None else vehicle_iou,
            providers=_providers_for_device(device),
        )
    if plate_detector is None:
        plate_detector = PlateDetector(
            Path(plate_model_path) if plate_model_path is not None else plate_defaults.model,
            confidence_threshold=(
                plate_defaults.confidence
                if plate_confidence is None
                else plate_confidence
            ),
            iou_threshold=plate_defaults.iou if plate_iou is None else plate_iou,
            providers=_providers_for_device(device),
        )

    records: list[dict[str, object]] = []
    raw_vehicle_detections = 0
    accepted_vehicle_detections = 0
    rejected_vehicle_detections = 0
    rejection_reasons: Counter[str] = Counter()
    raw_plate_candidates = 0
    selected_plate_candidates = 0
    rejected_plate_candidates = 0
    frames_with_plate_candidates = 0
    frames_read = 0
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

            validated = observation.validated
            decisions = resolve_frame_plate_ownership(
                detect_frame_plate_candidates(observation, plate_detector)
            )
            plate_json = [_plate_json(item) for item in decisions]
            selected_json = [
                item for item in plate_json if item["ownership_status"] == "selected"
            ]
            raw_vehicle_detections += len(validated)
            accepted_vehicle_detections += len(observation.accepted)
            rejected_vehicle_detections += len(observation.rejected)
            rejection_reasons.update(rejection_reason_counts(validated))
            raw_plate_candidates += len(decisions)
            selected_plate_candidates += len(selected_json)
            rejected_plate_candidates += len(decisions) - len(selected_json)
            frames_with_plate_candidates += bool(decisions)
            records.append(
                {
                    "frame_index": observation.frame_index,
                    "timestamp_seconds": round(observation.timestamp_seconds, 6),
                    "vehicles": _vehicle_json(observation),
                    "plate_candidates": plate_json,
                    "owned_plate_candidates": selected_json,
                    "validation": {
                        "raw_count": len(validated),
                        "accepted_count": len(observation.accepted),
                        "rejected_count": len(observation.rejected),
                        "rejection_reasons": rejection_reason_counts(validated),
                    },
                    "tracking": {
                        "active_track_ids": list(observation.active_track_ids),
                        "lost_track_ids": list(observation.lost_track_ids),
                        "reidentified_count": observation.reidentified_count,
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

    elapsed_seconds = (cv2.getTickCount() - started) / cv2.getTickFrequency()
    selected_validation = validation_config or VehicleValidationConfig()
    payload: dict[str, object] = {
        "status": "ok",
        "input": {
            "path": str(source_path),
            "type": "video",
            "fps": round(last_observation.fps, 6),
            "frame_count_metadata": None,
            "width": last_observation.width,
            "height": last_observation.height,
        },
        "summary": {
            "frames_read": frames_read,
            "frames_with_plate_candidates": int(frames_with_plate_candidates),
            "raw_vehicle_detections": raw_vehicle_detections,
            "accepted_vehicle_detections": accepted_vehicle_detections,
            "rejected_vehicle_detections": rejected_vehicle_detections,
            "raw_plate_candidates": raw_plate_candidates,
            "selected_plate_candidates": selected_plate_candidates,
            "rejected_plate_candidates": rejected_plate_candidates,
            "unique_track_ids_created": last_observation.created_track_count,
            "reidentified_count": last_observation.reidentified_count,
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
        },
        "frames": records,
        "performance": {
            "total_seconds": round(elapsed_seconds, 6),
            "ms_per_frame": round(elapsed_seconds * 1000.0 / frames_read, 6),
            "vehicle_detector_calls": int(getattr(vehicle_detector, "detect_call_count", frames_read)),
            "plate_detector_calls": int(getattr(plate_detector, "detect_call_count", 0)),
            "vehicle_model_path": str(getattr(vehicle_detector, "model_path", vehicle_model_path or "injected")),
            "plate_model_path": str(getattr(plate_detector, "model_path", plate_model_path or "injected")),
            "vehicle_validation_min_confidence": selected_validation.min_confidence,
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


__all__ = [
    "VideoPlateCandidate",
    "VideoPlateDecision",
    "crop_vehicle_roi",
    "detect_frame_plate_candidates",
    "resolve_frame_plate_ownership",
    "run_video_plate_detection",
]
