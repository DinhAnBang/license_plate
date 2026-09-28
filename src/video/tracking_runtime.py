"""Reusable video detection, validation, and tracking runtime."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace

import numpy as np

from ..vehicle_detector import VehicleDetector
from .source import VideoReader
from .tracking import (
    TrackedVehicle,
    VehicleTracker,
    VehicleTrackingConfig,
)
from .vehicle_stage import detect_frame_vehicles
from .vehicle_validation import (
    ValidatedVehicleDetection,
    VehicleValidationConfig,
    validate_frame_detections,
)


@dataclass(frozen=True, slots=True)
class TrackedVideoObservation:
    """One decoded frame after vehicle validation and tracking."""

    frame_index: int
    timestamp_seconds: float
    image: np.ndarray
    fps: float
    frame_count_metadata: int
    width: int
    height: int
    validated: tuple[ValidatedVehicleDetection, ...]
    accepted: tuple[ValidatedVehicleDetection, ...]
    rejected: tuple[ValidatedVehicleDetection, ...]
    tracked: tuple[TrackedVehicle, ...]
    active_track_ids: tuple[int, ...]
    lost_track_ids: tuple[int, ...]
    created_track_count: int
    reidentified_count: int
    removed_track_ids: tuple[int, ...]
    tracking_frame_stats: dict[str, int]
    tracking_counters: dict[str, int]
    track_history: dict[int, dict[str, object]]


def iter_tracked_video_frames(
    source: str,
    *,
    detector: VehicleDetector,
    validation_config: VehicleValidationConfig | None = None,
    tracking_config: VehicleTrackingConfig | None = None,
    max_frames: int | None = None,
) -> Iterator[TrackedVideoObservation]:
    """Yield one shared vehicle-tracking result per video frame."""

    selected_validation = validation_config or VehicleValidationConfig()
    tracking_defaults = VehicleTrackingConfig()
    selected_tracking = tracking_config or replace(
        tracking_defaults,
        high_confidence=selected_validation.min_confidence,
        new_track_confidence=max(
            tracking_defaults.new_track_confidence,
            selected_validation.min_confidence,
        ),
    )
    if selected_tracking.high_confidence != selected_validation.min_confidence:
        raise ValueError("validation min_confidence must equal tracking high_confidence")
    eligibility_validation = replace(
        selected_validation,
        min_confidence=selected_tracking.low_confidence,
        wide_edge_min_confidence=selected_tracking.low_confidence,
    )
    with VideoReader(source) as reader:
        metadata = reader.metadata
        tracker = VehicleTracker(fps=metadata.fps, config=selected_tracking)
        frames_read = 0
        for video_frame in reader:
            if max_frames is not None and frames_read >= max_frames:
                break
            detections = detect_frame_vehicles(video_frame.image, detector)
            validated = validate_frame_detections(
                detections,
                frame_width=metadata.width,
                frame_height=metadata.height,
                config=eligibility_validation,
            )
            eligible_indices = [index for index, item in enumerate(validated) if item.accepted]
            eligible = tuple(validated[index] for index in eligible_indices)
            tracked = tracker.update(
                tuple(item.detection for item in eligible),
                frame_index=video_frame.frame_index,
            )
            matched_indices = {eligible_indices[index] for index in tracker.last_detection_indices}
            validated = tuple(
                item if not item.accepted or index in matched_indices else
                replace(item, accepted=False, reasons=("unmatched_low_confidence",))
                for index, item in enumerate(validated)
            )
            accepted = tuple(item for item in validated if item.accepted)
            rejected = tuple(item for item in validated if not item.accepted)
            yield TrackedVideoObservation(
                frame_index=video_frame.frame_index,
                timestamp_seconds=video_frame.timestamp_seconds,
                image=video_frame.image,
                fps=metadata.fps,
                frame_count_metadata=metadata.frame_count,
                width=metadata.width,
                height=metadata.height,
                validated=validated,
                accepted=accepted,
                rejected=rejected,
                tracked=tracked,
                active_track_ids=tracker.active_track_ids,
                lost_track_ids=tracker.lost_track_ids,
                created_track_count=tracker.created_track_count,
                reidentified_count=tracker.reidentified_count,
                removed_track_ids=tracker.removed_track_ids,
                tracking_frame_stats=dict(tracker.last_frame_stats),
                tracking_counters=dict(tracker.counters),
                track_history={
                    key: {**value, "events": list(value["events"])}
                    for key, value in tracker.history.items()
                },
            )
            frames_read += 1


__all__ = ["TrackedVideoObservation", "iter_tracked_video_frames"]
