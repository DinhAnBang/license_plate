"""SORT tracker: Kalman prediction, IoU cost, and Hungarian assignment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .assignment import gated_assignment, iou_matrix
from .config import SORT_IOU_THRESHOLD, SORT_MAX_AGE, SORT_MIN_HITS
from .kalman_box_tracker import KalmanBoxTracker, KalmanConfig, normalize_box
from .tracker import TrackedDetection


class SortTracker:
    """Dependency-free SORT implementation compatible with ``PlateTracker``."""

    def __init__(
        self,
        iou_threshold: float = SORT_IOU_THRESHOLD,
        max_age: int = SORT_MAX_AGE,
        min_hits: int = SORT_MIN_HITS,
        kalman_config: KalmanConfig | None = None,
    ) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0.0 and 1.0")
        if max_age < 0:
            raise ValueError("max_age must be >= 0")
        if min_hits < 1:
            raise ValueError("min_hits must be >= 1")
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.kalman_config = kalman_config or KalmanConfig()
        self.reset()

    def reset(self) -> None:
        self.active_tracks: dict[int, KalmanBoxTracker] = {}
        self.finished_tracks: list[KalmanBoxTracker] = []
        self._next_track_id = 1
        self._last_frame_index: int | None = None
        self._finalized = False
        self.created_tracks = 0
        self.matched_associations = 0
        self.unmatched_detections = 0
        self.unmatched_tracks = 0
        self.tracks_removed_by_max_age = 0
        self.invalid_predictions = 0

    def update(
        self,
        detections: Sequence[Mapping[str, Any]],
        frame_index: int,
    ) -> list[TrackedDetection]:
        if self._finalized:
            raise RuntimeError("Tracker has been finalized; call reset() before reuse.")
        if not isinstance(frame_index, int) or frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        if self._last_frame_index is not None and frame_index <= self._last_frame_index:
            raise ValueError("frame_index must increase on every tracker update")
        self._last_frame_index = frame_index

        valid_detections: list[Mapping[str, Any]] = []
        valid_boxes: list[list[int]] = []
        for detection in detections:
            box = normalize_box(detection.get("box"))
            try:
                confidence = float(detection["conf"])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if box is None or not np.isfinite(confidence):
                continue
            valid_detections.append(detection)
            valid_boxes.append(box)

        track_ids: list[int] = []
        predicted_boxes: list[np.ndarray] = []
        invalid_track_ids: list[int] = []
        for track_id, track in self.active_tracks.items():
            prediction = track.predict()
            if prediction is None:
                invalid_track_ids.append(track_id)
                continue
            track_ids.append(track_id)
            predicted_boxes.append(prediction)
        for track_id in invalid_track_ids:
            self.finished_tracks.append(self.active_tracks.pop(track_id))
            self.invalid_predictions += 1

        overlaps = iou_matrix(predicted_boxes, valid_boxes)
        matched_track_indexes: set[int] = set()
        matched_detection_indexes: set[int] = set()
        detection_track_ids: dict[int, int] = {}

        for track_index, detection_index, _ in gated_assignment(overlaps, self.iou_threshold):
            track_id = track_ids[track_index]
            track = self.active_tracks[track_id]
            if not track.update(valid_detections[detection_index], frame_index):
                continue
            matched_track_indexes.add(track_index)
            matched_detection_indexes.add(detection_index)
            detection_track_ids[detection_index] = track_id
            self.matched_associations += 1

        unmatched_track_indexes = set(range(len(track_ids))) - matched_track_indexes
        unmatched_detection_indexes = set(range(len(valid_detections))) - matched_detection_indexes
        self.unmatched_tracks += len(unmatched_track_indexes)
        self.unmatched_detections += len(unmatched_detection_indexes)

        expired_track_ids: list[int] = []
        for track_index in sorted(unmatched_track_indexes):
            track_id = track_ids[track_index]
            if self.active_tracks[track_id].time_since_update > self.max_age:
                expired_track_ids.append(track_id)
        for track_id in expired_track_ids:
            self.finished_tracks.append(self.active_tracks.pop(track_id))
            self.tracks_removed_by_max_age += 1

        for detection_index in sorted(unmatched_detection_indexes):
            track_id = self._next_track_id
            self._next_track_id += 1
            self.created_tracks += 1
            self.active_tracks[track_id] = KalmanBoxTracker(
                valid_detections[detection_index],
                track_id=track_id,
                frame_index=frame_index,
                config=self.kalman_config,
            )
            detection_track_ids[detection_index] = track_id

        tracked: list[TrackedDetection] = []
        for detection_index, detection in enumerate(valid_detections):
            track_id = detection_track_ids[detection_index]
            track = self.active_tracks[track_id]
            if track.hits < self.min_hits:
                continue
            # Only the real current detection is emitted. Kalman predictions
            # never enter drawing, cropping, Top-K, OCR, or confidence output.
            tracked.append(
                {
                    "track_id": track_id,
                    "conf": float(detection["conf"]),
                    "box": valid_boxes[detection_index].copy(),
                }
            )
        return tracked

    def finalize(self) -> list[KalmanBoxTracker]:
        if not self._finalized:
            self.finished_tracks.extend(self.active_tracks.values())
            self.active_tracks.clear()
            self._finalized = True
        return list(self.finished_tracks)

    finish = finalize

    def track_summaries(self) -> list[dict[str, int]]:
        tracks = sorted(self.finished_tracks, key=lambda track: track.track_id)
        return [
            {
                "track_id": track.track_id,
                "first_frame": track.first_frame,
                "last_frame": track.last_frame,
                "hits": track.hits,
            }
            for track in tracks
        ]

    def statistics(self) -> dict[str, int]:
        return {
            "created_tracks": self.created_tracks,
            "matched_associations": self.matched_associations,
            "unmatched_detections": self.unmatched_detections,
            "unmatched_tracks": self.unmatched_tracks,
            "tracks_removed_by_max_age": self.tracks_removed_by_max_age,
            "invalid_predictions": self.invalid_predictions,
        }
