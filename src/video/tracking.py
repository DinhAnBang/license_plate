"""Kalman based, two pass vehicle tracking for the video pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..vehicle_detector import VehicleDetection
from .tracking_kalman import KalmanFilterXYAH, xyah_to_xyxy, xyxy_to_xyah


BBox = tuple[int, int, int, int]
TrackStatus = Literal["active", "lost", "removed"]


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ix1, iy1 = max(first[0], second[0]), max(first[1], second[1])
    ix2, iy2 = min(first[2], second[2]), min(first[3], second[3])
    overlap = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_b = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_a + area_b - overlap
    return overlap / union if union > 0 else 0.0


@dataclass(frozen=True, slots=True)
class VehicleTrackingConfig:
    """Detection bands and matching gates; time limits are FPS aware."""

    low_confidence: float = 0.10
    high_confidence: float = 0.50
    new_track_confidence: float = 0.55
    max_lost_seconds: float = 0.50
    first_match_cost_limit: float = 0.83
    second_match_cost_limit: float = 0.65
    min_iou: float = 0.10
    max_center_distance: float = 0.35
    class_mismatch_penalty: float = 0.10
    duplicate_iou_threshold: float = 0.75
    active_duplicate_iou_threshold: float = 0.90
    active_duplicate_min_frames: int = 2
    # Retained for old callers; appearance matching is disabled.
    archive_seconds: float = 10.0
    reidentification_similarity: float = 0.80
    reidentification_min_confidence: float = 0.50

    def __post_init__(self) -> None:
        if not 0 <= self.low_confidence <= self.high_confidence <= self.new_track_confidence <= 1:
            raise ValueError("tracking confidence thresholds must satisfy low <= high <= new <= 1")
        if self.max_lost_seconds <= 0:
            raise ValueError("max_lost_seconds must be positive")
        if not 0 < self.first_match_cost_limit <= 1 or not 0 < self.second_match_cost_limit <= 1:
            raise ValueError("matching cost limits must be in (0, 1]")
        if not 0 < self.min_iou <= 1 or self.max_center_distance <= 0:
            raise ValueError("invalid association geometry limits")
        if not 0 <= self.class_mismatch_penalty < 1:
            raise ValueError("class_mismatch_penalty must be in [0, 1)")
        if not 0 < self.duplicate_iou_threshold <= 1:
            raise ValueError("duplicate_iou_threshold must be in (0, 1]")
        if not 0 < self.active_duplicate_iou_threshold <= 1 or self.active_duplicate_min_frames < 1:
            raise ValueError("invalid active duplicate limits")


@dataclass(frozen=True, slots=True)
class TrackedVehicle:
    track_id: int
    detection: VehicleDetection
    status: TrackStatus
    age_frames: int
    hits: int
    missed_frames: int
    reidentified: bool = False


@dataclass(slots=True)
class _Track:
    track_id: int
    detection: VehicleDetection
    mean: np.ndarray
    covariance: np.ndarray
    first_frame: int
    last_seen_frame: int
    status: TrackStatus = "active"
    hits: int = 1
    missed_frames: int = 0
    predicted_bbox: tuple[float, float, float, float] | None = None
    removal_reason: str | None = None

    def predict(self, kalman: KalmanFilterXYAH) -> None:
        mean = self.mean.copy()
        if self.status == "lost":
            mean[7] = 0.0
        self.mean, self.covariance = kalman.predict(mean, self.covariance)
        self.predicted_bbox = xyah_to_xyxy(self.mean)

    def update(self, kalman: KalmanFilterXYAH, detection: VehicleDetection, frame_index: int) -> None:
        self.mean, self.covariance = kalman.update(
            self.mean, self.covariance, xyxy_to_xyah(detection.bbox)
        )
        self.detection = detection
        self.status = "active"
        self.hits += 1
        self.missed_frames = 0
        self.last_seen_frame = frame_index
        self.predicted_bbox = xyah_to_xyxy(self.mean)


def _assign(
    tracks: Sequence[_Track],
    detections: Sequence[VehicleDetection],
    config: VehicleTrackingConfig,
    cost_limit: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian assignment with explicit unmatched options and a cost gate."""

    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))
    costs = np.full((len(tracks), len(detections)), np.inf, dtype=np.float64)
    for row, track in enumerate(tracks):
        predicted = track.predicted_bbox or track.detection.bbox
        px = (predicted[0] + predicted[2]) / 2
        py = (predicted[1] + predicted[3]) / 2
        diagonal = max(1.0, math.hypot(predicted[2] - predicted[0], predicted[3] - predicted[1]))
        for column, detection in enumerate(detections):
            overlap = bbox_iou(predicted, detection.bbox)
            box = detection.bbox
            center_distance = math.hypot(px - (box[0] + box[2]) / 2, py - (box[1] + box[3]) / 2) / diagonal
            if overlap < config.min_iou and center_distance > config.max_center_distance:
                continue
            motion_score = max(0.0, 1.0 - center_distance)
            cost = 1.0 - (0.80 * overlap + 0.20 * motion_score)
            if track.detection.class_id != detection.class_id:
                cost += config.class_mismatch_penalty
            costs[row, column] = cost

    # A pair that fails the gate cannot displace a feasible match.
    row_count, column_count = costs.shape
    invalid = 1_000_000.0
    augmented = np.full((row_count + column_count, row_count + column_count), invalid)
    augmented[:row_count, :column_count] = np.where(costs <= cost_limit, costs, invalid)
    unmatched_cost = (cost_limit + 1e-6) / 2
    augmented[np.arange(row_count), column_count + np.arange(row_count)] = unmatched_cost
    augmented[row_count + np.arange(column_count), np.arange(column_count)] = unmatched_cost
    augmented[row_count:, column_count:] = 0
    rows, columns = linear_sum_assignment(augmented)
    matches = [
        (int(row), int(column))
        for row, column in zip(rows, columns)
        if row < row_count and column < column_count and costs[row, column] <= cost_limit
    ]
    used_rows = {row for row, _ in matches}
    used_columns = {column for _, column in matches}
    return (
        matches,
        [row for row in range(row_count) if row not in used_rows],
        [column for column in range(column_count) if column not in used_columns],
    )


class VehicleTracker:
    """Keep IDs through short gaps; weak detections only update existing IDs."""

    def __init__(self, *, fps: float, config: VehicleTrackingConfig | None = None) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.fps = float(fps)
        self.config = config or VehicleTrackingConfig()
        self.max_lost_frames = max(1, round(self.fps * self.config.max_lost_seconds))
        self.kalman = KalmanFilterXYAH()
        self._tracks: dict[int, _Track] = {}
        self._removed: dict[int, _Track] = {}
        self._next_track_id = 1
        self._last_frame = -1
        self._duplicate_streaks: dict[tuple[int, int], int] = {}
        self.last_detection_indices: tuple[int, ...] = ()
        self.last_frame_stats: dict[str, int] = {}
        self.counters: dict[str, int] = {
            "vehicle_detections_total": 0,
            "high_confidence_detections": 0,
            "low_confidence_detections": 0,
            "tracks_created": 0,
            "tracks_reactivated": 0,
            "tracks_lost": 0,
            "tracks_removed": 0,
            "second_stage_matches": 0,
            "duplicate_tracks_removed": 0,
            "duplicate_detections_suppressed": 0,
        }
        self.history: dict[int, dict[str, object]] = {}

    @property
    def created_track_count(self) -> int:
        return self.counters["tracks_created"]

    @property
    def reidentified_count(self) -> int:
        """Compatibility: count motion based LOST -> ACTIVE reactivations."""
        return self.counters["tracks_reactivated"]

    @property
    def active_track_ids(self) -> tuple[int, ...]:
        return tuple(sorted(id_ for id_, track in self._tracks.items() if track.status == "active"))

    @property
    def lost_track_ids(self) -> tuple[int, ...]:
        return tuple(sorted(id_ for id_, track in self._tracks.items() if track.status == "lost"))

    @property
    def removed_track_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._removed))

    def _record(self, track: _Track, frame_index: int, status: str) -> None:
        self.history[track.track_id]["events"].append({"frame_index": frame_index, "status": status})

    def _create(self, detection: VehicleDetection, frame_index: int) -> _Track:
        mean, covariance = self.kalman.initiate(xyxy_to_xyah(detection.bbox))
        track = _Track(self._next_track_id, detection, mean, covariance, frame_index, frame_index)
        self._tracks[track.track_id] = track
        self.history[track.track_id] = {
            "track_id": track.track_id,
            "first_frame": frame_index,
            "last_frame": frame_index,
            "hits": 1,
            "events": [{"frame_index": frame_index, "status": "created"}],
            "removal_reason": None,
        }
        self._next_track_id += 1
        self.counters["tracks_created"] += 1
        return track

    def _update(self, track: _Track, detection: VehicleDetection, frame_index: int) -> bool:
        reactivated = track.status == "lost"
        track.update(self.kalman, detection, frame_index)
        entry = self.history[track.track_id]
        entry["last_frame"] = frame_index
        entry["hits"] = track.hits
        if reactivated:
            self.counters["tracks_reactivated"] += 1
            self._record(track, frame_index, "reactivated")
        return reactivated

    def _remove_duplicate_tracks(
        self, assigned: dict[int, tuple[_Track, bool]], frame_index: int
    ) -> int:
        """Only remove active tracks whose observed boxes coincide repeatedly."""

        visible = sorted(assigned.items(), key=lambda item: item[1][0].track_id)
        current_streaks: dict[tuple[int, int], int] = {}
        removed: set[int] = set()
        for first_index in range(len(visible)):
            _, (first, _) = visible[first_index]
            if first.track_id in removed:
                continue
            for second_index in range(first_index + 1, len(visible)):
                _, (second, _) = visible[second_index]
                if second.track_id in removed:
                    continue
                if bbox_iou(first.detection.bbox, second.detection.bbox) < self.config.active_duplicate_iou_threshold:
                    continue
                pair = (first.track_id, second.track_id)
                streak = self._duplicate_streaks.get(pair, 0) + 1
                current_streaks[pair] = streak
                if streak < self.config.active_duplicate_min_frames:
                    continue
                loser = min((first, second), key=lambda track: (track.hits, -track.track_id))
                loser.status = "removed"
                loser.removal_reason = "duplicate"
                self.history[loser.track_id]["removal_reason"] = "duplicate"
                self._record(loser, frame_index, "removed_duplicate")
                self._removed[loser.track_id] = loser
                del self._tracks[loser.track_id]
                removed.add(loser.track_id)
                self.counters["tracks_removed"] += 1
                self.counters["duplicate_tracks_removed"] += 1
        self._duplicate_streaks = current_streaks
        if removed:
            for index in tuple(assigned):
                if assigned[index][0].track_id in removed:
                    del assigned[index]
        return len(removed)

    def update(
        self,
        detections: Sequence[VehicleDetection],
        signatures: Sequence[np.ndarray | None] | None = None,
        *,
        frame_index: int,
    ) -> tuple[TrackedVehicle, ...]:
        """Return visible tracks in detection order.

        The optional signatures argument is accepted for old callers but is
        never used to match or resurrect a track.
        """

        if frame_index <= self._last_frame:
            raise ValueError("frame_index must increase")
        if signatures is not None and len(signatures) != len(detections):
            raise ValueError("signatures must have one item per detection")
        self._last_frame = frame_index
        current = tuple(detections)
        config = self.config
        high_indices = [i for i, d in enumerate(current) if d.confidence >= config.high_confidence]
        low_indices = [
            i for i, d in enumerate(current)
            if config.low_confidence <= d.confidence < config.high_confidence
        ]
        frame_stats = {
            "vehicle_detections_total": len(current),
            "high_confidence_detections": len(high_indices),
            "low_confidence_detections": len(low_indices),
            "tracks_created": 0,
            "tracks_reactivated": 0,
            "tracks_lost": 0,
            "tracks_removed": 0,
            "second_stage_matches": 0,
            "duplicate_detections_suppressed": 0,
            "duplicate_tracks_removed": 0,
        }
        self.counters["vehicle_detections_total"] += len(current)
        self.counters["high_confidence_detections"] += len(high_indices)
        self.counters["low_confidence_detections"] += len(low_indices)

        for track in self._tracks.values():
            track.predict(self.kalman)
        pool = list(sorted(self._tracks.values(), key=lambda item: item.track_id))
        high = [current[i] for i in high_indices]
        first_matches, unmatched_tracks, unmatched_high = _assign(
            pool, high, config, config.first_match_cost_limit
        )
        assigned: dict[int, tuple[_Track, bool]] = {}
        for row, column in first_matches:
            track = pool[row]
            reactivated = self._update(track, high[column], frame_index)
            frame_stats["tracks_reactivated"] += int(reactivated)
            assigned[high_indices[column]] = (track, reactivated)

        remaining = [pool[row] for row in unmatched_tracks]
        low = [current[i] for i in low_indices]
        second_matches, unmatched_second, _ = _assign(
            remaining, low, config, config.second_match_cost_limit
        )
        for row, column in second_matches:
            track = remaining[row]
            reactivated = self._update(track, low[column], frame_index)
            frame_stats["tracks_reactivated"] += int(reactivated)
            frame_stats["second_stage_matches"] += 1
            self.counters["second_stage_matches"] += 1
            assigned[low_indices[column]] = (track, reactivated)

        for row in unmatched_second:
            track = remaining[row]
            if track.status == "active":
                track.status = "lost"
                self.counters["tracks_lost"] += 1
                frame_stats["tracks_lost"] += 1
                self._record(track, frame_index, "lost")
            track.missed_frames = frame_index - track.last_seen_frame
            if track.missed_frames > self.max_lost_frames:
                track.status = "removed"
                track.removal_reason = "timeout"
                self.history[track.track_id]["removal_reason"] = "timeout"
                self._record(track, frame_index, "removed")
                self._removed[track.track_id] = track
                del self._tracks[track.track_id]
                self.counters["tracks_removed"] += 1
                frame_stats["tracks_removed"] += 1

        for column in unmatched_high:
            index = high_indices[column]
            detection = current[index]
            if detection.confidence < config.new_track_confidence:
                continue
            # A second high box over an already visible car must not create
            # a duplicate ID. Geometrically distinct neighbours remain free.
            if any(
                bbox_iou(detection.bbox, item[0].detection.bbox) >= config.duplicate_iou_threshold
                for item in assigned.values()
            ):
                self.counters["duplicate_detections_suppressed"] += 1
                frame_stats["duplicate_detections_suppressed"] += 1
                continue
            track = self._create(detection, frame_index)
            assigned[index] = (track, False)
            frame_stats["tracks_created"] += 1

        removed_duplicates = self._remove_duplicate_tracks(assigned, frame_index)
        frame_stats["duplicate_tracks_removed"] = removed_duplicates
        frame_stats["tracks_removed"] += removed_duplicates

        self.last_frame_stats = frame_stats
        self.last_detection_indices = tuple(sorted(assigned))
        return tuple(
            TrackedVehicle(
                track_id=assigned[index][0].track_id,
                detection=current[index],
                status="active",
                age_frames=frame_index - assigned[index][0].first_frame + 1,
                hits=assigned[index][0].hits,
                missed_frames=0,
                reidentified=assigned[index][1],
            )
            for index in self.last_detection_indices
        )


__all__ = ["TrackedVehicle", "VehicleTracker", "VehicleTrackingConfig", "bbox_iou"]
