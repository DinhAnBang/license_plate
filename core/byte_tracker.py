"""BYTE two-stage association with explicit track lifecycle management."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

import numpy as np

from .assignment import gated_assignment, iou_matrix
from .config import (
    BYTE_HIGH_MATCH_IOU_THRESHOLD,
    BYTE_HIGH_THRESHOLD,
    BYTE_LOW_MATCH_IOU_THRESHOLD,
    BYTE_LOW_THRESHOLD,
    BYTE_REFERENCE_FPS,
    BYTE_TRACK_BUFFER_FRAMES_AT_30FPS,
    BYTE_UNCONFIRMED_MATCH_IOU_THRESHOLD,
    SORT_MIN_HITS,
)
from .kalman_box_tracker import KalmanBoxTracker, KalmanConfig, normalize_box
from .sort_tracker import SortTracker


class TrackState(str, Enum):
    """Lifecycle states used only by :class:`ByteTracker`."""

    NEW = "NEW"
    TRACKED = "TRACKED"
    LOST = "LOST"
    REMOVED = "REMOVED"


class ByteTrack(KalmanBoxTracker):
    """One BYTE track with confidence-source and lifecycle metadata."""

    def __init__(
        self,
        detection: Mapping[str, Any],
        track_id: int,
        frame_index: int,
        config: KalmanConfig | None = None,
    ) -> None:
        super().__init__(detection, track_id, frame_index, config=config)
        self.state = TrackState.NEW
        self.is_activated = False
        self.high_hits = 1
        self.low_hits = 0

    @property
    def start_frame(self) -> int:
        return self.first_frame

    def update_high(self, detection: Mapping[str, Any], frame_index: int) -> bool:
        if not super().update(detection, frame_index):
            return False
        self.high_hits += 1
        return True

    def update_low(self, detection: Mapping[str, Any], frame_index: int) -> bool:
        if not super().update(detection, frame_index):
            return False
        self.low_hits += 1
        return True


def split_detections(
    detections: Sequence[Mapping[str, Any]],
    low_thresh: float = BYTE_LOW_THRESHOLD,
    high_thresh: float = BYTE_HIGH_THRESHOLD,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Split detections into high, low, and discarded groups deterministically."""

    _validate_thresholds(low_thresh, high_thresh)
    high: list[Mapping[str, Any]] = []
    low: list[Mapping[str, Any]] = []
    discarded: list[Mapping[str, Any]] = []
    for detection in detections:
        try:
            confidence = float(detection["conf"])
        except (KeyError, TypeError, ValueError, OverflowError):
            discarded.append(detection)
            continue
        if not math.isfinite(confidence) or confidence < low_thresh:
            discarded.append(detection)
        elif confidence >= high_thresh:
            high.append(detection)
        else:
            low.append(detection)
    return high, low, discarded


class ByteTracker(SortTracker):
    """BYTE tracker with NEW, TRACKED, LOST, and REMOVED states.

    The class remains a ``SortTracker`` subtype for the existing mode contract,
    but owns separate BYTE collections and does not alter SORT's algorithm.
    """

    def __init__(
        self,
        track_high_thresh: float = BYTE_HIGH_THRESHOLD,
        track_low_thresh: float = BYTE_LOW_THRESHOLD,
        high_match_iou_threshold: float = BYTE_HIGH_MATCH_IOU_THRESHOLD,
        low_match_iou_threshold: float = BYTE_LOW_MATCH_IOU_THRESHOLD,
        unconfirmed_match_iou_threshold: float = BYTE_UNCONFIRMED_MATCH_IOU_THRESHOLD,
        track_buffer_frames_at_30fps: int = BYTE_TRACK_BUFFER_FRAMES_AT_30FPS,
        reference_fps: float = BYTE_REFERENCE_FPS,
        min_hits: int = SORT_MIN_HITS,
        *,
        max_age: int | None = None,
        kalman_config: KalmanConfig | None = None,
        trace_enabled: bool = False,
    ) -> None:
        _validate_thresholds(track_low_thresh, track_high_thresh)
        for name, value in (
            ("high_match_iou_threshold", high_match_iou_threshold),
            ("low_match_iou_threshold", low_match_iou_threshold),
            ("unconfirmed_match_iou_threshold", unconfirmed_match_iou_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")
        if max_age is not None:
            track_buffer_frames_at_30fps = max_age
        if not isinstance(track_buffer_frames_at_30fps, int) or track_buffer_frames_at_30fps < 0:
            raise ValueError("track_buffer_frames_at_30fps must be a non-negative integer")
        if not math.isfinite(reference_fps) or reference_fps <= 0.0:
            raise ValueError("reference_fps must be a positive finite number")
        if min_hits < 1:
            raise ValueError("min_hits must be >= 1")

        self.track_high_thresh = float(track_high_thresh)
        self.track_low_thresh = float(track_low_thresh)
        self.high_match_iou_threshold = float(high_match_iou_threshold)
        self.low_match_iou_threshold = float(low_match_iou_threshold)
        self.unconfirmed_match_iou_threshold = float(unconfirmed_match_iou_threshold)
        self.track_buffer_frames_at_30fps = int(track_buffer_frames_at_30fps)
        self.reference_fps = float(reference_fps)
        self.min_hits = int(min_hits)
        self.kalman_config = kalman_config or KalmanConfig()
        self.iou_threshold = self.high_match_iou_threshold
        self.trace_enabled = bool(trace_enabled)
        self.reset()

    @property
    def active_tracks(self) -> dict[int, ByteTrack]:
        """Compatibility view containing every non-removed BYTE track."""

        return {**self.tracked_tracks, **self.lost_tracks, **self.unconfirmed_tracks}

    @property
    def finished_tracks(self) -> list[ByteTrack]:
        """Compatibility view of tracks in the REMOVED state."""

        return list(self.removed_tracks.values())

    def reset(self) -> None:
        self.tracked_tracks: dict[int, ByteTrack] = {}
        self.lost_tracks: dict[int, ByteTrack] = {}
        self.unconfirmed_tracks: dict[int, ByteTrack] = {}
        self.removed_tracks: dict[int, ByteTrack] = {}
        self._next_track_id = 1
        self._last_frame_index: int | None = None
        self._finalized = False

        self.frame_rate = self.reference_fps
        self.effective_track_buffer = self.track_buffer_frames_at_30fps
        self.max_age = self.effective_track_buffer

        self.created_tracks = 0
        self.matched_associations = 0
        self.unmatched_detections = 0
        self.unmatched_tracks = 0
        self.tracks_removed_by_max_age = 0
        self.invalid_predictions = 0

        self.raw_detector_candidates = 0
        self.detections_after_low_threshold = 0
        self.detections_after_nms = 0
        self.high_detections = 0
        self.low_detections = 0
        self.high_matches = 0
        self.low_matches = 0
        self.low_score_recoveries = 0
        self.unmatched_high_detections = 0
        self.unmatched_low_detections = 0
        self.new_tracks_from_high = 0
        self.unmatched_tracks_after_round1 = 0
        self.unmatched_tracks_after_round2 = 0
        self.tracks_expired = 0

        self.unconfirmed_tracks_created = 0
        self.unconfirmed_confirmed = 0
        self.unconfirmed_removed = 0
        self.tracks_marked_lost = 0
        self.reactivated_tracks = 0
        self.removed_after_buffer = 0
        self.trace_records: list[dict[str, Any]] = []

    def configure_frame_rate(self, frame_rate: float) -> int:
        """Configure the BYTE lost buffer from video FPS and return its size."""

        try:
            value = float(frame_rate)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("frame_rate must be a positive finite number") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("frame_rate must be a positive finite number")
        if self._last_frame_index is not None:
            raise RuntimeError("frame_rate must be configured before tracker updates")
        self.frame_rate = value
        self.effective_track_buffer = max(
            0,
            int(round(value / self.reference_fps * self.track_buffer_frames_at_30fps)),
        )
        self.max_age = self.effective_track_buffer
        return self.effective_track_buffer

    def record_detector_stats(self, stats: Mapping[str, Any]) -> None:
        self.raw_detector_candidates += int(stats.get("raw_detector_candidates", 0))
        self.detections_after_low_threshold += int(stats.get("detections_after_threshold", 0))
        self.detections_after_nms += int(stats.get("detections_after_nms", 0))

    def update(
        self,
        detections: Sequence[Mapping[str, Any]],
        frame_index: int,
    ) -> list[dict[str, Any]]:
        self._validate_update(frame_index)
        high, low = self._valid_detections(detections)
        self.high_detections += len(high)
        self.low_detections += len(low)

        for track in list(self.lost_tracks.values()):
            if frame_index - track.last_frame > self.effective_track_buffer:
                self._remove_track(track, after_buffer=True)

        predicted: dict[int, np.ndarray] = {}
        pool = [*self.tracked_tracks.values(), *self.lost_tracks.values()]
        for track in [*pool, *self.unconfirmed_tracks.values()]:
            prediction = track.predict()
            if prediction is None:
                self.invalid_predictions += 1
                self._remove_track(track)
            else:
                predicted[track.track_id] = prediction
        pool = [track for track in pool if track.track_id in predicted]
        unconfirmed = [
            track for track in self.unconfirmed_tracks.values() if track.track_id in predicted
        ]

        high_boxes = [item[2] for item in high]
        low_boxes = [item[2] for item in low]
        emitted: dict[int, dict[str, Any]] = {}

        high_pairs = gated_assignment(
            iou_matrix([predicted[t.track_id] for t in pool], high_boxes),
            self.high_match_iou_threshold,
        )
        matched_pool: set[int] = set()
        matched_high: set[int] = set()
        for track_pos, detection_pos, overlap in high_pairs:
            track = pool[track_pos]
            source_index, detection, box, confidence = high[detection_pos]
            state_before = track.state
            if not track.update_high(detection, frame_index):
                continue
            matched_pool.add(track_pos)
            matched_high.add(detection_pos)
            self.high_matches += 1
            self.matched_associations += 1
            if state_before is TrackState.LOST:
                self._reactivate(track)
                stage = "reactivated_high"
                trace_result = "HIGH_REACTIVATION"
            else:
                stage = "high"
                trace_result = "HIGH_MATCH"
            emitted[source_index] = _tracked_detection(
                track.track_id, confidence, box, top_k_eligible=True, stage=stage
            )
            self._trace_match(
                frame_index, track, predicted[track.track_id], confidence, box,
                "HIGH", overlap, trace_result, state_before,
            )

        unmatched_pool_positions = sorted(set(range(len(pool))) - matched_pool)
        unmatched_current = [
            position
            for position in unmatched_pool_positions
            if pool[position].state is TrackState.TRACKED
        ]
        self.unmatched_tracks_after_round1 += len(unmatched_pool_positions)

        low_pairs = gated_assignment(
            iou_matrix(
                [predicted[pool[position].track_id] for position in unmatched_current],
                low_boxes,
            ),
            self.low_match_iou_threshold,
        )
        matched_current_positions: set[int] = set()
        matched_low: set[int] = set()
        for current_pos, detection_pos, overlap in low_pairs:
            pool_pos = unmatched_current[current_pos]
            track = pool[pool_pos]
            source_index, detection, box, confidence = low[detection_pos]
            if not track.update_low(detection, frame_index):
                continue
            matched_current_positions.add(current_pos)
            matched_low.add(detection_pos)
            self.low_matches += 1
            self.low_score_recoveries += 1
            self.matched_associations += 1
            emitted[source_index] = _tracked_detection(
                track.track_id, confidence, box, top_k_eligible=False, stage="low"
            )
            self._trace_match(
                frame_index, track, predicted[track.track_id], confidence, box,
                "LOW", overlap, "LOW_RECOVERY", TrackState.TRACKED,
            )

        unmatched_current_after_low = [
            pool_pos
            for current_pos, pool_pos in enumerate(unmatched_current)
            if current_pos not in matched_current_positions
        ]
        self.unmatched_tracks_after_round2 += len(unmatched_current_after_low)
        self.unmatched_tracks += len(unmatched_current_after_low)
        for pool_pos in unmatched_current_after_low:
            track = pool[pool_pos]
            state_before = track.state
            self._mark_lost(track)
            self._trace_unmatched(
                frame_index,
                track,
                predicted[track.track_id],
                high,
                low,
                state_before=state_before,
            )

        remaining_high_positions = sorted(set(range(len(high))) - matched_high)
        unconfirmed_pairs = gated_assignment(
            iou_matrix(
                [predicted[t.track_id] for t in unconfirmed],
                [high_boxes[position] for position in remaining_high_positions],
            ),
            self.unconfirmed_match_iou_threshold,
        )
        matched_unconfirmed: set[int] = set()
        confirmed_high_positions: set[int] = set()
        for unconfirmed_pos, remaining_pos, overlap in unconfirmed_pairs:
            track = unconfirmed[unconfirmed_pos]
            high_pos = remaining_high_positions[remaining_pos]
            source_index, detection, box, confidence = high[high_pos]
            if not track.update_high(detection, frame_index):
                continue
            matched_unconfirmed.add(unconfirmed_pos)
            confirmed_high_positions.add(high_pos)
            self._activate(track)
            self.high_matches += 1
            self.matched_associations += 1
            self.unconfirmed_confirmed += 1
            emitted[source_index] = _tracked_detection(
                track.track_id, confidence, box, top_k_eligible=True,
                stage="unconfirmed_high",
            )
            self._trace_match(
                frame_index, track, predicted[track.track_id], confidence, box,
                "HIGH", overlap, "HIGH_CONFIRMATION", TrackState.NEW,
            )

        for position, track in enumerate(unconfirmed):
            if position not in matched_unconfirmed:
                self._trace_unmatched(frame_index, track, predicted[track.track_id], high, low)
                self._remove_track(track, unconfirmed=True)

        remaining_high_positions = [
            position
            for position in remaining_high_positions
            if position not in confirmed_high_positions
        ]
        unmatched_low_positions = sorted(set(range(len(low))) - matched_low)
        self.unmatched_high_detections += len(remaining_high_positions)
        self.unmatched_low_detections += len(unmatched_low_positions)
        self.unmatched_detections += len(remaining_high_positions) + len(unmatched_low_positions)

        for detection_pos in remaining_high_positions:
            source_index, detection, box, confidence = high[detection_pos]
            track = ByteTrack(
                detection, track_id=self._next_track_id, frame_index=frame_index,
                config=self.kalman_config,
            )
            self._next_track_id += 1
            self.unconfirmed_tracks[track.track_id] = track
            self.created_tracks += 1
            self.new_tracks_from_high += 1
            self.unconfirmed_tracks_created += 1
            emitted[source_index] = _tracked_detection(
                track.track_id, confidence, box, top_k_eligible=True, stage="new_high"
            )
            self._trace_new(frame_index, track, confidence, box)

        return [emitted[index] for index in sorted(emitted)]

    def finalize(self) -> list[ByteTrack]:
        if not self._finalized:
            for track in list(self.unconfirmed_tracks.values()):
                self._remove_track(track, unconfirmed=True)
            for track in [*self.tracked_tracks.values(), *self.lost_tracks.values()]:
                self._remove_track(track)
            self._finalized = True
        return self.finished_tracks

    finish = finalize

    def track_summaries(self) -> list[dict[str, Any]]:
        """Return confirmed tracks only; debug counters stay out of official JSON."""

        tracks = sorted(
            (track for track in self.removed_tracks.values() if track.is_activated),
            key=lambda track: track.track_id,
        )
        return [self._summary(track) for track in tracks]

    def all_track_summaries(self) -> list[dict[str, Any]]:
        """Return development summaries, including rejected unconfirmed tracks."""

        known = self.active_tracks | self.removed_tracks
        return [self._summary(known[track_id]) for track_id in sorted(known)]

    @staticmethod
    def _summary(track: ByteTrack) -> dict[str, Any]:
        return {
            "track_id": track.track_id,
            "state": track.state.value,
            "is_activated": track.is_activated,
            "first_frame": track.first_frame,
            "last_frame": track.last_frame,
            "hits": track.hits,
            "high_hits": track.high_hits,
            "low_hits": track.low_hits,
            "high_ratio": track.high_hits / track.hits if track.hits else 0.0,
        }

    def statistics(self) -> dict[str, int | float]:
        return {
            "created_tracks": self.created_tracks,
            "matched_associations": self.matched_associations,
            "unmatched_detections": self.unmatched_detections,
            "unmatched_tracks": self.unmatched_tracks,
            "tracks_removed_by_max_age": self.tracks_removed_by_max_age,
            "invalid_predictions": self.invalid_predictions,
            "raw_detector_candidates": self.raw_detector_candidates,
            "detections_after_low_threshold": self.detections_after_low_threshold,
            "detections_after_nms": self.detections_after_nms,
            "high_detections": self.high_detections,
            "low_detections": self.low_detections,
            "high_matches": self.high_matches,
            "low_matches": self.low_matches,
            "low_score_recoveries": self.low_score_recoveries,
            "unmatched_high_detections": self.unmatched_high_detections,
            "unmatched_low_detections": self.unmatched_low_detections,
            "new_tracks_from_high": self.new_tracks_from_high,
            "new_tracks_created": self.created_tracks,
            "unmatched_tracks_after_round1": self.unmatched_tracks_after_round1,
            "unmatched_tracks_after_round2": self.unmatched_tracks_after_round2,
            "tracks_expired": self.tracks_expired,
            "unconfirmed_tracks_created": self.unconfirmed_tracks_created,
            "unconfirmed_confirmed": self.unconfirmed_confirmed,
            "unconfirmed_removed": self.unconfirmed_removed,
            "tracks_marked_lost": self.tracks_marked_lost,
            "lost_count": self.tracks_marked_lost,
            "reactivated_tracks": self.reactivated_tracks,
            "reactivated_count": self.reactivated_tracks,
            "removed_after_buffer": self.removed_after_buffer,
            "effective_track_buffer": self.effective_track_buffer,
            "frame_rate": self.frame_rate,
        }

    def _validate_update(self, frame_index: int) -> None:
        if self._finalized:
            raise RuntimeError("Tracker has been finalized; call reset() before reuse.")
        if not isinstance(frame_index, int) or frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        if self._last_frame_index is not None and frame_index <= self._last_frame_index:
            raise ValueError("frame_index must increase on every tracker update")
        self._last_frame_index = frame_index

    def _valid_detections(
        self, detections: Sequence[Mapping[str, Any]]
    ) -> tuple[
        list[tuple[int, Mapping[str, Any], list[int], float]],
        list[tuple[int, Mapping[str, Any], list[int], float]],
    ]:
        valid: list[tuple[int, Mapping[str, Any], list[int], float]] = []
        for source_index, detection in enumerate(detections):
            box = normalize_box(detection.get("box"))
            try:
                confidence = float(detection["conf"])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if box is None or not math.isfinite(confidence) or confidence < self.track_low_thresh:
                continue
            valid.append((source_index, detection, box, confidence))
        return (
            [item for item in valid if item[3] >= self.track_high_thresh],
            [item for item in valid if item[3] < self.track_high_thresh],
        )

    def _activate(self, track: ByteTrack) -> None:
        self.unconfirmed_tracks.pop(track.track_id, None)
        track.state = TrackState.TRACKED
        track.is_activated = True
        self.tracked_tracks[track.track_id] = track

    def _mark_lost(self, track: ByteTrack) -> None:
        self.tracked_tracks.pop(track.track_id, None)
        track.state = TrackState.LOST
        self.lost_tracks[track.track_id] = track
        self.tracks_marked_lost += 1

    def _reactivate(self, track: ByteTrack) -> None:
        self.lost_tracks.pop(track.track_id, None)
        track.state = TrackState.TRACKED
        self.tracked_tracks[track.track_id] = track
        self.reactivated_tracks += 1

    def _remove_track(
        self,
        track: ByteTrack,
        *,
        unconfirmed: bool = False,
        after_buffer: bool = False,
    ) -> None:
        was_unconfirmed = track.state is TrackState.NEW
        self.tracked_tracks.pop(track.track_id, None)
        self.lost_tracks.pop(track.track_id, None)
        self.unconfirmed_tracks.pop(track.track_id, None)
        track.state = TrackState.REMOVED
        self.removed_tracks[track.track_id] = track
        if unconfirmed or was_unconfirmed:
            self.unconfirmed_removed += 1
        if after_buffer:
            self.removed_after_buffer += 1
            self.tracks_removed_by_max_age += 1
            self.tracks_expired += 1

    def _trace_match(
        self,
        frame_index: int,
        track: ByteTrack,
        predicted_box: Sequence[float],
        confidence: float,
        box: Sequence[int],
        classification: str,
        overlap: float,
        result: str,
        state_before: TrackState,
    ) -> None:
        if not self.trace_enabled:
            return
        self.trace_records.append(
            {
                "frame_index": frame_index,
                "track_id": track.track_id,
                "predicted_box": _rounded_box(predicted_box),
                "detector_confidence": float(confidence),
                "detection_box": [int(value) for value in box],
                "classification": classification,
                "best_iou": round(float(overlap), 6),
                "association_result": result,
                "state_before": state_before.value,
                "state_after": track.state.value,
            }
        )

    def _trace_unmatched(
        self,
        frame_index: int,
        track: ByteTrack,
        predicted_box: Sequence[float],
        high: Sequence[tuple[int, Mapping[str, Any], list[int], float]],
        low: Sequence[tuple[int, Mapping[str, Any], list[int], float]],
        *,
        state_before: TrackState | None = None,
    ) -> None:
        if not self.trace_enabled:
            return
        candidates = [("HIGH", item) for item in high] + [("LOW", item) for item in low]
        if candidates:
            overlaps = iou_matrix([predicted_box], [item[1][2] for item in candidates])[0]
            best = int(np.argmax(overlaps))
            classification, (_, _, box, confidence) = candidates[best]
            best_iou = float(overlaps[best])
        else:
            classification, box, confidence, best_iou = "NONE", None, None, 0.0
        self.trace_records.append(
            {
                "frame_index": frame_index,
                "track_id": track.track_id,
                "predicted_box": _rounded_box(predicted_box),
                "detector_confidence": confidence,
                "detection_box": box.copy() if box is not None else None,
                "classification": classification,
                "best_iou": round(best_iou, 6),
                "association_result": "UNMATCHED" if candidates else "NO_DETECTION",
                "state_before": (state_before or track.state).value,
                "state_after": track.state.value,
            }
        )

    def _trace_new(
        self,
        frame_index: int,
        track: ByteTrack,
        confidence: float,
        box: Sequence[int],
    ) -> None:
        if not self.trace_enabled:
            return
        self.trace_records.append(
            {
                "frame_index": frame_index,
                "track_id": track.track_id,
                "predicted_box": None,
                "detector_confidence": float(confidence),
                "detection_box": [int(value) for value in box],
                "classification": "HIGH",
                "best_iou": None,
                "association_result": "NEW_TRACK",
                "state_before": None,
                "state_after": TrackState.NEW.value,
            }
        )


def _tracked_detection(
    track_id: int,
    confidence: float,
    box: Sequence[int],
    *,
    top_k_eligible: bool,
    stage: str,
) -> dict[str, Any]:
    return {
        "track_id": int(track_id),
        "conf": float(confidence),
        "box": [int(value) for value in box],
        "top_k_eligible": bool(top_k_eligible),
        "association_stage": stage,
    }


def _rounded_box(box: Sequence[float]) -> list[float]:
    return [round(float(value), 3) for value in box]


def _validate_thresholds(low_thresh: float, high_thresh: float) -> None:
    if not 0.0 <= low_thresh < high_thresh <= 1.0:
        raise ValueError("thresholds must satisfy 0 <= low_thresh < high_thresh <= 1")
