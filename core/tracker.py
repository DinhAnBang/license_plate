"""Lightweight IoU-based tracker for license plate detections."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict


class TrackedDetection(TypedDict):
    track_id: int
    conf: float
    box: list[int]


@dataclass
class Track:
    """State for one track; no image frames are retained in memory."""

    track_id: int
    box: list[int]
    first_frame: int
    last_frame: int
    hits: int = 1
    missed: int = 0


def calculate_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Return IoU for two ``[x1, y1, x2, y2]`` boxes, or 0 for invalid boxes."""

    try:
        if len(box_a) != 4 or len(box_b) != 4:
            return 0.0
        values_a = [float(value) for value in box_a]
        values_b = [float(value) for value in box_b]
    except (TypeError, ValueError, OverflowError):
        return 0.0

    if not all(math.isfinite(value) for value in values_a + values_b):
        return 0.0
    ax1, ay1, ax2, ay2 = values_a
    bx1, by1, bx2, by2 = values_b
    if ax2 <= ax1 or ay2 <= ay1 or bx2 <= bx1 or by2 <= by1:
        return 0.0

    # Intersection-over-Union = intersection area / union area. The validity
    # checks above prevent zero-area boxes and division by zero.
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


class PlateTracker:
    """Track detections with greedy, one-to-one IoU matching."""

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 10) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0.0 and 1.0")
        if max_missed < 0:
            raise ValueError("max_missed must be >= 0")
        self.iou_threshold = float(iou_threshold)
        self.max_missed = int(max_missed)
        self.reset()

    def reset(self) -> None:
        """Start a fresh tracking run and reset the next ID to 1."""

        self.active_tracks: dict[int, Track] = {}
        self.finished_tracks: list[Track] = []
        self._next_track_id = 1
        self._last_frame_index: int | None = None
        self._finalized = False

    def update(
        self,
        detections: Sequence[Mapping[str, Any]],
        frame_index: int,
    ) -> list[TrackedDetection]:
        """Update tracks and return all valid current detections with IDs."""

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
            box = self._normalise_box(detection.get("box"))
            if box is None:
                continue
            try:
                confidence = float(detection["conf"])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(confidence):
                continue
            valid_detections.append(detection)
            valid_boxes.append(box)

        # Build all eligible pairs, then greedily consume the highest IoU pair.
        candidates: list[tuple[float, int, int]] = []
        for track_id, track in self.active_tracks.items():
            for detection_index, box in enumerate(valid_boxes):
                iou = calculate_iou(track.box, box)
                if iou >= self.iou_threshold:
                    candidates.append((iou, track_id, detection_index))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

        assigned_tracks: set[int] = set()
        assigned_detections: set[int] = set()
        detection_track_ids: dict[int, int] = {}
        for _, track_id, detection_index in candidates:
            if track_id in assigned_tracks or detection_index in assigned_detections:
                continue
            assigned_tracks.add(track_id)
            assigned_detections.add(detection_index)
            detection_track_ids[detection_index] = track_id

            track = self.active_tracks[track_id]
            track.box = valid_boxes[detection_index].copy()
            track.last_frame = frame_index
            track.hits += 1
            track.missed = 0

        # Unmatched tracks remain alive through short detector misses. No
        # predicted box is emitted or drawn while a track is missed.
        expired_track_ids: list[int] = []
        for track_id, track in self.active_tracks.items():
            if track_id not in assigned_tracks:
                track.missed += 1
                if track.missed > self.max_missed:
                    expired_track_ids.append(track_id)
        for track_id in expired_track_ids:
            self.finished_tracks.append(self.active_tracks.pop(track_id))

        # Every unmatched current detection starts a new monotonic track ID.
        for detection_index, box in enumerate(valid_boxes):
            if detection_index in assigned_detections:
                continue
            track_id = self._next_track_id
            self._next_track_id += 1
            self.active_tracks[track_id] = Track(
                track_id=track_id,
                box=box.copy(),
                first_frame=frame_index,
                last_frame=frame_index,
            )
            detection_track_ids[detection_index] = track_id

        tracked: list[TrackedDetection] = []
        for detection_index, detection in enumerate(valid_detections):
            tracked.append(
                {
                    "track_id": detection_track_ids[detection_index],
                    "conf": float(detection["conf"]),
                    "box": valid_boxes[detection_index].copy(),
                }
            )
        return tracked

    def finalize(self) -> list[Track]:
        """Move all remaining active tracks to finished tracks and return them."""

        if not self._finalized:
            self.finished_tracks.extend(self.active_tracks.values())
            self.active_tracks.clear()
            self._finalized = True
        return list(self.finished_tracks)

    finish = finalize

    def track_summaries(self) -> list[dict[str, int]]:
        """Return compact summaries sorted by track ID."""

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

    @staticmethod
    def _normalise_box(box: Any) -> list[int] | None:
        if not isinstance(box, Sequence) or isinstance(box, (str, bytes)) or len(box) != 4:
            return None
        try:
            values = [float(value) for value in box]
        except (TypeError, ValueError, OverflowError):
            return None
        if not all(math.isfinite(value) for value in values):
            return None
        x1, y1, x2, y2 = [int(round(value)) for value in values]
        if x1 >= x2 or y1 >= y2:
            return None
        return [x1, y1, x2, y2]
