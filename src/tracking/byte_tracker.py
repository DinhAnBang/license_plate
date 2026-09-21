"""ByteTrack-style vehicle tracker with project-specific identity hardening.

The association core intentionally stays class-agnostic: Kalman prediction,
IoU, score fusion, and Hungarian assignment determine identity.  Temporal
class voting, cross-class detection deduplication, and persistent active-track
duplicate suppression are project-specific extensions and are kept explicit
in the configuration and diagnostics below.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import time

import numpy as np

from src.vehicle_detector import VEHICLE_CLASSES, VehicleDetection

from .kalman_filter import KalmanFilterXYAH
from .deduplication import deduplicate_vehicle_detections
from .matching import (
    fuse_detection_scores,
    iou_distance,
    iou_matrix,
    linear_assignment,
)
from .track import TrackState, VehicleTrack


@dataclass(frozen=True, slots=True)
class TrackedVehicle:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    age: int
    hits: int
    current_detection_class_id: int | None = None
    current_detection_class_name: str | None = None


class ByteTracker:
    """ByteTrack-style two-stage association for vehicle identity.

    ``match_cost_threshold`` and ``second_match_cost_threshold`` are maximum
    costs where IoU cost is ``1 - IoU``. They are not minimum IoU values.
    New HIGH detections start as tentative and are hidden from downstream
    output until ``min_confirmed_hits`` HIGH observations have matched. LOW
    detections can update confirmed tracks but never create or confirm one.

    The core association is deliberately class-independent.  Class voting and
    duplicate cleanup are project-specific hardening, not standard ByteTrack.
    """

    def __init__(
        self,
        fps: float,
        track_high_threshold: float = 0.25,
        track_low_threshold: float = 0.10,
        new_track_threshold: float = 0.25,
        match_cost_threshold: float = 0.80,
        second_match_cost_threshold: float = 0.50,
        unconfirmed_match_cost_threshold: float = 0.80,
        track_buffer_seconds: float = 1.0,
        min_confirmed_hits: int = 2,
        duplicate_iou_threshold: float = 0.85,
        fuse_score: bool = True,
        debug: bool = False,
        cross_class_dedup_enabled: bool = True,
        cross_class_duplicate_iou_threshold: float = 0.90,
        cross_class_duplicate_area_ratio_threshold: float = 0.80,
        cross_class_duplicate_center_distance_threshold: float | None = None,
        active_duplicate_suppression_enabled: bool = False,
        active_duplicate_iou_threshold: float = 0.90,
        active_duplicate_min_frames: int = 3,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be greater than zero")
        if not 0 <= track_low_threshold <= track_high_threshold <= 1:
            raise ValueError(
                "Thresholds must satisfy 0 <= track_low_threshold "
                "<= track_high_threshold <= 1"
            )
        if not track_high_threshold <= new_track_threshold <= 1:
            raise ValueError(
                "new_track_threshold must be between track_high_threshold and 1"
            )
        if not 0 <= match_cost_threshold <= 1:
            raise ValueError("match_cost_threshold must be between 0 and 1")
        if not 0 <= second_match_cost_threshold <= 1:
            raise ValueError(
                "second_match_cost_threshold must be between 0 and 1"
            )
        if not 0 <= unconfirmed_match_cost_threshold <= 1:
            raise ValueError(
                "unconfirmed_match_cost_threshold must be between 0 and 1"
            )
        if track_buffer_seconds < 0:
            raise ValueError("track_buffer_seconds cannot be negative")
        if not isinstance(min_confirmed_hits, int) or min_confirmed_hits < 1:
            raise ValueError("min_confirmed_hits must be an integer >= 1")
        if not 0 < duplicate_iou_threshold <= 1:
            raise ValueError("duplicate_iou_threshold must be in (0, 1]")
        if not 0 < cross_class_duplicate_iou_threshold <= 1:
            raise ValueError(
                "cross_class_duplicate_iou_threshold must be in (0, 1]"
            )
        if not 0 < cross_class_duplicate_area_ratio_threshold <= 1:
            raise ValueError(
                "cross_class_duplicate_area_ratio_threshold must be in (0, 1]"
            )
        if (
            cross_class_duplicate_center_distance_threshold is not None
            and cross_class_duplicate_center_distance_threshold < 0
        ):
            raise ValueError(
                "cross_class_duplicate_center_distance_threshold cannot be negative"
            )
        if not 0 < active_duplicate_iou_threshold <= 1:
            raise ValueError("active_duplicate_iou_threshold must be in (0, 1]")
        if not isinstance(active_duplicate_min_frames, int) or active_duplicate_min_frames < 1:
            raise ValueError("active_duplicate_min_frames must be an integer >= 1")

        self.fps = float(fps)
        self.track_high_threshold = float(track_high_threshold)
        self.track_low_threshold = float(track_low_threshold)
        self.new_track_threshold = float(new_track_threshold)
        self.match_cost_threshold = float(match_cost_threshold)
        self.second_match_cost_threshold = float(second_match_cost_threshold)
        self.unconfirmed_match_cost_threshold = float(
            unconfirmed_match_cost_threshold
        )
        self.track_buffer_seconds = float(track_buffer_seconds)
        self.max_lost_frames = int(round(self.fps * self.track_buffer_seconds))
        self.min_confirmed_hits = min_confirmed_hits
        self.duplicate_iou_threshold = float(duplicate_iou_threshold)
        self.fuse_score = fuse_score
        self.debug = debug
        self.cross_class_dedup_enabled = bool(cross_class_dedup_enabled)
        self.cross_class_duplicate_iou_threshold = float(
            cross_class_duplicate_iou_threshold
        )
        self.cross_class_duplicate_area_ratio_threshold = float(
            cross_class_duplicate_area_ratio_threshold
        )
        self.cross_class_duplicate_center_distance_threshold = (
            None
            if cross_class_duplicate_center_distance_threshold is None
            else float(cross_class_duplicate_center_distance_threshold)
        )
        self.active_duplicate_suppression_enabled = bool(
            active_duplicate_suppression_enabled
        )
        self.active_duplicate_iou_threshold = float(active_duplicate_iou_threshold)
        self.active_duplicate_min_frames = active_duplicate_min_frames

        self.kalman_filter = KalmanFilterXYAH()
        self.tracked_tracks: list[VehicleTrack] = []
        self.unconfirmed_tracks: list[VehicleTrack] = []
        self.lost_tracks: list[VehicleTrack] = []
        self.removed_tracks: list[VehicleTrack] = []
        self._all_tracks: dict[int, VehicleTrack] = {}
        self._next_track_id = 1
        self._last_frame_index = -1
        self.last_debug_stats: dict[str, object] = {}
        self._duplicate_pair_streaks: dict[tuple[int, int], int] = {}
        self.cross_class_dedup_seconds = 0.0
        self.active_duplicate_suppression_seconds = 0.0
        self._diagnostics: dict[str, int] = {
            "cross_class_duplicate_detection_groups": 0,
            "cross_class_detections_removed": 0,
            "cross_class_matches": 0,
            "class_switch_matches": 0,
            "active_duplicate_track_pairs": 0,
            "active_duplicate_tracks_removed": 0,
            "track_fragmentation_candidates": 0,
        }

    @property
    def all_tracks(self) -> tuple[VehicleTrack, ...]:
        return tuple(
            self._all_tracks[track_id] for track_id in sorted(self._all_tracks)
        )

    @property
    def unique_track_count(self) -> int:
        return len(self._all_tracks)

    @property
    def diagnostics(self) -> dict[str, int]:
        """Return cumulative V2.1 diagnostics without exposing mutable state."""

        return dict(self._diagnostics)

    def update(
        self,
        detections: list[VehicleDetection],
        frame_index: int,
    ) -> list[TrackedVehicle]:
        """Advance the tracker by one video frame."""

        if frame_index < 0 or frame_index <= self._last_frame_index:
            raise ValueError(
                "frame_index must be non-negative and strictly increasing"
            )
        self._last_frame_index = frame_index

        valid_detections = [
            detection
            for detection in detections
            if detection.class_id in VEHICLE_CLASSES
            and detection.confidence >= self.track_low_threshold
        ]
        dedup_groups = 0
        dedup_removed = 0
        dedup_started = time.perf_counter()
        if self.cross_class_dedup_enabled:
            deduplication = deduplicate_vehicle_detections(
                valid_detections,
                iou_threshold=self.cross_class_duplicate_iou_threshold,
                area_ratio_threshold=self.cross_class_duplicate_area_ratio_threshold,
                center_distance_threshold=(
                    self.cross_class_duplicate_center_distance_threshold
                ),
            )
            valid_detections = list(deduplication.detections)
            dedup_groups = deduplication.duplicate_groups
            dedup_removed = deduplication.detections_removed
            self._diagnostics["cross_class_duplicate_detection_groups"] += (
                dedup_groups
            )
            self._diagnostics["cross_class_detections_removed"] += dedup_removed
        self.cross_class_dedup_seconds += time.perf_counter() - dedup_started
        high_detections = [
            detection
            for detection in valid_detections
            if detection.confidence >= self.track_high_threshold
        ]
        low_detections = [
            detection
            for detection in valid_detections
            if detection.confidence < self.track_high_threshold
        ]

        before_tracked = len(self.tracked_tracks)
        before_unconfirmed = len(self.unconfirmed_tracks)
        before_lost = len(self.lost_tracks)
        track_pool = self.tracked_tracks + self.lost_tracks
        for track in track_pool:
            track.predict(self.kalman_filter, frame_index)
        tentative_pool = list(self.unconfirmed_tracks)
        for track in tentative_pool:
            track.predict(self.kalman_filter, frame_index)

        first_iou = iou_matrix(track_pool, high_detections)
        first_cost = 1.0 - first_iou
        if self.fuse_score:
            first_cost = fuse_detection_scores(first_cost, high_detections)
        first_matches, unmatched_pool, unmatched_high = linear_assignment(
            first_cost, self.match_cost_threshold
        )

        frame_cross_class_matches = 0
        frame_class_switch_matches = 0
        for track_index, detection_index in first_matches:
            track = track_pool[track_index]
            detection = high_detections[detection_index]
            cross_class, class_switch = self._record_class_match(
                track, detection
            )
            frame_cross_class_matches += int(cross_class)
            frame_class_switch_matches += int(class_switch)
            if track.state is TrackState.LOST:
                track.re_activate(self.kalman_filter, detection, frame_index)
            else:
                track.update(self.kalman_filter, detection, frame_index)

        if first_iou.size:
            fragmentation_iou_threshold = 1.0 - self.match_cost_threshold
            fragmentation_candidates = sum(
                int(
                    np.any(
                        first_iou[:, detection_index]
                        >= fragmentation_iou_threshold
                    )
                )
                for detection_index in unmatched_high
            )
        else:
            fragmentation_candidates = 0
        self._diagnostics["track_fragmentation_candidates"] += (
            fragmentation_candidates
        )

        second_candidates = [
            track_pool[index]
            for index in unmatched_pool
            if track_pool[index].state is TrackState.TRACKED
        ]
        second_cost = iou_distance(second_candidates, low_detections)
        second_matches, unmatched_second, _ = linear_assignment(
            second_cost, self.second_match_cost_threshold
        )
        for track_index, detection_index in second_matches:
            cross_class, class_switch = self._record_class_match(
                second_candidates[track_index], low_detections[detection_index]
            )
            frame_cross_class_matches += int(cross_class)
            frame_class_switch_matches += int(class_switch)
            second_candidates[track_index].update(
                self.kalman_filter,
                low_detections[detection_index],
                frame_index,
            )

        newly_lost = 0
        for track_index in unmatched_second:
            track = second_candidates[track_index]
            track.mark_lost(frame_index)
            newly_lost += 1

        remaining_high = [
            high_detections[index] for index in unmatched_high
        ]
        tentative_cost = iou_distance(tentative_pool, remaining_high)
        if self.fuse_score:
            tentative_cost = fuse_detection_scores(
                tentative_cost, remaining_high
            )
        (
            tentative_matches,
            unmatched_tentative,
            unmatched_remaining_high,
        ) = linear_assignment(
            tentative_cost, self.unconfirmed_match_cost_threshold
        )

        newly_confirmed = 0
        for track_index, detection_index in tentative_matches:
            track = tentative_pool[track_index]
            should_confirm = track.hits + 1 >= self.min_confirmed_hits
            cross_class, class_switch = self._record_class_match(
                track, remaining_high[detection_index]
            )
            frame_cross_class_matches += int(cross_class)
            frame_class_switch_matches += int(class_switch)
            track.update(
                self.kalman_filter,
                remaining_high[detection_index],
                frame_index,
                confirmed=should_confirm,
            )
            if should_confirm:
                self._register_confirmed(track)
                newly_confirmed += 1

        removed_tentative = 0
        for track_index in unmatched_tentative:
            tentative_pool[track_index].mark_removed()
            removed_tentative += 1

        new_tracks = 0
        created_tracks: list[VehicleTrack] = []
        for detection_index in unmatched_remaining_high:
            detection = remaining_high[detection_index]
            if detection.confidence < self.new_track_threshold:
                continue
            track = VehicleTrack(detection)
            confirmed = self.min_confirmed_hits <= 1
            track.activate(
                self.kalman_filter,
                frame_index,
                self._next_track_id,
                confirmed=confirmed,
            )
            self._next_track_id += 1
            if confirmed:
                self._register_confirmed(track)
                newly_confirmed += 1
            created_tracks.append(track)
            new_tracks += 1

        all_current_tracks = track_pool + tentative_pool + created_tracks
        newly_removed = 0
        for track in all_current_tracks:
            if (
                track.state is TrackState.LOST
                and frame_index - track.last_frame > self.max_lost_frames
            ):
                track.mark_removed()
                newly_removed += 1

        self.tracked_tracks = self._unique_tracks(
            track
            for track in all_current_tracks
            if track.state is TrackState.TRACKED
        )
        self.unconfirmed_tracks = self._unique_tracks(
            track
            for track in all_current_tracks
            if track.state is TrackState.NEW
        )
        self.lost_tracks = self._unique_tracks(
            track
            for track in all_current_tracks
            if track.state is TrackState.LOST
        )
        self.removed_tracks = self._unique_tracks(
            [*self.removed_tracks]
            + [
                track
                for track in all_current_tracks
                if track.state is TrackState.REMOVED
            ]
        )

        (
            self.tracked_tracks,
            self.lost_tracks,
            duplicate_tracks,
        ) = self._suppress_duplicate_tracks(
            self.tracked_tracks, self.lost_tracks
        )
        active_duplicate_started = time.perf_counter()
        (
            self.tracked_tracks,
            active_duplicate_tracks,
            active_duplicate_pairs,
        ) = self._suppress_persistent_active_duplicates(
            self.tracked_tracks, frame_index
        )
        self.active_duplicate_suppression_seconds += (
            time.perf_counter() - active_duplicate_started
        )
        duplicate_tracks.extend(active_duplicate_tracks)
        self.removed_tracks = self._unique_tracks(
            [*self.removed_tracks, *duplicate_tracks]
        )
        self._diagnostics["cross_class_matches"] += frame_cross_class_matches
        self._diagnostics["class_switch_matches"] += frame_class_switch_matches
        self._diagnostics["active_duplicate_track_pairs"] += len(
            active_duplicate_pairs
        )
        self._diagnostics["active_duplicate_tracks_removed"] += len(
            active_duplicate_tracks
        )

        self.last_debug_stats = {
            "frame": frame_index,
            "detections_total": len(valid_detections),
            "detections_high": len(high_detections),
            "detections_low": len(low_detections),
            "tracks_before_tracked": before_tracked,
            "tracks_before_unconfirmed": before_unconfirmed,
            "tracks_before_lost": before_lost,
            "first_matches": len(first_matches),
            "first_unmatched_tracks": len(unmatched_pool),
            "first_unmatched_high": len(unmatched_high),
            "second_matches": len(second_matches),
            "tentative_matches": len(tentative_matches),
            "newly_confirmed": newly_confirmed,
            "removed_tentative": removed_tentative,
            "new_tracks": new_tracks,
            "newly_lost": newly_lost,
            "newly_removed": newly_removed,
            "duplicates_removed": len(duplicate_tracks),
            "cross_class_duplicate_detection_groups": dedup_groups,
            "cross_class_detections_removed": dedup_removed,
            "cross_class_matches": frame_cross_class_matches,
            "class_switch_matches": frame_class_switch_matches,
            "active_duplicate_track_pairs": len(active_duplicate_pairs),
            "active_duplicate_tracks_removed": len(active_duplicate_tracks),
            "active_duplicate_pair_streaks": {
                f"{first_id}/{second_id}": streak
                for (first_id, second_id), streak in sorted(
                    self._duplicate_pair_streaks.items()
                )
            },
            "track_fragmentation_candidates": fragmentation_candidates,
        }
        self._print_debug()
        return [self._to_output(track) for track in self.tracked_tracks]

    @staticmethod
    def _unique_tracks(tracks: Iterable[VehicleTrack]) -> list[VehicleTrack]:
        by_id: dict[int, VehicleTrack] = {}
        for track in tracks:
            by_id[track.track_id] = track
        return [by_id[track_id] for track_id in sorted(by_id)]

    def _register_confirmed(self, track: VehicleTrack) -> None:
        self._all_tracks[track.track_id] = track

    def _record_class_match(
        self, track: VehicleTrack, detection: VehicleDetection
    ) -> tuple[bool, bool]:
        """Record class changes without letting them influence association."""

        cross_class = track.class_id != detection.class_id
        class_switch = track.current_detection_class_id != detection.class_id
        return cross_class, class_switch

    def _suppress_duplicate_tracks(
        self,
        tracked_tracks: list[VehicleTrack],
        lost_tracks: list[VehicleTrack],
    ) -> tuple[list[VehicleTrack], list[VehicleTrack], list[VehicleTrack]]:
        """Remove near-identical tracks across TRACKED and LOST sets.

        This mirrors ByteTrack's duplicate-set cleanup. A pair is considered
        duplicate only when its IoU reaches ``duplicate_iou_threshold``. The
        decision is class-agnostic because class is not identity. The track
        with the stronger identity history wins: hits, confirmed age, older
        first frame, then smaller ID. This does not merge or add hit counts.
        """

        similarities = iou_matrix(tracked_tracks, lost_tracks)
        remove_tracked: set[int] = set()
        remove_lost: set[int] = set()
        for tracked_index, lost_index in np.argwhere(
            similarities >= self.duplicate_iou_threshold
        ):
            tracked = tracked_tracks[int(tracked_index)]
            lost = lost_tracks[int(lost_index)]
            tracked_rank = self._identity_rank(tracked)
            lost_rank = self._identity_rank(lost)
            if tracked_rank >= lost_rank:
                remove_lost.add(int(lost_index))
            else:
                remove_tracked.add(int(tracked_index))

        duplicates: list[VehicleTrack] = []
        kept_tracked: list[VehicleTrack] = []
        for index, track in enumerate(tracked_tracks):
            if index in remove_tracked:
                track.mark_removed()
                duplicates.append(track)
            else:
                kept_tracked.append(track)
        kept_lost: list[VehicleTrack] = []
        for index, track in enumerate(lost_tracks):
            if index in remove_lost:
                track.mark_removed()
                duplicates.append(track)
            else:
                kept_lost.append(track)
        return kept_tracked, kept_lost, self._unique_tracks(duplicates)

    def _suppress_persistent_active_duplicates(
        self, tracked_tracks: list[VehicleTrack], frame_index: int
    ) -> tuple[list[VehicleTrack], list[VehicleTrack], list[tuple[int, int]]]:
        """Suppress TRACKED↔TRACKED duplicates only after persistent overlap.

        This is a project-specific extension. A single overlapping frame is
        intentionally insufficient because two real vehicles can occlude one
        another. Pair streaks are frame-consecutive and are cleaned whenever a
        pair disappears or either track is removed.
        """

        del frame_index  # The update call already enforces strictly increasing frames.
        similarities = iou_matrix(tracked_tracks, tracked_tracks)
        previous_streaks = self._duplicate_pair_streaks
        current_streaks: dict[tuple[int, int], int] = {}
        active_pairs: list[tuple[int, int]] = []
        by_id = {track.track_id: track for track in tracked_tracks}

        for first_index in range(len(tracked_tracks)):
            for second_index in range(first_index + 1, len(tracked_tracks)):
                if (
                    similarities[first_index, second_index]
                    < self.active_duplicate_iou_threshold
                ):
                    continue
                first_id = tracked_tracks[first_index].track_id
                second_id = tracked_tracks[second_index].track_id
                pair = (min(first_id, second_id), max(first_id, second_id))
                current_streaks[pair] = previous_streaks.get(pair, 0) + 1
                active_pairs.append(pair)

        if not self.active_duplicate_suppression_enabled:
            # Still measure active duplicate pairs for Stage B diagnostics, but
            # do not remove any track when this project-specific extension is
            # disabled.
            self._duplicate_pair_streaks.clear()
            return tracked_tracks, [], active_pairs

        duplicate_ids: set[int] = set()
        duplicate_tracks: list[VehicleTrack] = []
        for pair in sorted(current_streaks):
            if current_streaks[pair] < self.active_duplicate_min_frames:
                continue
            first = by_id.get(pair[0])
            second = by_id.get(pair[1])
            if first is None or second is None:
                continue
            if first.track_id in duplicate_ids or second.track_id in duplicate_ids:
                continue
            _keeper, duplicate = sorted(
                (first, second), key=self._identity_rank, reverse=True
            )
            duplicate.mark_removed()
            duplicate_ids.add(duplicate.track_id)
            duplicate_tracks.append(duplicate)

        self._duplicate_pair_streaks = {
            pair: streak
            for pair, streak in current_streaks.items()
            if pair[0] not in duplicate_ids and pair[1] not in duplicate_ids
        }
        kept_tracks = [
            track for track in tracked_tracks if track.track_id not in duplicate_ids
        ]
        return kept_tracks, self._unique_tracks(duplicate_tracks), active_pairs

    @staticmethod
    def _identity_rank(track: VehicleTrack) -> tuple[int, int, int, int]:
        """Rank identity history without using semantic class or confidence."""

        return (
            track.hits,
            track.track_length,
            -track.first_frame,
            -track.track_id,
        )

    @staticmethod
    def _to_output(track: VehicleTrack) -> TrackedVehicle:
        return TrackedVehicle(
            track_id=track.track_id,
            class_id=track.class_id,
            class_name=track.class_name,
            confidence=track.confidence,
            bbox=track.bbox_as_int,
            age=track.age,
            hits=track.hits,
            current_detection_class_id=track.current_detection_class_id,
            current_detection_class_name=track.current_detection_class_name,
        )

    def _print_debug(self) -> None:
        if not self.debug:
            return
        stats = self.last_debug_stats
        print(f"Frame: {stats['frame']}")
        print("Detections:")
        print(f"  total = {stats['detections_total']}")
        print(f"  high = {stats['detections_high']}")
        print(f"  low = {stats['detections_low']}")
        print("Tracks before:")
        print(f"  tracked = {stats['tracks_before_tracked']}")
        print(f"  unconfirmed = {stats['tracks_before_unconfirmed']}")
        print(f"  lost = {stats['tracks_before_lost']}")
        print("First association:")
        print(f"  matches = {stats['first_matches']}")
        print(f"  unmatched_tracks = {stats['first_unmatched_tracks']}")
        print(f"  unmatched_high = {stats['first_unmatched_high']}")
        print("Second association:")
        print(f"  matches = {stats['second_matches']}")
        print("Tentative association:")
        print(f"  matches = {stats['tentative_matches']}")
        print(f"  confirmed = {stats['newly_confirmed']}")
        print(f"  removed = {stats['removed_tentative']}")
        print(f"New tracks = {stats['new_tracks']}")
        print(f"Lost = {stats['newly_lost']}")
        print(f"Removed = {stats['newly_removed']}")
        print(f"Duplicates removed = {stats['duplicates_removed']}")
        print(
            "Cross-class dedup = "
            f"groups={stats['cross_class_duplicate_detection_groups']}, "
            f"removed={stats['cross_class_detections_removed']}"
        )
        print(
            "Class-independent matches = "
            f"cross_class={stats['cross_class_matches']}, "
            f"switches={stats['class_switch_matches']}"
        )
        print(
            "Active duplicate pairs = "
            f"{stats['active_duplicate_track_pairs']}, "
            f"removed={stats['active_duplicate_tracks_removed']}"
        )
