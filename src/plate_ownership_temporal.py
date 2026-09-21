"""Stateful V3.2 plate ownership using relative geometry and history.

V3.1 remains in :mod:`src.plate_ownership` for regression compatibility.
This resolver consumes raw per-track plate candidates, resolves one frame at a
time, and updates bounded history only from trusted ownership decisions.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from math import isfinite
from typing import Iterable, Sequence

import numpy as np

from .plate_detector import TrackedPlateCandidate
from .plate_ownership import (
    _intersection_area,
    bbox_iou,
    intersection_over_plate_area,
    point_inside_bbox,
)


BBox = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class RelativePlateGeometry:
    """Plate location and scale normalized by its parent vehicle bbox."""

    center_x: float
    center_y: float
    relative_width: float
    relative_height: float
    relative_area: float


@dataclass(frozen=True, slots=True)
class TemporalPlateOwnershipConfig:
    history_size: int = 20
    min_history_samples: int = 5
    definite_duplicate_iou_threshold: float = 0.80
    ownership_conflict_iou_threshold: float = 0.70
    overlap_over_smaller_threshold: float = 0.85
    temporal_weight: float = 0.40
    tight_parent_weight: float = 0.25
    containment_weight: float = 0.15
    plate_conf_weight: float = 0.15
    vehicle_conf_weight: float = 0.05
    cold_start_temporal_score: float = 0.50
    min_history_update_margin: float = 0.10
    center_x_tolerance: float = 0.20
    center_y_tolerance: float = 0.20
    relative_width_tolerance: float = 0.50
    relative_height_tolerance: float = 0.50
    minimum_center_scale: float = 0.05
    minimum_relative_size_scale: float = 0.05

    def __post_init__(self) -> None:
        if self.history_size < 1:
            raise ValueError("history_size must be >= 1")
        if self.min_history_samples < 1:
            raise ValueError("min_history_samples must be >= 1")
        if self.min_history_samples > self.history_size:
            raise ValueError("min_history_samples cannot exceed history_size")
        for name in (
            "definite_duplicate_iou_threshold",
            "ownership_conflict_iou_threshold",
            "overlap_over_smaller_threshold",
        ):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.ownership_conflict_iou_threshold > self.definite_duplicate_iou_threshold:
            raise ValueError(
                "ownership_conflict_iou_threshold cannot exceed "
                "definite_duplicate_iou_threshold"
            )
        weights = (
            self.temporal_weight,
            self.tight_parent_weight,
            self.containment_weight,
            self.plate_conf_weight,
            self.vehicle_conf_weight,
        )
        if any(weight < 0.0 for weight in weights):
            raise ValueError("owner score weights cannot be negative")
        if not np.isclose(sum(weights), 1.0):
            raise ValueError("owner score weights must sum to 1")
        for name in (
            "cold_start_temporal_score",
            "min_history_update_margin",
            "center_x_tolerance",
            "center_y_tolerance",
            "relative_width_tolerance",
            "relative_height_tolerance",
            "minimum_center_scale",
            "minimum_relative_size_scale",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} cannot be negative")


@dataclass(frozen=True, slots=True)
class TemporalCandidateDiagnostic:
    track_id: int
    plate_bbox: BBox
    vehicle_bbox: BBox
    relative_geometry: RelativePlateGeometry
    containment_score: float
    temporal_score: float
    tight_parent_score: float
    owner_score: float
    history_samples: int
    history_reliable: bool
    conflict_group: int | None
    selected: bool
    history_updated: bool


@dataclass(frozen=True, slots=True)
class TemporalPlateOwnershipStats:
    raw_candidates: int
    invalid_candidates: int
    conflict_groups: int
    definite_duplicate_groups: int
    ambiguous_conflict_groups: int
    removed_conflict_candidates: int
    final_results: int
    changed_owner_groups: int
    history_updates: int
    elapsed_ms: float
    relative_geometry_ms: float
    history_scoring_ms: float


@dataclass(frozen=True, slots=True)
class TemporalPlateOwnershipResolution:
    candidates: tuple[TrackedPlateCandidate, ...]
    stats: TemporalPlateOwnershipStats
    diagnostics: tuple[TemporalCandidateDiagnostic, ...] = ()


@dataclass(slots=True)
class _CandidateState:
    candidate: TrackedPlateCandidate
    geometry: RelativePlateGeometry
    containment_score: float
    temporal_score: float
    tight_parent_score: float = 0.0
    owner_score: float = 0.0
    history_samples: int = 0
    history_reliable: bool = False
    conflict_group: int | None = None
    selected: bool = False
    history_updated: bool = False


def _valid_bbox(bbox: BBox) -> bool:
    return (
        len(bbox) == 4
        and all(isfinite(float(value)) for value in bbox)
        and bbox[2] > bbox[0]
        and bbox[3] > bbox[1]
    )


def compute_relative_geometry(
    plate_bbox: BBox,
    vehicle_bbox: BBox,
) -> RelativePlateGeometry:
    """Return finite plate geometry normalized by a vehicle bbox."""

    if not _valid_bbox(vehicle_bbox):
        raise ValueError(f"vehicle_bbox must have positive finite area: {vehicle_bbox}")
    if not _valid_bbox(plate_bbox):
        raise ValueError(f"plate_bbox must have positive finite area: {plate_bbox}")

    vehicle_width = float(vehicle_bbox[2] - vehicle_bbox[0])
    vehicle_height = float(vehicle_bbox[3] - vehicle_bbox[1])
    plate_width = float(plate_bbox[2] - plate_bbox[0])
    plate_height = float(plate_bbox[3] - plate_bbox[1])
    plate_center_x = (plate_bbox[0] + plate_bbox[2]) / 2.0
    plate_center_y = (plate_bbox[1] + plate_bbox[3]) / 2.0
    vehicle_area = vehicle_width * vehicle_height
    values = (
        (plate_center_x - vehicle_bbox[0]) / vehicle_width,
        (plate_center_y - vehicle_bbox[1]) / vehicle_height,
        plate_width / vehicle_width,
        plate_height / vehicle_height,
        (plate_width * plate_height) / vehicle_area,
    )
    if not all(isfinite(value) for value in values):
        raise ValueError("relative plate geometry must be finite")
    return RelativePlateGeometry(*values)


def _clamp_score(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


class TemporalPlateOwnershipResolver:
    """Resolve raw per-track candidates while learning bounded track history."""

    def __init__(self, config: TemporalPlateOwnershipConfig | None = None) -> None:
        self.config = config or TemporalPlateOwnershipConfig()
        self._history: dict[int, deque[RelativePlateGeometry]] = {}
        self.last_resolution: TemporalPlateOwnershipResolution | None = None
        self.total_elapsed_seconds = 0.0
        self.total_relative_geometry_seconds = 0.0
        self.total_history_scoring_seconds = 0.0
        self.total_history_updates = 0

    @property
    def history(self) -> dict[int, tuple[RelativePlateGeometry, ...]]:
        """Expose immutable snapshots for debug/tests."""

        return {track_id: tuple(values) for track_id, values in self._history.items()}

    def cleanup(self, active_track_ids: Iterable[int]) -> None:
        """Drop histories for tracks known to be removed."""

        active = {int(track_id) for track_id in active_track_ids}
        for track_id in tuple(self._history):
            if track_id not in active:
                del self._history[track_id]

    def resolve(
        self,
        frame_index: int,
        candidates: Sequence[TrackedPlateCandidate],
    ) -> TemporalPlateOwnershipResolution:
        """Resolve one frame of raw candidates and safely update history."""

        started = time.perf_counter()
        if frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if any(candidate.frame_index != frame_index for candidate in candidates):
            raise ValueError("All plate ownership candidates must match frame_index")

        geometry_started = time.perf_counter()
        states: list[_CandidateState] = []
        invalid_count = 0
        for candidate in candidates:
            if not self._valid_candidate(candidate):
                invalid_count += 1
                continue
            geometry = compute_relative_geometry(
                candidate.plate_bbox, candidate.vehicle_bbox
            )
            states.append(
                _CandidateState(
                    candidate=candidate,
                    geometry=geometry,
                    containment_score=intersection_over_plate_area(
                        candidate.plate_bbox, candidate.vehicle_bbox
                    ),
                    temporal_score=0.0,
                )
            )
        geometry_elapsed = time.perf_counter() - geometry_started

        history_scoring_started = time.perf_counter()
        for state in states:
            (
                state.temporal_score,
                state.history_samples,
                state.history_reliable,
            ) = self._temporal_score(state.candidate.track_id, state.geometry)
        history_scoring_elapsed = time.perf_counter() - history_scoring_started

        groups = self._conflict_groups(states)
        conflict_groups = 0
        definite_groups = 0
        ambiguous_groups = 0
        removed_candidates = 0
        changed_owner_groups = 0
        history_updates = 0

        selected: list[TrackedPlateCandidate] = []
        for group_index, group in enumerate(groups):
            for state in group:
                state.conflict_group = group_index if len(group) > 1 else None
            if len(group) == 1:
                state = group[0]
                state.tight_parent_score = 1.0
                state.owner_score = self._owner_score(state)
                owner = state
            else:
                conflict_groups += 1
                if self._group_is_definite_duplicate(group):
                    definite_groups += 1
                else:
                    ambiguous_groups += 1
                max_relative_area = max(
                    state.geometry.relative_area for state in group
                )
                for state in group:
                    state.tight_parent_score = _clamp_score(
                        state.geometry.relative_area / max_relative_area
                        if max_relative_area > 0.0
                        else 0.0
                    )
                    state.owner_score = self._owner_score(state)
                owner = max(group, key=self._owner_key)
                removed_candidates += len(group) - 1
                if self._legacy_owner(group) != owner.candidate.track_id:
                    changed_owner_groups += 1
            owner.selected = True
            selected.append(owner.candidate)

            if len(group) == 1:
                self._update_history(owner)
                owner.history_updated = True
                history_updates += 1
            else:
                ordered = sorted(group, key=self._owner_key, reverse=True)
                margin = ordered[0].owner_score - ordered[1].owner_score
                if margin >= self.config.min_history_update_margin:
                    self._update_history(owner)
                    owner.history_updated = True
                    history_updates += 1

        selected.sort(key=lambda candidate: candidate.track_id)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.total_elapsed_seconds += elapsed_ms / 1000.0
        self.total_relative_geometry_seconds += geometry_elapsed
        self.total_history_scoring_seconds += history_scoring_elapsed
        self.total_history_updates += history_updates

        resolution = TemporalPlateOwnershipResolution(
            candidates=tuple(selected),
            stats=TemporalPlateOwnershipStats(
                raw_candidates=len(candidates),
                invalid_candidates=invalid_count,
                conflict_groups=conflict_groups,
                definite_duplicate_groups=definite_groups,
                ambiguous_conflict_groups=ambiguous_groups,
                removed_conflict_candidates=removed_candidates,
                final_results=len(selected),
                changed_owner_groups=changed_owner_groups,
                history_updates=history_updates,
                elapsed_ms=elapsed_ms,
                relative_geometry_ms=geometry_elapsed * 1000.0,
                history_scoring_ms=history_scoring_elapsed * 1000.0,
            ),
            diagnostics=tuple(self._diagnostic(state) for state in states),
        )
        self.last_resolution = resolution
        return resolution

    def _valid_candidate(self, candidate: TrackedPlateCandidate) -> bool:
        return (
            _valid_bbox(candidate.plate_bbox)
            and _valid_bbox(candidate.vehicle_bbox)
            and point_inside_bbox(
                (
                    (candidate.plate_bbox[0] + candidate.plate_bbox[2]) / 2.0,
                    (candidate.plate_bbox[1] + candidate.plate_bbox[3]) / 2.0,
                ),
                candidate.vehicle_bbox,
            )
        )

    def _history_for(self, track_id: int) -> deque[RelativePlateGeometry]:
        return self._history.setdefault(
            track_id, deque(maxlen=self.config.history_size)
        )

    def _temporal_score(
        self, track_id: int, geometry: RelativePlateGeometry
    ) -> tuple[float, int, bool]:
        history = self._history.get(track_id)
        samples = len(history) if history is not None else 0
        if samples < self.config.min_history_samples:
            return self.config.cold_start_temporal_score, samples, False

        values = np.asarray(
            [
                (
                    item.center_x,
                    item.center_y,
                    item.relative_width,
                    item.relative_height,
                )
                for item in history
            ],
            dtype=np.float64,
        )
        current = np.asarray(
            [
                geometry.center_x,
                geometry.center_y,
                geometry.relative_width,
                geometry.relative_height,
            ],
            dtype=np.float64,
        )
        median = np.median(values, axis=0)
        mad = np.median(np.abs(values - median), axis=0)
        robust_scale = 1.4826 * mad
        configured_tolerance = np.asarray(
            [
                self.config.center_x_tolerance,
                self.config.center_y_tolerance,
                self.config.relative_width_tolerance,
                self.config.relative_height_tolerance,
            ],
            dtype=np.float64,
        )
        minimum_scale = np.asarray(
            [
                self.config.minimum_center_scale,
                self.config.minimum_center_scale,
                self.config.minimum_relative_size_scale,
                self.config.minimum_relative_size_scale,
            ],
            dtype=np.float64,
        )
        scale = np.maximum(robust_scale, minimum_scale)
        scale = np.maximum(scale, configured_tolerance)
        deviation = float(np.mean(np.abs(current - median) / scale))
        return float(np.exp(-deviation)), samples, True

    def _conflict_groups(
        self, states: Sequence[_CandidateState]
    ) -> list[list[_CandidateState]]:
        parents = list(range(len(states)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(first: int, second: int) -> None:
            first_root = find(first)
            second_root = find(second)
            if first_root != second_root:
                parents[second_root] = first_root

        for first in range(len(states)):
            for second in range(first + 1, len(states)):
                first_candidate = states[first].candidate
                second_candidate = states[second].candidate
                if bbox_iou(
                    first_candidate.plate_bbox, second_candidate.plate_bbox
                ) >= self.config.ownership_conflict_iou_threshold:
                    union(first, second)
                    continue
                overlap = _intersection_area(
                    first_candidate.plate_bbox, second_candidate.plate_bbox
                )
                first_area = (
                    first_candidate.plate_bbox[2] - first_candidate.plate_bbox[0]
                ) * (
                    first_candidate.plate_bbox[3] - first_candidate.plate_bbox[1]
                )
                second_area = (
                    second_candidate.plate_bbox[2] - second_candidate.plate_bbox[0]
                ) * (
                    second_candidate.plate_bbox[3] - second_candidate.plate_bbox[1]
                )
                smaller_area = min(first_area, second_area)
                if (
                    smaller_area > 0.0
                    and overlap / smaller_area
                    >= self.config.overlap_over_smaller_threshold
                ):
                    union(first, second)

        groups: dict[int, list[_CandidateState]] = {}
        for index, state in enumerate(states):
            groups.setdefault(find(index), []).append(state)
        return [groups[root] for root in sorted(groups)]

    def _group_is_definite_duplicate(self, group: Sequence[_CandidateState]) -> bool:
        for first in range(len(group)):
            for second in range(first + 1, len(group)):
                if (
                    bbox_iou(
                        group[first].candidate.plate_bbox,
                        group[second].candidate.plate_bbox,
                    )
                    >= self.config.definite_duplicate_iou_threshold
                ):
                    return True
        return False

    def _owner_score(self, state: _CandidateState) -> float:
        config = self.config
        return (
            config.temporal_weight * state.temporal_score
            + config.tight_parent_weight * state.tight_parent_score
            + config.containment_weight * state.containment_score
            + config.plate_conf_weight * _clamp_score(
                state.candidate.plate_confidence
            )
            + config.vehicle_conf_weight * _clamp_score(
                state.candidate.vehicle_confidence
            )
        )

    @staticmethod
    def _owner_key(state: _CandidateState) -> tuple[float, float, float, float, float, int]:
        return (
            state.owner_score,
            state.temporal_score,
            state.tight_parent_score,
            state.candidate.plate_confidence,
            state.candidate.vehicle_confidence,
            -state.candidate.track_id,
        )

    @staticmethod
    def _legacy_owner(group: Sequence[_CandidateState]) -> int:
        return max(
            group,
            key=lambda state: (
                state.containment_score,
                state.candidate.plate_confidence,
                state.candidate.vehicle_confidence,
                -state.candidate.track_id,
            ),
        ).candidate.track_id

    def _update_history(self, state: _CandidateState) -> None:
        self._history_for(state.candidate.track_id).append(state.geometry)

    def _diagnostic(self, state: _CandidateState) -> TemporalCandidateDiagnostic:
        return TemporalCandidateDiagnostic(
            track_id=state.candidate.track_id,
            plate_bbox=state.candidate.plate_bbox,
            vehicle_bbox=state.candidate.vehicle_bbox,
            relative_geometry=state.geometry,
            containment_score=state.containment_score,
            temporal_score=state.temporal_score,
            tight_parent_score=state.tight_parent_score,
            owner_score=state.owner_score,
            history_samples=state.history_samples,
            history_reliable=state.history_reliable,
            conflict_group=state.conflict_group,
            selected=state.selected,
            history_updated=state.history_updated,
        )
