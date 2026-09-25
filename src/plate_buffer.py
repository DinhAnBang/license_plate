"""Memory-bounded per-track temporal plate candidate buffers for V4."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .plate_quality import PlateQualityMetrics


@dataclass(frozen=True, slots=True)
class PlateBufferConfig:
    top_k: int = 4
    min_frame_gap: int = 2
    min_quality_score: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.top_k, int) or self.top_k <= 0:
            raise ValueError("top_k must be an integer greater than zero")
        if not isinstance(self.min_frame_gap, int) or self.min_frame_gap < 0:
            raise ValueError("min_frame_gap must be a non-negative integer")
        if not 0.0 <= self.min_quality_score <= 1.0:
            raise ValueError("min_quality_score must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class BufferedPlateCandidate:
    frame_index: int
    track_id: int
    plate_class_id: int
    plate_class_name: str
    plate_confidence: float
    bbox: tuple[int, int, int, int]
    quality: PlateQualityMetrics
    crop: np.ndarray

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if not 0.0 <= self.plate_confidence <= 1.0:
            raise ValueError("plate_confidence must be in [0, 1]")
        if len(self.bbox) != 4 or self.bbox[2] <= self.bbox[0] or self.bbox[3] <= self.bbox[1]:
            raise ValueError("bbox must be a valid xyxy box")
        if (
            not isinstance(self.crop, np.ndarray)
            or self.crop.ndim != 3
            or self.crop.shape[2] != 3
            or self.crop.size == 0
        ):
            raise ValueError("crop must be a non-empty HxWx3 NumPy array")
        object.__setattr__(self, "crop", np.ascontiguousarray(self.crop.copy()))


@dataclass(frozen=True, slots=True)
class TrackPlateSummary:
    track_id: int
    vehicle_observation_count: int
    plate_observation_count: int
    valid_plate_crop_count: int
    invalid_plate_crop_count: int
    top_k_count: int
    first_plate_frame: int | None
    last_plate_frame: int | None
    plate_detection_ratio: float
    vuong_count: int
    dai_count: int
    layout_votes: dict[str, float]
    dominant_plate_class: str | None


@dataclass(slots=True)
class _TrackState:
    retained: list[BufferedPlateCandidate] = field(default_factory=list)
    vehicle_observations: int = 0
    plate_observations: int = 0
    valid_crops: int = 0
    invalid_crops: int = 0
    first_plate_frame: int | None = None
    last_plate_frame: int | None = None
    layout_counts: dict[str, int] = field(default_factory=lambda: {"vuong": 0, "dai": 0})
    layout_votes: dict[str, float] = field(default_factory=lambda: {"vuong": 0.0, "dai": 0.0})


class PlateBufferManager:
    """Collect evidence while retaining only deterministic temporal Top-K crops.

    With ``min_frame_gap=2``, candidates one frame apart conflict because their
    absolute difference is less than 2; candidates two frames apart may coexist.
    """

    def __init__(self, config: PlateBufferConfig | None = None) -> None:
        self.config = config or PlateBufferConfig()
        self._tracks: dict[int, _TrackState] = {}

    def _state(self, track_id: int) -> _TrackState:
        return self._tracks.setdefault(int(track_id), _TrackState())

    @staticmethod
    def _rank(candidate: BufferedPlateCandidate) -> tuple[float, float, float, int]:
        return (
            candidate.quality.total_score,
            candidate.plate_confidence,
            candidate.quality.sharpness_raw,
            -candidate.frame_index,
        )

    @staticmethod
    def _record_plate_metadata(
        state: _TrackState, frame_index: int, plate_class_name: str
    ) -> None:
        state.plate_observations += 1
        state.first_plate_frame = (
            frame_index
            if state.first_plate_frame is None
            else min(state.first_plate_frame, frame_index)
        )
        state.last_plate_frame = (
            frame_index
            if state.last_plate_frame is None
            else max(state.last_plate_frame, frame_index)
        )
        state.layout_counts[plate_class_name] = state.layout_counts.get(plate_class_name, 0) + 1

    def mark_vehicle_seen(self, track_id: int, frame_index: int) -> None:
        if frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        self._state(track_id).vehicle_observations += 1

    def record_invalid_plate(
        self, track_id: int, frame_index: int, plate_class_name: str
    ) -> None:
        state = self._state(track_id)
        self._record_plate_metadata(state, frame_index, plate_class_name)
        state.invalid_crops += 1

    def add_plate(self, candidate: BufferedPlateCandidate) -> bool:
        """Record a resolved event and return whether the new crop remains retained."""

        state = self._state(candidate.track_id)
        self._record_plate_metadata(state, candidate.frame_index, candidate.plate_class_name)
        state.valid_crops += 1
        state.layout_votes[candidate.plate_class_name] = (
            state.layout_votes.get(candidate.plate_class_name, 0.0)
            + candidate.quality.total_score * candidate.plate_confidence
        )
        if candidate.quality.total_score < self.config.min_quality_score:
            return False

        # The manager owns its retained image memory. This second copy keeps a
        # caller from mutating a candidate's ndarray after insertion.
        stored_candidate = BufferedPlateCandidate(
            frame_index=candidate.frame_index,
            track_id=candidate.track_id,
            plate_class_id=candidate.plate_class_id,
            plate_class_name=candidate.plate_class_name,
            plate_confidence=candidate.plate_confidence,
            bbox=candidate.bbox,
            quality=candidate.quality,
            crop=candidate.crop,
        )

        conflicts = [
            existing
            for existing in state.retained
            if abs(stored_candidate.frame_index - existing.frame_index)
            < self.config.min_frame_gap
        ]
        if conflicts:
            winner = max([stored_candidate, *conflicts], key=self._rank)
            conflict_ids = {id(item) for item in conflicts}
            state.retained = [
                item for item in state.retained if id(item) not in conflict_ids
            ]
            state.retained.append(winner)
        else:
            state.retained.append(stored_candidate)

        state.retained.sort(key=self._rank, reverse=True)
        if len(state.retained) > self.config.top_k:
            del state.retained[self.config.top_k :]
        return any(item is stored_candidate for item in state.retained)

    def get_top_candidates(self, track_id: int) -> tuple[BufferedPlateCandidate, ...]:
        state = self._tracks.get(int(track_id))
        return tuple(state.retained) if state is not None else ()

    def get_track_summary(self, track_id: int) -> TrackPlateSummary:
        state = self._tracks.get(int(track_id), _TrackState())
        ratio = (
            state.plate_observations / state.vehicle_observations
            if state.vehicle_observations
            else 0.0
        )
        positive_votes = {name: vote for name, vote in state.layout_votes.items() if vote > 0.0}
        dominant = (
            max(positive_votes, key=lambda name: (positive_votes[name], name))
            if positive_votes
            else None
        )
        return TrackPlateSummary(
            track_id=int(track_id),
            vehicle_observation_count=state.vehicle_observations,
            plate_observation_count=state.plate_observations,
            valid_plate_crop_count=state.valid_crops,
            invalid_plate_crop_count=state.invalid_crops,
            top_k_count=len(state.retained),
            first_plate_frame=state.first_plate_frame,
            last_plate_frame=state.last_plate_frame,
            plate_detection_ratio=float(ratio),
            vuong_count=state.layout_counts.get("vuong", 0),
            dai_count=state.layout_counts.get("dai", 0),
            layout_votes=dict(state.layout_votes),
            dominant_plate_class=dominant,
        )

    def finalize_all(self) -> tuple[TrackPlateSummary, ...]:
        return tuple(self.get_track_summary(track_id) for track_id in sorted(self._tracks))

    @property
    def total_retained_candidates(self) -> int:
        return sum(len(state.retained) for state in self._tracks.values())
