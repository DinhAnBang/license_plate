"""Rebuildable Top-K evidence buffer for tracked video vehicles."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..config import VideoPlateBufferConfig
from ..core.plate_quality import VideoPlateQualityMetrics
if TYPE_CHECKING:
    from .plate_stage import VideoPlateCandidate




@dataclass(frozen=True, slots=True)
class VideoBufferedPlate:
    candidate: VideoPlateCandidate
    quality: VideoPlateQualityMetrics
    crop: np.ndarray

    def __post_init__(self) -> None:
        if (
            not isinstance(self.crop, np.ndarray)
            or self.crop.ndim != 3
            or self.crop.shape[2] != 3
            or self.crop.size == 0
        ):
            raise ValueError("crop must be a non-empty HxWx3 image")
        object.__setattr__(self, "crop", np.ascontiguousarray(self.crop.copy()))


class VideoPlateBuffer:
    """Keep all evidence and rebuild each track's Top-K selection."""

    def __init__(self, config: VideoPlateBufferConfig | None = None) -> None:
        self.config = config or VideoPlateBufferConfig()
        self._all: dict[int, list[VideoBufferedPlate]] = {}
        self._eligible: dict[int, list[VideoBufferedPlate]] = {}
        self._selected: dict[int, tuple[VideoBufferedPlate, ...]] = {}

    @property
    def track_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._all))

    def add(self, item: VideoBufferedPlate) -> None:
        track_id = item.candidate.track_id
        previous = self._all.get(track_id, ())
        if previous and item.candidate.frame_index < previous[-1].candidate.frame_index:
            raise ValueError("out-of-order plate evidence for track")
        self._all.setdefault(track_id, []).append(item)
        if self.rejection_reason(item) is None:
            self._eligible.setdefault(track_id, []).append(item)
        self._selected[track_id] = self._rebuild(track_id)

    def all_candidates(self, track_id: int) -> tuple[VideoBufferedPlate, ...]:
        return tuple(self._all.get(track_id, ()))

    def selected(self, track_id: int) -> tuple[VideoBufferedPlate, ...]:
        return self._selected.get(track_id, ())

    def eligible_count(self, track_id: int) -> int:
        return len(self._eligible.get(track_id, ()))

    def rejection_reason(self, item: VideoBufferedPlate) -> str | None:
        if item.quality.width < self.config.min_crop_width:
            return "crop_too_narrow"
        if item.quality.height < self.config.min_crop_height:
            return "crop_too_short"
        if item.quality.sharpness_score < self.config.min_sharpness_score:
            return "sharpness_too_low"
        if item.quality.total_score < self.config.min_quality_score:
            return "quality_score_too_low"
        return None

    def _ranked(self, track_id: int) -> list[VideoBufferedPlate]:
        return sorted(
            self._eligible.get(track_id, ()),
            key=lambda item: (
                item.quality.total_score,
                item.quality.sharpness_score,
                item.candidate.plate_confidence,
                item.candidate.frame_index,
                item.candidate.candidate_index,
            ),
            reverse=True,
        )

    def _rebuild(self, track_id: int) -> tuple[VideoBufferedPlate, ...]:
        # Weighted interval scheduling with a K limit. A greedy choice of a
        # centre frame can block two good, well-separated side frames.
        items = sorted(
            self._eligible.get(track_id, ()),
            key=lambda item: (item.candidate.frame_index, item.candidate.candidate_index),
        )
        if not items:
            return ()
        frames = [item.candidate.frame_index for item in items]
        previous = [
            bisect_right(frames, frame - self.config.min_frame_gap, 0, index)
            for index, frame in enumerate(frames)
        ]
        rank = lambda index: (
            items[index].quality.total_score,
            items[index].quality.sharpness_score,
            items[index].candidate.plate_confidence,
            items[index].candidate.frame_index,
            items[index].candidate.candidate_index,
        )

        def objective(indices: tuple[int, ...]) -> tuple:
            return (
                sum(items[index].quality.total_score for index in indices),
                tuple(sorted((rank(index) for index in indices), reverse=True)),
            )

        k_limit = min(self.config.top_k, len(items))
        dp: list[list[tuple[int, ...] | None]] = [
            [None] * (k_limit + 1) for _ in range(len(items) + 1)
        ]
        for row in dp:
            row[0] = ()
        for count in range(1, len(items) + 1):
            for k in range(1, k_limit + 1):
                skip = dp[count - 1][k]
                prior = dp[previous[count - 1]][k - 1]
                take = prior + (count - 1,) if prior is not None else None
                dp[count][k] = max(
                    (choice for choice in (skip, take) if choice is not None),
                    key=objective, default=None,
                )
        best = max(
            (choice for choice in dp[len(items)] if choice is not None),
            key=objective,
        )
        return tuple(items[index] for index in sorted(best, key=rank, reverse=True))


__all__ = [
    "VideoBufferedPlate",
    "VideoPlateBuffer",
    "VideoPlateBufferConfig",
]
