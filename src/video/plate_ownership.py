"""Video-only plate ownership and cross-track conflict resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from ..config import VideoPlateOwnershipConfig
from ..core.plate_geometry import (
    bbox_inside, bbox_iou as _bbox_iou,
    overlap_over_smaller as _overlap_over_smaller, valid_bbox,
)
if TYPE_CHECKING:
    from .plate_stage import VideoPlateCandidate


@dataclass(frozen=True, slots=True)
class VideoPlateDecision:
    candidate: VideoPlateCandidate
    ownership_status: str
    conflict_group: int | None
    ownership_score: float


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
    conflict_iou_threshold: float | None = None,
    overlap_over_smaller_threshold: float | None = None,
) -> tuple[VideoPlateDecision, ...]:
    """Mark cross-track plate conflicts without discarding raw candidates.

    Preserve every raw candidate in the decisions, but accept at most one
    plate per track and frame after resolving cross-track conflicts.
    """

    if conflict_iou_threshold is None or overlap_over_smaller_threshold is None:
        defaults = VideoPlateOwnershipConfig()
        if conflict_iou_threshold is None:
            conflict_iou_threshold = defaults.conflict_iou_threshold
        if overlap_over_smaller_threshold is None:
            overlap_over_smaller_threshold = defaults.overlap_over_smaller_threshold
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
        and valid_bbox(candidate.plate_bbox)
        and bbox_inside(candidate.plate_bbox, candidate.vehicle_bbox)
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


__all__ = ["VideoPlateDecision", "resolve_frame_plate_ownership"]
