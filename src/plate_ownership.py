"""Deterministic per-frame ownership resolution for duplicate plate boxes."""

from __future__ import annotations

import time
from dataclasses import dataclass
from math import isfinite

from .plate_detector import TrackedPlateCandidate


PLATE_DUPLICATE_IOU_THRESHOLD = 0.80
BBox = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class PlateOwnershipStats:
    raw_candidates: int
    invalid_candidates: int
    duplicate_groups: int
    removed_duplicate_candidates: int
    final_results: int
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class PlateOwnershipResolution:
    candidates: tuple[TrackedPlateCandidate, ...]
    stats: PlateOwnershipStats


def _valid_bbox(bbox: BBox) -> bool:
    if len(bbox) != 4 or not all(isfinite(value) for value in bbox):
        return False
    x1, y1, x2, y2 = bbox
    return x2 > x1 and y2 > y1


def bbox_center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def point_inside_bbox(point: tuple[float, float], bbox: BBox) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def _intersection_area(box_a: BBox, box_b: BBox) -> float:
    intersection_width = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    intersection_height = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    return intersection_width * intersection_height


def bbox_iou(box_a: BBox, box_b: BBox) -> float:
    if not _valid_bbox(box_a) or not _valid_bbox(box_b):
        return 0.0
    intersection = _intersection_area(box_a, box_b)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def intersection_over_plate_area(plate_bbox: BBox, vehicle_bbox: BBox) -> float:
    if not _valid_bbox(plate_bbox) or not _valid_bbox(vehicle_bbox):
        return 0.0
    plate_area = (plate_bbox[2] - plate_bbox[0]) * (
        plate_bbox[3] - plate_bbox[1]
    )
    if plate_area <= 0:
        return 0.0
    return _intersection_area(plate_bbox, vehicle_bbox) / plate_area


def _valid_candidate(candidate: TrackedPlateCandidate) -> bool:
    return (
        _valid_bbox(candidate.plate_bbox)
        and _valid_bbox(candidate.vehicle_bbox)
        and point_inside_bbox(
            bbox_center(candidate.plate_bbox), candidate.vehicle_bbox
        )
    )


def _owner_key(candidate: TrackedPlateCandidate) -> tuple[float, float, float, int]:
    return (
        intersection_over_plate_area(candidate.plate_bbox, candidate.vehicle_bbox),
        candidate.plate_confidence,
        candidate.vehicle_confidence,
        -candidate.track_id,
    )


def resolve_plate_ownership(
    candidates: list[TrackedPlateCandidate],
    duplicate_iou_threshold: float = PLATE_DUPLICATE_IOU_THRESHOLD,
    debug: bool = False,
) -> list[TrackedPlateCandidate]:
    """Return valid, unique physical plates for one frame."""

    return list(
        resolve_plate_ownership_detailed(
            candidates,
            duplicate_iou_threshold=duplicate_iou_threshold,
            debug=debug,
        ).candidates
    )


def resolve_plate_ownership_detailed(
    candidates: list[TrackedPlateCandidate],
    duplicate_iou_threshold: float = PLATE_DUPLICATE_IOU_THRESHOLD,
    debug: bool = False,
) -> PlateOwnershipResolution:
    """Resolve duplicate groups and return candidates plus debug statistics."""

    started = time.perf_counter()
    if not 0.0 < duplicate_iou_threshold <= 1.0:
        raise ValueError("duplicate_iou_threshold must be in (0, 1]")
    if not candidates:
        return PlateOwnershipResolution(
            candidates=(),
            stats=PlateOwnershipStats(0, 0, 0, 0, 0, 0.0),
        )

    frame_index = candidates[0].frame_index
    if any(candidate.frame_index != frame_index for candidate in candidates):
        raise ValueError("All plate ownership candidates must belong to one frame")

    valid_candidates = [candidate for candidate in candidates if _valid_candidate(candidate)]
    invalid_count = len(candidates) - len(valid_candidates)
    candidate_count = len(valid_candidates)
    parents = list(range(candidate_count))

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

    for first in range(candidate_count):
        for second in range(first + 1, candidate_count):
            if (
                bbox_iou(
                    valid_candidates[first].plate_bbox,
                    valid_candidates[second].plate_bbox,
                )
                >= duplicate_iou_threshold
            ):
                union(first, second)

    groups: dict[int, list[TrackedPlateCandidate]] = {}
    for index, candidate in enumerate(valid_candidates):
        groups.setdefault(find(index), []).append(candidate)

    selected: list[TrackedPlateCandidate] = []
    duplicate_groups = 0
    removed_duplicates = 0
    for group in groups.values():
        owner = max(group, key=_owner_key)
        selected.append(owner)
        if len(group) > 1:
            duplicate_groups += 1
            removed_duplicates += len(group) - 1
            if debug:
                print(f"Frame {frame_index}")
                print("Plate ownership group:")
                print(f"candidates = {len(group)}")
                for candidate in sorted(group, key=lambda item: item.track_id):
                    containment = intersection_over_plate_area(
                        candidate.plate_bbox, candidate.vehicle_bbox
                    )
                    print("candidate:")
                    print(f"  track={candidate.track_id}")
                    print(f"  plate_conf={candidate.plate_confidence:.4f}")
                    print(f"  containment={containment:.3f}")
                    print(f"  bbox={list(candidate.plate_bbox)}")
                print("selected owner:")
                print(f"  track={owner.track_id}")

    selected.sort(key=lambda candidate: candidate.track_id)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return PlateOwnershipResolution(
        candidates=tuple(selected),
        stats=PlateOwnershipStats(
            raw_candidates=len(candidates),
            invalid_candidates=invalid_count,
            duplicate_groups=duplicate_groups,
            removed_duplicate_candidates=removed_duplicates,
            final_results=len(selected),
            elapsed_ms=elapsed_ms,
        ),
    )
