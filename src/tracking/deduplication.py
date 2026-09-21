"""Conservative detector cleanup specific to cross-class vehicle duplicates.

Per-class NMS remains the detector's responsibility.  This module only
removes a detection when a *different-class* detection has near-identical
geometry.  It is a project-specific hardening step, not part of standard
ByteTrack association.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.vehicle_detector import VehicleDetection

from .matching import iou_matrix


@dataclass(frozen=True, slots=True)
class DetectionDeduplicationResult:
    detections: tuple[VehicleDetection, ...]
    duplicate_groups: int
    detections_removed: int


def deduplicate_vehicle_detections(
    detections: Sequence[VehicleDetection],
    *,
    iou_threshold: float = 0.90,
    area_ratio_threshold: float = 0.80,
    center_distance_threshold: float | None = None,
) -> DetectionDeduplicationResult:
    """Remove conservative cross-class duplicate detection groups.

    Candidate edges are only formed between different classes and require the
    configured IoU plus area-ratio checks.  Connected components are then
    resolved as groups so a three-way chain cannot leave a duplicate behind.
    The highest-confidence member wins; input order is the deterministic tie
    breaker.
    """

    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    if not 0.0 < area_ratio_threshold <= 1.0:
        raise ValueError("area_ratio_threshold must be in (0, 1]")
    if center_distance_threshold is not None and center_distance_threshold < 0:
        raise ValueError("center_distance_threshold cannot be negative")

    items = list(detections)
    if len(items) < 2:
        return DetectionDeduplicationResult(tuple(items), 0, 0)

    similarities = iou_matrix(items, items)
    boxes = np.asarray([item.bbox for item in items], dtype=np.float64)
    widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    areas = widths * heights
    centers = np.column_stack(
        ((boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0)
    )

    parent = list(range(len(items)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for first in range(len(items)):
        for second in range(first + 1, len(items)):
            if items[first].class_id == items[second].class_id:
                continue
            if similarities[first, second] < iou_threshold:
                continue
            max_area = max(areas[first], areas[second])
            if max_area <= 0.0:
                continue
            if min(areas[first], areas[second]) / max_area < area_ratio_threshold:
                continue
            if center_distance_threshold is not None:
                center_distance = float(np.linalg.norm(centers[first] - centers[second]))
                normalization = max(
                    widths[first], heights[first], widths[second], heights[second], 1.0
                )
                if center_distance / normalization > center_distance_threshold:
                    continue
            union(first, second)

    groups: dict[int, list[int]] = {}
    for index in range(len(items)):
        groups.setdefault(find(index), []).append(index)

    kept = [True] * len(items)
    duplicate_groups = 0
    detections_removed = 0
    for group in groups.values():
        if len(group) < 2:
            continue
        winner = min(
            group,
            key=lambda index: (-float(items[index].confidence), index),
        )
        duplicate_groups += 1
        for index in group:
            if index == winner:
                continue
            kept[index] = False
            detections_removed += 1

    return DetectionDeduplicationResult(
        tuple(item for index, item in enumerate(items) if kept[index]),
        duplicate_groups,
        detections_removed,
    )
