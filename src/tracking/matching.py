"""IoU costs, score fusion, and Hungarian assignment.

The legacy ``apply_class_gate`` helper remains available for compatibility,
but the V2.1 tracker core deliberately does not call it: class is an
attribute, not an identity constraint.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from scipy.optimize import linear_sum_assignment


INF_COST = 1_000_000.0


class _BoxItem(Protocol):
    @property
    def bbox(self) -> tuple[int, int, int, int] | np.ndarray: ...


class _ClassItem(Protocol):
    class_id: int


class _ScoredItem(Protocol):
    confidence: float


def _as_boxes(items: Sequence[_BoxItem] | np.ndarray) -> np.ndarray:
    if isinstance(items, np.ndarray):
        boxes = np.asarray(items, dtype=np.float64)
    else:
        boxes = np.asarray([item.bbox for item in items], dtype=np.float64)
    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"Expected boxes with shape (N, 4), got {boxes.shape}")
    return boxes


def iou_matrix(
    first: Sequence[_BoxItem] | np.ndarray,
    second: Sequence[_BoxItem] | np.ndarray,
) -> np.ndarray:
    """Return all pairwise IoU similarities for two ``xyxy`` box sets."""

    first_boxes = _as_boxes(first)
    second_boxes = _as_boxes(second)
    if len(first_boxes) == 0 or len(second_boxes) == 0:
        return np.zeros((len(first_boxes), len(second_boxes)), dtype=np.float64)

    top_left = np.maximum(first_boxes[:, None, :2], second_boxes[None, :, :2])
    bottom_right = np.minimum(first_boxes[:, None, 2:], second_boxes[None, :, 2:])
    intersection_size = np.maximum(0.0, bottom_right - top_left)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]

    first_size = np.maximum(0.0, first_boxes[:, 2:] - first_boxes[:, :2])
    second_size = np.maximum(0.0, second_boxes[:, 2:] - second_boxes[:, :2])
    first_area = first_size[:, 0] * first_size[:, 1]
    second_area = second_size[:, 0] * second_size[:, 1]
    union = first_area[:, None] + second_area[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def iou_distance(
    tracks: Sequence[_BoxItem] | np.ndarray,
    detections: Sequence[_BoxItem] | np.ndarray,
) -> np.ndarray:
    """Return IoU cost, defined as ``1 - IoU similarity``."""

    return 1.0 - iou_matrix(tracks, detections)


def apply_class_gate(
    cost_matrix: np.ndarray,
    tracks: Sequence[_ClassItem],
    detections: Sequence[_ClassItem],
) -> np.ndarray:
    """Apply the legacy explicit class gate for compatibility-only callers."""

    costs = np.asarray(cost_matrix, dtype=np.float64).copy()
    if costs.shape != (len(tracks), len(detections)):
        raise ValueError(
            "Cost matrix shape does not match tracks/detections: "
            f"{costs.shape} != {(len(tracks), len(detections))}"
        )
    if costs.size == 0:
        return costs
    track_classes = np.asarray([track.class_id for track in tracks])
    detection_classes = np.asarray(
        [detection.class_id for detection in detections]
    )
    costs[track_classes[:, None] != detection_classes[None, :]] = INF_COST
    return costs


def fuse_detection_scores(
    cost_matrix: np.ndarray, detections: Sequence[_ScoredItem]
) -> np.ndarray:
    """Fuse HIGH-detection confidence with IoU similarity."""

    costs = np.asarray(cost_matrix, dtype=np.float64)
    if costs.shape[1] != len(detections):
        raise ValueError("Detection count does not match cost-matrix columns")
    if costs.size == 0:
        return costs.copy()
    scores = np.asarray([detection.confidence for detection in detections])
    fused_similarity = (1.0 - costs) * scores[None, :]
    return 1.0 - fused_similarity


def linear_assignment(
    cost_matrix: np.ndarray, cost_limit: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Solve a thresholded assignment exactly with the Hungarian algorithm.

    ``cost_limit`` is a maximum cost, not a minimum IoU. An augmented matrix
    gives every row and column an explicit unmatched option, so invalid pairs
    cannot distort the feasible assignment.
    """

    costs = np.asarray(cost_matrix, dtype=np.float64)
    if costs.ndim != 2:
        raise ValueError(f"Expected a 2D cost matrix, got {costs.shape}")
    if not np.isfinite(cost_limit) or cost_limit < 0:
        raise ValueError("cost_limit must be a finite non-negative number")

    row_count, column_count = costs.shape
    if row_count == 0 or column_count == 0:
        return [], list(range(row_count)), list(range(column_count))

    size = row_count + column_count
    augmented = np.full((size, size), INF_COST, dtype=np.float64)
    valid = np.isfinite(costs) & (costs <= cost_limit)
    augmented[:row_count, :column_count] = np.where(
        valid, costs, INF_COST
    )

    # A real edge at exactly the limit remains marginally preferable to two
    # unmatched edges. This also maximizes feasible match cardinality.
    unmatched_cost = (cost_limit + 1e-6) / 2.0
    augmented[
        np.arange(row_count), column_count + np.arange(row_count)
    ] = unmatched_cost
    augmented[
        row_count + np.arange(column_count), np.arange(column_count)
    ] = unmatched_cost
    augmented[row_count:, column_count:] = 0.0

    assigned_rows, assigned_columns = linear_sum_assignment(augmented)
    matches = sorted(
        (int(row), int(column))
        for row, column in zip(assigned_rows, assigned_columns, strict=True)
        if row < row_count
        and column < column_count
        and valid[row, column]
    )
    matched_rows = {row for row, _ in matches}
    matched_columns = {column for _, column in matches}
    unmatched_rows = [
        index for index in range(row_count) if index not in matched_rows
    ]
    unmatched_columns = [
        index for index in range(column_count) if index not in matched_columns
    ]
    return matches, unmatched_rows, unmatched_columns
