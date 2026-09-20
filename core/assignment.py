"""IoU and dependency-free linear assignment utilities for SORT."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def iou_matrix(
    track_boxes: Sequence[Sequence[float]] | np.ndarray,
    detection_boxes: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Return pairwise IoU with shape ``(tracks, detections)``.

    Invalid and zero-area boxes have zero overlap. Empty inputs retain the
    expected rectangular shape.
    """

    tracks = _box_array(track_boxes)
    detections = _box_array(detection_boxes)
    overlaps = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    if not len(tracks) or not len(detections):
        return overlaps

    valid_tracks = np.all(np.isfinite(tracks), axis=1)
    valid_detections = np.all(np.isfinite(detections), axis=1)
    track_widths = tracks[:, 2] - tracks[:, 0]
    track_heights = tracks[:, 3] - tracks[:, 1]
    detection_widths = detections[:, 2] - detections[:, 0]
    detection_heights = detections[:, 3] - detections[:, 1]
    valid_tracks &= (track_widths > 0.0) & (track_heights > 0.0)
    valid_detections &= (detection_widths > 0.0) & (detection_heights > 0.0)

    left = np.maximum(tracks[:, None, 0], detections[None, :, 0])
    top = np.maximum(tracks[:, None, 1], detections[None, :, 1])
    right = np.minimum(tracks[:, None, 2], detections[None, :, 2])
    bottom = np.minimum(tracks[:, None, 3], detections[None, :, 3])
    intersection = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
    track_areas = np.maximum(0.0, track_widths * track_heights)
    detection_areas = np.maximum(0.0, detection_widths * detection_heights)
    union = track_areas[:, None] + detection_areas[None, :] - intersection
    valid_pairs = valid_tracks[:, None] & valid_detections[None, :] & (union > 0.0)
    np.divide(intersection, union, out=overlaps, where=valid_pairs)
    return np.clip(overlaps, 0.0, 1.0)


def linear_assignment(cost_matrix: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Solve rectangular minimum-cost assignment with the Hungarian method.

    The implementation uses the shortest augmenting-path form of the
    Hungarian algorithm and has no SciPy dependency. The result is a sorted
    ``(row, column)`` integer array with ``min(rows, columns)`` assignments.
    """

    costs = np.asarray(cost_matrix, dtype=np.float64)
    if costs.ndim != 2:
        raise ValueError("cost_matrix must be a two-dimensional matrix")
    rows, columns = costs.shape
    if rows == 0 or columns == 0:
        return np.empty((0, 2), dtype=np.int64)

    finite = costs[np.isfinite(costs)]
    if finite.size:
        scale = max(1.0, float(np.max(np.abs(finite))))
        replacement = scale * (rows + columns + 1) * 1_000_000.0
    else:
        replacement = 1_000_000.0
    safe_costs = np.where(np.isfinite(costs), costs, replacement)

    transposed = rows > columns
    working = safe_costs.T if transposed else safe_costs
    assignments = _hungarian_rows_le_columns(working)
    if transposed:
        assignments = assignments[:, [1, 0]]
    order = np.lexsort((assignments[:, 1], assignments[:, 0]))
    return assignments[order]


def gated_assignment(
    overlaps: np.ndarray,
    threshold: float,
) -> list[tuple[int, int, float]]:
    """Return Hungarian matches whose overlap clears ``threshold``."""

    if overlaps.ndim != 2:
        raise ValueError("overlaps must be a two-dimensional matrix")
    if overlaps.shape[0] == 0 or overlaps.shape[1] == 0:
        return []
    assignments = linear_assignment(1.0 - overlaps)
    return [
        (
            int(track_index),
            int(detection_index),
            float(overlaps[int(track_index), int(detection_index)]),
        )
        for track_index, detection_index in assignments
        if overlaps[int(track_index), int(detection_index)] >= threshold
    ]


def _hungarian_rows_le_columns(costs: np.ndarray) -> np.ndarray:
    """Hungarian solver for a finite matrix where rows <= columns."""

    row_count, column_count = costs.shape
    if row_count > column_count:
        raise ValueError("internal Hungarian matrix must have rows <= columns")

    # One-based arrays follow the standard augmenting-path formulation.
    row_potential = np.zeros(row_count + 1, dtype=np.float64)
    column_potential = np.zeros(column_count + 1, dtype=np.float64)
    column_match = np.zeros(column_count + 1, dtype=np.int64)
    predecessor = np.zeros(column_count + 1, dtype=np.int64)

    for row in range(1, row_count + 1):
        column_match[0] = row
        current_column = 0
        minimum = np.full(column_count + 1, np.inf, dtype=np.float64)
        used = np.zeros(column_count + 1, dtype=bool)
        while True:
            used[current_column] = True
            current_row = int(column_match[current_column])
            delta = np.inf
            next_column = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                reduced_cost = (
                    costs[current_row - 1, column - 1]
                    - row_potential[current_row]
                    - column_potential[column]
                )
                if reduced_cost < minimum[column]:
                    minimum[column] = reduced_cost
                    predecessor[column] = current_column
                if minimum[column] < delta:
                    delta = minimum[column]
                    next_column = column

            for column in range(column_count + 1):
                if used[column]:
                    row_potential[column_match[column]] += delta
                    column_potential[column] -= delta
                else:
                    minimum[column] -= delta
            current_column = next_column
            if column_match[current_column] == 0:
                break

        while True:
            previous_column = int(predecessor[current_column])
            column_match[current_column] = column_match[previous_column]
            current_column = previous_column
            if current_column == 0:
                break

    pairs = [
        (int(column_match[column]) - 1, column - 1)
        for column in range(1, column_count + 1)
        if column_match[column] != 0
    ]
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def _box_array(boxes: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    array = np.asarray(boxes, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4)")
    return array
