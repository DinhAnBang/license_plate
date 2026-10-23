"""Minimum-cost assignment for the small detection matrices used by ALPR."""

from __future__ import annotations

import numpy as np


def linear_sum_assignment(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a minimum-cost assignment for a finite rectangular matrix.

    This is the shortest augmenting path Hungarian algorithm. Detection and
    tracking call it with square matrices that include unmatched choices.
    """

    costs = np.asarray(cost_matrix, dtype=np.float64)
    if costs.ndim != 2:
        raise ValueError("Expected a 2D cost matrix")
    if not np.isfinite(costs).all():
        raise ValueError("Cost matrix must contain only finite values")

    rows, columns = costs.shape
    if rows == 0 or columns == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy()

    transposed = rows > columns
    if transposed:
        costs = costs.T
        rows, columns = costs.shape
    values = costs.tolist()

    row_potential = [0.0] * (rows + 1)
    column_potential = [0.0] * (columns + 1)
    assigned_row = [0] * (columns + 1)
    previous_column = [0] * (columns + 1)

    for row in range(1, rows + 1):
        assigned_row[0] = row
        column = 0
        best = [float("inf")] * (columns + 1)
        visited = [False] * (columns + 1)
        while True:
            visited[column] = True
            current_row = assigned_row[column]
            improvement = float("inf")
            next_column = 0
            for candidate in range(1, columns + 1):
                if visited[candidate]:
                    continue
                reduced_cost = (
                    values[current_row - 1][candidate - 1]
                    - row_potential[current_row]
                    - column_potential[candidate]
                )
                if reduced_cost < best[candidate]:
                    best[candidate] = reduced_cost
                    previous_column[candidate] = column
                if best[candidate] < improvement:
                    improvement = best[candidate]
                    next_column = candidate
            for candidate in range(columns + 1):
                if visited[candidate]:
                    row_potential[assigned_row[candidate]] += improvement
                    column_potential[candidate] -= improvement
                else:
                    best[candidate] -= improvement
            column = next_column
            if assigned_row[column] == 0:
                break
        while True:
            prior = previous_column[column]
            assigned_row[column] = assigned_row[prior]
            column = prior
            if column == 0:
                break

    pairs = [
        (assigned_row[column] - 1, column - 1)
        for column in range(1, columns + 1)
        if assigned_row[column]
    ]
    if transposed:
        pairs = [(column, row) for row, column in pairs]
    pairs.sort()
    return (
        np.asarray([row for row, _ in pairs], dtype=np.int64),
        np.asarray([column for _, column in pairs], dtype=np.int64),
    )
