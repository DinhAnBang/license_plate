"""Unit tests for SORT IoU matrices and Hungarian assignment."""

from __future__ import annotations

import numpy as np

from core.assignment import gated_assignment, iou_matrix, linear_assignment


def _cost(pairs: np.ndarray, matrix: np.ndarray) -> float:
    return sum(float(matrix[row, column]) for row, column in pairs)


def test_linear_assignment() -> None:
    assert linear_assignment([[3.0]]).tolist() == [[0, 0]]

    obvious = np.asarray([[1.0, 9.0], [8.0, 2.0]])
    assert linear_assignment(obvious).tolist() == [[0, 0], [1, 1]]

    # Greedily taking cost 1 first forces cost 100. The global optimum is 4.
    greedy_trap = np.asarray([[1.0, 2.0], [2.0, 100.0]])
    optimum = linear_assignment(greedy_trap)
    assert optimum.tolist() == [[0, 1], [1, 0]]
    assert _cost(optimum, greedy_trap) == 4.0

    for matrix in (
        np.asarray([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0]]),
        np.asarray([[4.0, 1.0], [2.0, 0.0], [3.0, 2.0]]),
    ):
        first = linear_assignment(matrix)
        second = linear_assignment(matrix)
        assert np.array_equal(first, second)
        assert len(first) == min(matrix.shape)
        assert len(set(first[:, 0])) == len(first)
        assert len(set(first[:, 1])) == len(first)

    assert linear_assignment(np.empty((0, 3))).shape == (0, 2)
    assert linear_assignment(np.empty((3, 0))).shape == (0, 2)

    overlaps = np.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=np.float64)
    assert gated_assignment(overlaps, 0.25) == [(0, 0, 0.9), (1, 1, 0.8)]


def test_iou_matrix() -> None:
    overlaps = iou_matrix(
        [[0, 0, 10, 10], [20, 20, 20, 30], [float("nan"), 0, 1, 1]],
        [[0, 0, 10, 10], [5, 0, 15, 10]],
    )
    assert overlaps.shape == (3, 2)
    assert overlaps[0, 0] == 1.0
    assert 0.0 < overlaps[0, 1] < 1.0
    assert np.all(overlaps[1:] == 0.0)
    assert np.all((0.0 <= overlaps) & (overlaps <= 1.0))
    assert iou_matrix([], [[0, 0, 1, 1]]).shape == (0, 1)
    assert iou_matrix([[0, 0, 1, 1]], []).shape == (1, 0)


def main() -> int:
    test_linear_assignment()
    test_iou_matrix()
    print("Hungarian assignment and IoU matrix tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
