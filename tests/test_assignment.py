"""Check the bundled assignment solver against exhaustive small problems."""

from itertools import permutations

import numpy as np
import pytest

from src.core.assignment import linear_sum_assignment


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (3, 2), (4, 4), (5, 3)])
def test_assignment_has_minimum_cost(shape: tuple[int, int]) -> None:
    random = np.random.default_rng(12345)
    rows, columns = shape
    for _ in range(30):
        costs = random.integers(-5, 10, size=shape).astype(np.float64)
        assigned_rows, assigned_columns = linear_sum_assignment(costs)
        assert len(assigned_rows) == min(shape)
        assert len(set(assigned_rows)) == len(assigned_rows)
        assert len(set(assigned_columns)) == len(assigned_columns)
        actual = costs[assigned_rows, assigned_columns].sum()
        if rows <= columns:
            optimum = min(
                sum(costs[row, column] for row, column in enumerate(chosen))
                for chosen in permutations(range(columns), rows)
            )
        else:
            optimum = min(
                sum(costs[row, column] for column, row in enumerate(chosen))
                for chosen in permutations(range(rows), columns)
            )
        assert actual == optimum


def test_assignment_handles_empty_matrix() -> None:
    rows, columns = linear_sum_assignment(np.empty((0, 3)))
    assert rows.size == columns.size == 0


def test_assignment_rejects_non_finite_costs() -> None:
    with pytest.raises(ValueError, match="finite"):
        linear_sum_assignment(np.array([[float("inf")]]))
