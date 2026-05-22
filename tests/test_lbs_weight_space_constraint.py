"""Tests for ``simkit.lbs_weight_space_constraint``."""

from __future__ import annotations

import numpy as np

from simkit.lbs_weight_space_constraint import lbs_weight_space_constraint


def _unit_triangle_mesh() -> tuple[np.ndarray, np.ndarray]:
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    T = np.array([[0, 1, 2]], dtype=int)
    return V, T


def test_lbs_weight_space_constraint_shape() -> None:
    V, T = _unit_triangle_mesh()
    n, d = V.shape
    # Pin vertex 0: u_0 = 0 in both coordinates.
    C = np.zeros((2, n * d))
    C[0, 0] = 1.0
    C[1, 1] = 1.0

    A = lbs_weight_space_constraint(V, T, C)
    assert A.ndim == 2
    assert A.shape[1] == n
