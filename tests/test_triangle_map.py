"""Tests for ``simkit.triangle_map``."""

from __future__ import annotations

import numpy as np

from simkit.triangle_map import triangle_map


def test_triangle_map_sparse_gather_shape() -> None:
    F = np.array([[0, 1, 2], [1, 2, 3]])
    nv = 4
    M = triangle_map(F, nv)

    assert M.shape == (3 * F.shape[0], nv)
    assert M.nnz == F.size

    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    corners = (M @ vertices).reshape(-1, 3)
    expected = vertices[F].reshape(-1, 3)
    assert np.allclose(corners, expected)
