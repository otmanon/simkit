"""Tests for ``simkit.edge_length_jacobian``."""

from __future__ import annotations

import numpy as np

from simkit.edge_length_jacobian import edge_length_jacobian
from simkit.edge_lengths import edge_lengths
from simkit.gradient_cfd import gradient_cfd

FD_STEP = 1e-6
TOL = 1e-5


def test_edge_length_jacobian_shape_and_fd() -> None:
    X = np.array([[0.0, 0.0], [1.0, 0.0]])
    E = np.array([[0, 1]])
    J = edge_length_jacobian(X, E)
    assert J.shape == (1, X.size)

    y = X.reshape(-1)

    def length_flat(coords: np.ndarray) -> np.ndarray:
        pts = coords.reshape(2, 2)
        return edge_lengths(pts, E)

    g_fd = gradient_cfd(length_flat, y, FD_STEP).reshape(-1)
    g_j = J.toarray().reshape(-1)
    assert np.allclose(g_j, g_fd, atol=TOL)
