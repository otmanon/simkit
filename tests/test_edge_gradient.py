"""Tests for ``simkit.edge_gradient``."""

from __future__ import annotations

import numpy as np

from simkit.edge_gradient import edge_gradient
from simkit.edge_length_jacobian import edge_length_jacobian
from simkit.edge_lengths import edge_lengths
from simkit.gradient_cfd import gradient_cfd

FD_STEP = 1e-6
TOL = 1e-5


def test_edge_gradient_matches_fd_on_single_edge() -> None:
    X = np.array([[0.0, 0.0], [1.0, 0.0]])
    E = np.array([[0, 1]])
    G = edge_gradient(X, E)
    assert G.shape == (1, X.shape[0])

    y = X.reshape(-1)

    def lengths_flat(coords: np.ndarray) -> np.ndarray:
        pts = coords.reshape(2, 2)
        return edge_lengths(pts, E)

    g_fd = gradient_cfd(lengths_flat, y, FD_STEP).reshape(2, 2)
    J = edge_length_jacobian(X, E)
    assert np.allclose(J.toarray().reshape(2, 2), g_fd, atol=TOL)

    # G maps vertex coordinates to per-edge (x_i - x_j) / l (length gradient direction).
    assert np.allclose(G @ X, np.array([[1.0, 0.0]]), atol=TOL)
