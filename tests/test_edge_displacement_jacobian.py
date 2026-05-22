"""Tests for ``simkit.edge_displacement_jacobian``."""

from __future__ import annotations

import numpy as np

from simkit.edge_displacement_jacobian import edge_displacement_jacobian


def test_jacobian_shape_and_matches_endpoint_differences() -> None:
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    E = np.array([[0, 1], [1, 2], [2, 0]])
    J = edge_displacement_jacobian(X, E)
    assert J.shape == (E.shape[0], X.shape[0])

    disp = (J @ X).toarray() if hasattr(J @ X, "toarray") else J @ X
    expected = X[E[:, 0], :] - X[E[:, 1], :]
    assert np.allclose(disp, expected, atol=1e-12)
