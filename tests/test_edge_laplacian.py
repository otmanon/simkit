"""Tests for ``simkit.edge_laplacian``."""

from __future__ import annotations

import numpy as np
import scipy as sp

from simkit.edge_laplacian import edge_laplacian


def test_edge_laplacian_shape_and_annihilates_constants() -> None:
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    E = np.array([[0, 1], [1, 2], [2, 0]])
    L = edge_laplacian(X, E)
    assert L.shape == (X.shape[0], X.shape[0])
    assert sp.sparse.isspmatrix(L)
    ones = np.ones(X.shape[0])
    assert np.allclose(L @ ones, 0.0, atol=1e-12)
