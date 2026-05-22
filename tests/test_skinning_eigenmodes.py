"""Tests for ``simkit.skinning_eigenmodes``."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cvxopt")
pytestmark = pytest.mark.solvers

from simkit.skinning_eigenmodes import skinning_eigenmodes


def _unit_triangle_mesh() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    T = np.array([[0, 1, 2]], dtype=int)
    return X, T


def test_skinning_eigenmodes_returns_modes_and_jacobian() -> None:
    X, T = _unit_triangle_mesh()
    n, dim = X.shape
    k = 2
    bI = np.array([0], dtype=int)

    W, E, B = skinning_eigenmodes(X, T, k, bI=bI)

    assert W.shape == (n, k)
    assert E.shape == (k,)
    assert B.shape[0] == n * dim
    assert np.all(np.isfinite(W))
    assert np.all(np.isfinite(E))
