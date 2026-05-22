"""Tests for ``simkit.linear_modal_analysis``."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cvxopt")
pytestmark = pytest.mark.solvers

from simkit.linear_modal_analysis import linear_modal_analysis


def _unit_triangle_mesh() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    T = np.array([[0, 1, 2]], dtype=int)
    return X, T


def test_linear_modal_analysis_returns_eigenvalues_and_shapes() -> None:
    X, T = _unit_triangle_mesh()
    k = 2
    # Pin one vertex to remove rigid modes; free meshes yield a singular Hessian.
    bI = np.array([0], dtype=int)
    E, B = linear_modal_analysis(X, T, k=k, bI=bI)

    n, dim = X.shape
    assert E.shape == (k,)
    assert B.shape == (n * dim, k)
    assert np.all(np.isfinite(E))
    assert np.all(np.isfinite(B))
    assert np.all(E >= 0.0)


def test_linear_modal_analysis_with_boundary_indices() -> None:
    X, T = _unit_triangle_mesh()
    k = 2
    bI = np.array([0], dtype=int)
    E, B = linear_modal_analysis(X, T, k=k, bI=bI)

    n, dim = X.shape
    assert E.shape == (k,)
    assert B.shape == (n * dim, k)
    # Pinned vertex DOFs should remain zero in the returned basis.
    assert np.allclose(B[0:dim, :], 0.0, atol=1e-12)
