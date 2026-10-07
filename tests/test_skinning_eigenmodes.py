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


def test_per_element_stiffness_makes_modes_material_aware() -> None:
    # A strip whose left half is 1000x stiffer than its right half. With a
    # per-element mu the lowest non-rigid modes vary across the soft half and
    # stay nearly constant on the stiff half; with a uniform mu they do not.
    nx, ny = 21, 4
    xs, ys = np.meshgrid(np.linspace(0, 2, nx), np.linspace(0, 0.3, ny), indexing="ij")
    X = np.c_[xs.ravel(), ys.ravel()]
    idx = np.arange(nx * ny).reshape(nx, ny)
    T = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
            T += [[a, b, c], [a, c, d]]
    T = np.array(T)
    mu = np.where(X[T].mean(axis=1)[:, 0] < 1.0, 1e3, 1.0)

    W_mat, E_mat, B_mat = skinning_eigenmodes(X, T, 3, mu=mu)
    W_uni, E_uni, B_uni = skinning_eigenmodes(X, T, 3)
    assert W_mat.shape == (X.shape[0], 3) and B_mat.shape == (X.size, 18)

    stiff, soft = X[:, 0] < 1.0, X[:, 0] > 1.0
    for j in (1, 2):  # mode 0 is the constant
        span_stiff = np.ptp(W_mat[stiff, j])
        span_soft = np.ptp(W_mat[soft, j])
        assert span_stiff < 0.05 * span_soft  # flat on the stiff half
        assert np.ptp(W_uni[stiff, j]) > 0.3 * np.ptp(W_uni[soft, j])  # uniform: not flat
