"""Tests for ``simkit.dirichlet_laplacian``."""

from __future__ import annotations

import numpy as np
import pytest
import scipy as sp

from simkit.dirichlet_laplacian import dirichlet_laplacian


def _unit_triangle() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    T = np.array([[0, 1, 2]])
    return X, T


def _unit_tet() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    T = np.array([[0, 1, 2, 3]])
    return X, T


@pytest.mark.parametrize("mesh", ["triangle", "tet"])
def test_scalar_laplacian_shape_symmetry_psd(mesh: str) -> None:
    X, T = _unit_triangle() if mesh == "triangle" else _unit_tet()
    n = X.shape[0]
    L = dirichlet_laplacian(X, T, mu=1.0, vector=False)
    assert L.shape == (n, n)
    assert sp.sparse.isspmatrix_csc(L)
    D = L.toarray()
    assert np.allclose(D, D.T, atol=1e-12)
    v = np.random.default_rng(0).standard_normal(n)
    assert v @ D @ v >= -1e-10


@pytest.mark.parametrize("mesh", ["triangle", "tet"])
def test_vector_laplacian_shape_symmetry_psd(mesh: str) -> None:
    X, T = _unit_triangle() if mesh == "triangle" else _unit_tet()
    n, dim = X.shape
    L = dirichlet_laplacian(X, T, mu=1.0, vector=True)
    assert L.shape == (n * dim, n * dim)
    D = L.toarray()
    assert np.allclose(D, D.T, atol=1e-12)
    v = np.random.default_rng(1).standard_normal(n * dim)
    assert v @ D @ v >= -1e-10


def test_scalar_laplacian_nullspace_includes_constants() -> None:
    X, T = _unit_triangle()
    L = dirichlet_laplacian(X, T, mu=1.0, vector=False)
    ones = np.ones(X.shape[0])
    assert np.allclose(L @ ones, 0.0, atol=1e-10)
