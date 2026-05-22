"""Tests for ``simkit.deformation_jacobian``."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

from simkit.deformation_gradient import deformation_gradient
from simkit.deformation_jacobian import deformation_jacobian, membrane_deformation_jacobian


def _unit_simplex(dim: int):
    if dim == 2:
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        T = np.array([[0, 1, 2]])
    elif dim == 3:
        X = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        T = np.array([[0, 1, 2, 3]])
    else:
        raise ValueError(dim)
    return X, T


@pytest.mark.parametrize("dim", [2, 3])
def test_deformation_jacobian_agrees_with_deformation_gradient(dim: int) -> None:
    rng = np.random.default_rng(1)
    X, T = _unit_simplex(dim)
    J = deformation_jacobian(X, T)
    assert isinstance(J, sps.csc_matrix)
    assert J.shape == (T.shape[0] * dim * dim, X.shape[0] * dim)

    U = X + 0.1 * rng.standard_normal(X.shape)
    F_jacobian = (J @ U.reshape(-1, 1)).reshape(-1, dim, dim)
    F_direct = deformation_gradient(X, T, U)
    assert np.allclose(F_jacobian, F_direct, atol=1e-10)


@pytest.mark.parametrize("dim", [2, 3])
def test_deformation_jacobian_recovers_random_linear_maps(dim: int) -> None:
    rng = np.random.default_rng(2)
    X, T = _unit_simplex(dim)
    J = deformation_jacobian(X, T)
    for _ in range(5):
        A = rng.standard_normal((dim, dim))
        U = (A @ X.T).T
        F = (J @ U.reshape(-1, 1)).reshape(-1, dim, dim)
        assert np.allclose(F[0], A, atol=1e-10)


def test_membrane_deformation_jacobian_matches_volume_jacobian_in_2d() -> None:
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    T = np.array([[0, 1, 2]])
    J = deformation_jacobian(X, T)
    Jm = membrane_deformation_jacobian(X, T)
    x = X.reshape(-1, 1)

    def apply(j):
        out = j @ x
        return out.toarray() if hasattr(out, "toarray") else out

    assert np.allclose(apply(J), apply(Jm), atol=1e-12)
