"""Tests for ``simkit.deformation_gradient``."""

from __future__ import annotations

import numpy as np
import pytest

from simkit.deformation_gradient import deformation_gradient


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
def test_identity_deformation_gives_identity_gradient(dim: int) -> None:
    X, T = _unit_simplex(dim)
    F = deformation_gradient(X, T, X)
    assert F.shape == (1, dim, dim)
    assert np.allclose(F[0], np.eye(dim), atol=1e-12)


@pytest.mark.parametrize("dim", [2, 3])
def test_uniform_linear_map_matches_jacobian(dim: int) -> None:
    from simkit.deformation_jacobian import deformation_jacobian

    rng = np.random.default_rng(0)
    X, T = _unit_simplex(dim)
    A = rng.standard_normal((dim, dim))
    U = (A @ X.T).T
    F = deformation_gradient(X, T, U)
    J = deformation_jacobian(X, T)
    F_jacobian = (J @ U.reshape(-1, 1)).reshape(-1, dim, dim)
    assert np.allclose(F, F_jacobian, atol=1e-10)
