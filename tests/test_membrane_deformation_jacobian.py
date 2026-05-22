"""Tests for ``simkit.membrane_deformation_jacobian``."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps

from simkit.membrane_deformation_jacobian import (
    membrane_deformation_gradient,
    membrane_deformation_jacobian,
)


def _unit_triangle_xy() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    T = np.array([[0, 1, 2]])
    return X, T


def test_membrane_deformation_jacobian_shape_on_2d_triangle_mesh() -> None:
    X, T = _unit_triangle_xy()
    n, t = X.shape[0], T.shape[0]
    J = membrane_deformation_jacobian(X, T)
    assert isinstance(J, sps.spmatrix)
    assert J.shape == (t * 3 * 2, n * 3)

    Y = X + 0.05 * np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [-0.05, -0.05, 0.0]])
    F_jacobian = (J @ Y.reshape(-1, 1)).reshape(t, 3, 2)
    F_direct = membrane_deformation_gradient(Y, X, T)
    assert np.allclose(F_jacobian, F_direct, atol=1e-10)
