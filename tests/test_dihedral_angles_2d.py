"""Tests for ``simkit.dihedral_angles_2d`` (the 2D hinge member)."""

from __future__ import annotations

import numpy as np
import pytest

from simkit.dihedral_angles_2d import (
    dihedral_angles_2d,
    dihedral_angles_2d_velocities,
    dihedral_angles_2d_gradient_element,
    dihedral_angles_2d_hessian_element,
)
from simkit.gradient_cfd import gradient_cfd

ANGLE_TOL = 1e-10
FD_STEP = 1e-6
GRAD_TOL = 1e-5


def _right_angle_hinge():
    """Single hinge with a 90-degree turn at the center vertex."""
    X = np.array([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]])
    C = np.array([[0, 1, 2]], dtype=int)
    return X, C


def test_dihedral_angles_2d_right_angle() -> None:
    X, C = _right_angle_hinge()
    theta = dihedral_angles_2d(X, C)
    assert theta.shape == (1, 1)
    assert theta[0, 0] == pytest.approx(0.5 * np.pi, abs=ANGLE_TOL)


def test_dihedral_angles_2d_velocities_shape() -> None:
    X, C = _right_angle_hinge()
    V = np.zeros_like(X)
    theta_dot = dihedral_angles_2d_velocities(X, V, C)
    assert theta_dot.shape == (C.shape[0], 1)


def test_dihedral_angles_2d_gradient_element_shape_and_fd() -> None:
    # A short beam, perturbed off straight so the gradient is nonzero.
    X = np.zeros((4, 2)); X[:, 0] = np.arange(4)
    C = np.array([[i, i + 1, i + 2] for i in range(2)], dtype=int)
    rng = np.random.default_rng(0)
    Xd = X + 0.05 * rng.standard_normal(X.shape)

    J = dihedral_angles_2d_gradient_element(Xd, C)
    assert J.shape == (C.shape[0], 6)

    # Compare the first hinge's compact gradient to a finite difference.
    tri = C[0]

    def angle0(coords):
        Y = Xd.copy()
        Y[tri] = coords.reshape(3, 2)
        return dihedral_angles_2d(Y, C[:1]).flatten()

    g_fd = gradient_cfd(angle0, Xd[tri].flatten(), FD_STEP).reshape(6)
    assert np.allclose(J[0], g_fd, atol=GRAD_TOL)


def test_dihedral_angles_2d_hessian_element_symmetric() -> None:
    X, C = _right_angle_hinge()
    blocks = dihedral_angles_2d_hessian_element(X, C)
    assert blocks.shape == (C.shape[0], 6, 6)
    for i in range(C.shape[0]):
        assert np.allclose(blocks[i], blocks[i].T, atol=1e-10)
