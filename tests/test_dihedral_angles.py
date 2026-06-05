"""Tests for ``simkit.dihedral_angles`` (the 3D member + the dim dispatcher)."""

from __future__ import annotations

import numpy as np
import pytest

from simkit.dihedral_angles import (
    dihedral_angles,
    dihedral_angles_gradient_element,
    dihedral_angles_hessian_element,
)
from simkit.dihedral_angles_2d import dihedral_angles_2d
from simkit.dihedral_angles_3d import (
    dihedral_angles_3d,
    dihedral_angles_3d_gradient_element,
    dihedral_angles_3d_hessian_element,
)
from simkit.dihedral_wedges import dihedral_wedges
from simkit.gradient_cfd import gradient_cfd


FD_STEP = 1e-6
GRAD_TOL = 1e-4
ANGLE_TOL = 1e-5


def _right_angle_hinge():
    """Two triangles meeting at 90 degrees along the x-axis."""
    X = np.array(
        [
            [0.0, 1.0, 0.0],  # 0 apex tri1
            [0.0, 0.0, 0.0],  # 1 shared
            [1.0, 0.0, 0.0],  # 2 shared
            [0.0, 0.0, 1.0],  # 3 apex tri2
        ]
    )
    F = np.array([[0, 1, 2], [1, 2, 3]])
    D = dihedral_wedges(F)
    return X, D


def test_dihedral_angles_3d_right_angle() -> None:
    X, D = _right_angle_hinge()
    theta = dihedral_angles_3d(X, D)
    assert theta.shape == (1, 1)
    assert abs(theta[0, 0]) == pytest.approx(0.5 * np.pi, abs=ANGLE_TOL)


def test_dispatcher_matches_per_dimension() -> None:
    # 3D
    X3, D = _right_angle_hinge()
    assert np.allclose(dihedral_angles(X3, D), dihedral_angles_3d(X3, D))
    # 2D (triple connectivity, (n,2) positions)
    X2 = np.array([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]])
    C = np.array([[0, 1, 2]], dtype=int)
    assert np.allclose(dihedral_angles(X2, C), dihedral_angles_2d(X2, C))


def test_dihedral_angles_3d_gradient_element_matches_fd() -> None:
    X, D = _right_angle_hinge()
    g = dihedral_angles_3d_gradient_element(X, D).reshape(12)

    tri = D[0]

    def angle_flat(coords):
        Y = X.copy()
        Y[tri] = coords.reshape(4, 3)
        return dihedral_angles_3d(Y, D[:1]).flatten()

    g_fd = gradient_cfd(angle_flat, X[tri].flatten(), FD_STEP).reshape(12)
    assert np.allclose(g, g_fd, atol=GRAD_TOL)
    # dispatcher returns the same compact gradient
    assert np.allclose(dihedral_angles_gradient_element(X, D), dihedral_angles_3d_gradient_element(X, D))


def test_dihedral_angles_3d_hessian_element_symmetric() -> None:
    X, D = _right_angle_hinge()
    H = dihedral_angles_3d_hessian_element(X, D)
    assert H.shape == (D.shape[0], 12, 12)
    assert np.allclose(H[0], H[0].T, atol=1e-10)
    assert np.allclose(dihedral_angles_hessian_element(X, D), H)
