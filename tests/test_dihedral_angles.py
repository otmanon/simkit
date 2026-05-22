"""Tests for ``simkit.dihedral_angles``."""

from __future__ import annotations

import numpy as np
import pytest

from simkit.dihedral_angles import (
    dihedral_angles,
    dihedral_angles_element,
    dihedral_angles_gradient,
    dihedral_angles_gradient_element,
    dihedral_angles_hessian,
    dihedral_angles_hessian_element,
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


def test_dihedral_angles_element_right_angle() -> None:
    x0 = np.array([[0.0, 1.0, 0.0]])
    x1 = np.array([[0.0, 0.0, 0.0]])
    x2 = np.array([[1.0, 0.0, 0.0]])
    x3 = np.array([[0.0, 0.0, 1.0]])
    theta = dihedral_angles_element(x0, x1, x2, x3)
    assert theta.shape == (1, 1)
    assert abs(theta[0, 0]) == pytest.approx(0.5 * np.pi, abs=ANGLE_TOL)


def test_dihedral_angles_matches_element_tier() -> None:
    X, D = _right_angle_hinge()
    theta = dihedral_angles(X, D)
    x0 = X[D[:, 0]]
    x1 = X[D[:, 1]]
    x2 = X[D[:, 2]]
    x3 = X[D[:, 3]]
    theta_element = dihedral_angles_element(x0, x1, x2, x3)
    assert np.allclose(theta, theta_element, atol=ANGLE_TOL)


def test_dihedral_angles_gradient_element_matches_fd() -> None:
    x0 = np.array([[0.0, 1.0, 0.0]])
    x1 = np.array([[0.0, 0.0, 0.0]])
    x2 = np.array([[1.0, 0.0, 0.0]])
    x3 = np.array([[0.0, 0.0, 1.0]])
    y = np.concatenate([x0, x1, x2, x3], axis=1).flatten()

    def angle_flat(coords: np.ndarray) -> np.ndarray:
        c = coords.reshape(1, 12)
        return dihedral_angles_element(c[:, :3], c[:, 3:6], c[:, 6:9], c[:, 9:]).flatten()

    g_fd = gradient_cfd(angle_flat, y, FD_STEP).reshape(12)
    g = dihedral_angles_gradient_element(x0, x1, x2, x3).reshape(12)
    assert np.allclose(g, g_fd, atol=GRAD_TOL)


def test_dihedral_angles_global_gradient_shape() -> None:
    X, D = _right_angle_hinge()
    g = dihedral_angles_gradient(X, D)
    assert g.shape == (X.shape[0] * 3, 1)


def test_dihedral_angles_hessian_is_symmetric() -> None:
    X, D = _right_angle_hinge()
    H = dihedral_angles_hessian(X, D).toarray()
    assert np.allclose(H, H.T, atol=1e-10)


def test_dihedral_angles_hessian_element_blocks_are_symmetric() -> None:
    x0 = np.array([[0.0, 1.0, 0.0]])
    x1 = np.array([[0.0, 0.0, 0.0]])
    x2 = np.array([[1.0, 0.0, 0.0]])
    x3 = np.array([[0.0, 0.0, 1.0]])
    H = dihedral_angles_hessian_element(x0, x1, x2, x3)[0]
    assert np.allclose(H, H.T, atol=1e-10)
