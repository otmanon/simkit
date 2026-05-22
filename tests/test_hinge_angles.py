"""Tests for ``simkit.hinge_angles``."""

from __future__ import annotations

import numpy as np
import pytest

from simkit.hinge_angles import hinge_angles, hinge_angle_velocities

ANGLE_TOL = 1e-10


def _right_angle_hinge() -> tuple[np.ndarray, np.ndarray]:
    """Single hinge with a 90-degree angle at the center vertex."""
    # A=(0,1), B=(0,0), C=(1,0) -> edges B-A and C-B are perpendicular.
    X = np.array([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]])
    H = np.array([[0, 1, 2]], dtype=int)
    return X, H


def test_hinge_angles_right_angle() -> None:
    X, H = _right_angle_hinge()
    theta = hinge_angles(X, H)
    assert theta.shape == (1, 1)
    assert theta[0, 0] == pytest.approx(0.5 * np.pi, abs=ANGLE_TOL)


def test_hinge_angle_velocities_shape() -> None:
    X, H = _right_angle_hinge()
    V = np.zeros_like(X)
    theta_dot = hinge_angle_velocities(X, V, H)
    assert theta_dot.shape == (H.shape[0], 1)
