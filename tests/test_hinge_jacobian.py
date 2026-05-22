"""Tests for ``simkit.hinge_jacobian``."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps

from simkit.hinge_jacobian import hinge_jacobian, hinge_jacobian_compact


def _right_angle_hinge() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]])
    H = np.array([[0, 1, 2], [1, 2, 0]], dtype=int)
    return X, H


def test_hinge_jacobian_compact_shape() -> None:
    X, H = _right_angle_hinge()
    J = hinge_jacobian_compact(X, H)
    assert J.shape == (H.shape[0], 6)


def test_hinge_jacobian_sparse_shape() -> None:
    X, H = _right_angle_hinge()
    J = hinge_jacobian(X, H)
    assert isinstance(J, sps.csc_matrix)
    assert J.shape == (H.shape[0], 2 * X.shape[0])
