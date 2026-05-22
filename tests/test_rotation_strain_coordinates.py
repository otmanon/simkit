"""Tests for ``simkit.rotation_strain_coordinates``."""

from __future__ import annotations

import numpy as np

from simkit.rotation_strain_coordinates import RSPrecompute, rotation_strain_coordinates


def _unit_triangle_mesh() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    T = np.array([[0, 1, 2]], dtype=int)
    return X, T


def test_rs_precompute_attributes_have_expected_shapes() -> None:
    X, T = _unit_triangle_mesh()
    n, dim = X.shape
    nt = T.shape[0]

    pre = RSPrecompute(X, T, pinned=np.array([0], dtype=int))

    assert pre.X.shape == (n, dim)
    assert pre.T.shape == (nt, 3)
    assert pre.J.shape[0] == nt * dim * dim
    assert pre.K.shape[0] == n * dim


def test_rotation_strain_coordinates_returns_displacement_shape() -> None:
    X, T = _unit_triangle_mesh()
    n, dim = X.shape
    u = np.zeros((n, dim))

    u_rs, pre = rotation_strain_coordinates(X, T, u, pinned=np.array([0], dtype=int))

    assert u_rs.shape == (n, dim)
    assert isinstance(pre, RSPrecompute)
