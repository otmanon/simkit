"""Tests for ``simkit.biharmonic_coordinates``."""

from __future__ import annotations

import numpy as np

from simkit.biharmonic_coordinates import biharmonic_coordinates


def _unit_tet():
    X = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    T = np.array([[0, 1, 2, 3]])
    return X, T


def test_biharmonic_coordinates_interpolate_handles() -> None:
    X, T = _unit_tet()
    bI = np.array([0, 1])
    W = biharmonic_coordinates(X, T, bI)
    assert W.shape == (X.shape[0], 2)
    assert np.allclose(W[bI], np.eye(2), atol=1e-10)


def test_biharmonic_coordinates_partition_of_unity() -> None:
    X, T = _unit_tet()
    bI = np.array([0, 2, 3])
    W = biharmonic_coordinates(X, T, bI)
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-10)


def test_biharmonic_coordinates_ignore_duplicate_handles() -> None:
    X, T = _unit_tet()
    bI = np.array([1, 1, 3])
    W = biharmonic_coordinates(X, T, bI)
    assert W.shape[1] == 2
    assert np.allclose(W[1], np.array([1.0, 0.0]), atol=1e-10)
