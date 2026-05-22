"""Tests for ``simkit.harmonic_coordinates``."""

from __future__ import annotations

import numpy as np
import pytest

from simkit.harmonic_coordinates import harmonic_coordinates


def _unit_tet() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    T = np.array([[0, 1, 2, 3]])
    return X, T


def test_harmonic_coordinates_partition_of_unity_on_unit_tet() -> None:
    X, T = _unit_tet()
    bI = np.array([0, 2, 3])

    W = harmonic_coordinates(X, T, bI)

    assert W.shape == (X.shape[0], bI.shape[0])
    assert np.allclose(W[bI], np.eye(bI.shape[0]), atol=1e-10)
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-10)
