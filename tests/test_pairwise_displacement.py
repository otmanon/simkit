"""Tests for ``simkit.pairwise_displacement``."""

from __future__ import annotations

import numpy as np

from simkit.pairwise_displacement import pairwise_displacement


def test_pairwise_displacement_is_difference_between_rows() -> None:
    X = np.array([[1.0, 0.0], [0.0, 2.0]])
    Y = np.array([[0.0, 1.0], [2.0, 0.0], [1.0, 1.0]])
    D = pairwise_displacement(X, Y)
    assert D.shape == (X.shape[0], Y.shape[0], X.shape[1])
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            assert np.allclose(D[i, j], X[i] - Y[j])
