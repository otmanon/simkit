"""Tests for ``simkit.pairwise_distance``."""

from __future__ import annotations

import numpy as np

from simkit.pairwise_distance import pairwise_distance


def test_pairwise_distance_to_self_is_zero() -> None:
    X = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
    R = pairwise_distance(X, X)
    assert np.allclose(np.diag(R), 0.0, atol=1e-12)
