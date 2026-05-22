"""Tests for ``simkit.random_impulse_vibes``."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cvxopt")

from simkit.random_impulse_vibes import random_impulse_vibes


def _unit_triangle_mesh() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    T = np.array([[0, 1, 2]], dtype=int)
    return X, T


def test_random_impulse_vibes_returns_weight_matrix_shapes() -> None:
    X, T = _unit_triangle_mesh()
    n, m = X.shape[0], 2

    G, cI, G_full = random_impulse_vibes(X, T, m)

    assert G.shape == (n, m)
    assert cI.shape == (n,)
    assert G_full.shape == (n, m)
    assert np.all(np.isfinite(G))
    assert np.allclose(G.sum(axis=1), 1.0, atol=1e-12)
