"""Tests for ``simkit.random_impulse_vibes``."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cvxopt")
pytestmark = [pytest.mark.solvers, pytest.mark.learn]

from simkit.random_impulse_vibes import random_impulse_vibes


def _unit_tet() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    T = np.array([[0, 1, 2, 3]], dtype=int)
    return X, T


def test_random_impulse_vibes_shapes_and_partition_of_unity() -> None:
    X, T = _unit_tet()
    m = 2
    G, cI, G_full = random_impulse_vibes(X, T, m, h=0.1, ord=1)

    n = X.shape[0]
    assert G.shape == (n, m)
    assert G_full.shape == (n, m)
    assert cI.shape == (n,)
    assert np.all(cI >= 0) and np.all(cI < m)
    assert np.allclose(G.sum(axis=1), 1.0, atol=1e-10)
    assert np.all(np.isfinite(G))
    assert np.all(np.isfinite(G_full))
