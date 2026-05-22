"""Tests for ``simkit.simplex_vertex_map``."""

from __future__ import annotations

import numpy as np

from simkit.simplex_vertex_map import simplex_vertex_map


def test_simplex_vertex_map_gathers_simplex_vertices() -> None:
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    T = np.array([[0, 1, 2]], dtype=int)
    S = simplex_vertex_map(T)

    gathered = S @ X
    expected = X[T].reshape(-1, X.shape[1])
    assert np.allclose(gathered, expected)
