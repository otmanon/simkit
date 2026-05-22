"""Tests for ``simkit.random_edges``."""

from __future__ import annotations

import numpy as np

from simkit.random_edges import random_undirected_edges


def _unit_triangle_mesh() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    T = np.array([[0, 1, 2]], dtype=int)
    return X, T


def test_random_undirected_edges_count_on_triangle_mesh() -> None:
    X, _T = _unit_triangle_mesh()
    n = X.shape[0]
    max_edges = n * (n - 1) // 2

    edges = random_undirected_edges(n, num_edges=2)
    assert edges.shape == (2, 2)
    assert np.all(edges[:, 0] < edges[:, 1])

    edges_all = random_undirected_edges(n, num_edges=max_edges + 5)
    assert edges_all.shape == (max_edges, 2)
