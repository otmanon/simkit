"""Tests for ``simkit.vertex_to_simplex_adjacency``."""

from __future__ import annotations

import numpy as np

from simkit.vertex_to_simplex_adjacency import vertex_to_simplex_adjacency


def test_vertex_to_simplex_adjacency_row_sums() -> None:
    T = np.array(
        [
            [0, 1, 2, 3],
            [0, 1, 2, 4],
            [2, 3, 4, 5],
        ]
    )
    nv = 6
    A = vertex_to_simplex_adjacency(T, nv)

    assert A.shape == (nv, T.shape[0])

    expected_counts = np.zeros(nv, dtype=int)
    for simplex in T:
        for vertex in simplex:
            expected_counts[vertex] += 1

    row_sums = np.asarray(A.sum(axis=1)).ravel()
    assert np.array_equal(row_sums, expected_counts)
