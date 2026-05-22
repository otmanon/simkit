"""Tests for ``simkit.edge_face_adjacency``."""

from __future__ import annotations

import numpy as np

from simkit.edge_face_adjacency import edge_face_adjacency
from simkit.edges import edges


def test_interior_edge_adjacent_to_two_faces() -> None:
    F = np.array([[0, 1, 2], [0, 1, 3]])
    E = edges(F)
    A = edge_face_adjacency(F, E)
    assert A.shape == (2, E.shape[0])

    interior = np.array([0, 1])
    e_idx = np.where(np.all(E == interior, axis=1))[0][0]
    assert A[:, e_idx].toarray().flatten().sum() == 2
