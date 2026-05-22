"""Tests for ``simkit.dihedral_wedges``."""

from __future__ import annotations

import numpy as np

from simkit.dihedral_wedges import dihedral_wedges


def test_dihedral_wedges_two_triangle_mesh() -> None:
    F = np.array([[0, 1, 2], [1, 2, 3]])
    D = dihedral_wedges(F)
    assert D.shape == (1, 4)
    shared = {1, 2}
    assert shared.issubset(set(D[0, 1:3]))
    assert D[0, 0] in {0, 3}
    assert D[0, 3] in {0, 3}
    assert D[0, 0] != D[0, 3]


def test_dihedral_wedges_drops_boundary_edges() -> None:
    F = np.array([[0, 1, 2]])
    D = dihedral_wedges(F)
    assert D.shape == (0, 4)


def test_dihedral_wedges_square_two_triangles() -> None:
    F = np.array([[0, 1, 2], [0, 2, 3]])
    D = dihedral_wedges(F)
    assert D.shape[0] == 1
    assert len(np.unique(D)) == 4
