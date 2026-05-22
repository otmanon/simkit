"""Tests for ``simkit.dihedral_wedge_map``."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps

from simkit.dihedral_wedge_map import dihedral_wedge_map


def test_dihedral_wedge_map_shape_and_entries() -> None:
    D = np.array([[3, 0, 1, 2], [5, 1, 2, 4]])
    nv = 6
    M = dihedral_wedge_map(D, nv)
    assert isinstance(M, sps.csc_matrix)
    assert M.shape == (4 * D.shape[0], nv)
    assert M.nnz == 4 * D.shape[0]
    assert np.allclose(M.toarray().sum(axis=1), 1.0)


def test_dihedral_wedge_map_gathers_hinge_vertices() -> None:
    D = np.array([[2, 0, 1, 3]])
    nv = 4
    M = dihedral_wedge_map(D, nv)
    values = np.arange(nv, dtype=float)
    gathered = M @ values
    expected = np.array([2.0, 0.0, 1.0, 3.0])
    assert np.allclose(gathered, expected)
