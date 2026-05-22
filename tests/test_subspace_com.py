"""Tests for ``simkit.subspace_com``."""

from __future__ import annotations

import numpy as np
import scipy as sp

from simkit.massmatrix import massmatrix
from simkit.subspace_com import subspace_com


def _unit_tet() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    T = np.array([[0, 1, 2, 3]])
    return X, T


def test_subspace_com_matches_mass_weighted_centroid() -> None:
    X, T = _unit_tet()
    dim = X.shape[1]
    n = X.shape[0]

    B = sp.sparse.identity(n * dim)
    z = X.reshape(-1, 1)

    com = subspace_com(z, B, X, T)

    M = massmatrix(X, T)
    m = M.diagonal()
    com_expected = np.average(X, axis=0, weights=m)

    assert com.shape == (1, dim)
    assert np.allclose(com, com_expected[None, :], atol=1e-10)
