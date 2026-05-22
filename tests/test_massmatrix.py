"""Tests for ``simkit.massmatrix``."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps

from simkit.massmatrix import massmatrix


def _unit_tet() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    T = np.array([[0, 1, 2, 3]])
    return X, T


def test_massmatrix_is_diagonal_with_positive_entries_on_unit_tet() -> None:
    X, T = _unit_tet()
    M = massmatrix(X, T)
    assert isinstance(M, sps.dia_matrix)
    assert M.shape == (X.shape[0], X.shape[0])
    diag = M.diagonal()
    assert np.all(diag > 0)
    assert np.allclose(M.toarray(), np.diag(diag), rtol=0, atol=0)
