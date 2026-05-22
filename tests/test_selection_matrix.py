"""Tests for ``simkit.selection_matrix``."""

from __future__ import annotations

import numpy as np

from simkit.selection_matrix import selection_matrix


def test_selection_matrix_selects_indices() -> None:
    n = 6
    cI = np.array([1, 4, 0])
    S = selection_matrix(cI, n)
    x = np.arange(n, dtype=float)

    selected = (S @ x.reshape(-1, 1)).flatten()
    assert np.allclose(selected, x[cI])
