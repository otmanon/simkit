"""Tests for ``simkit.remove_redundant_columns``."""

from __future__ import annotations

import numpy as np
import scipy as sp

from simkit.remove_redundant_columns import remove_redundant_columns


def test_remove_redundant_columns_drops_duplicates() -> None:
    col = np.array([1.0, 2.0, 3.0, 4.0])
    B = sp.sparse.csc_matrix(np.column_stack([col, col, 2 * col]))
    B_reduced = remove_redundant_columns(B)

    assert B_reduced.shape[0] == B.shape[0]
    assert B_reduced.shape[1] < B.shape[1]
    assert B_reduced.shape[1] >= 1
