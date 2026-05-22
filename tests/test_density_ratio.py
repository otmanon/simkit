"""Tests for ``simkit.density_ratio``."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps

from simkit.density_ratio import density_ratio


def test_density_ratio_of_sparse_identity() -> None:
    A = sps.identity(3, format="csc")
    assert density_ratio(A) == 1.0 / 3.0


def test_density_ratio_of_empty_matrix() -> None:
    A = sps.csc_matrix((4, 5))
    assert density_ratio(A) == 0.0


def test_density_ratio_of_dense_matrix() -> None:
    A = sps.csc_matrix(np.ones((2, 2)))
    assert density_ratio(A) == 1.0
