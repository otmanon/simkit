"""Tests for ``simkit.subspace_corrolation``."""

from __future__ import annotations

import numpy as np
import scipy as sp

from simkit.subspace_corrolation import subspace_corrolation


def test_subspace_corrolation_matrix_shape() -> None:
    n = 10
    rng = np.random.default_rng(2)
    A = rng.standard_normal((n, n))
    C = A.T @ A + 0.1 * np.eye(n)

    rho = subspace_corrolation(C)

    assert rho.shape == (n, n)
    assert isinstance(rho, sp.sparse.csc_matrix)
    assert np.allclose(rho.diagonal(), 1.0, atol=1e-10)
