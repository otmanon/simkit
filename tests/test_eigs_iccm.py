"""Tests for ``simkit.eigs_iccm``."""

from __future__ import annotations

import numpy as np
import pytest
import scipy as sp

pytest.importorskip("cvxopt")
pytestmark = pytest.mark.solvers

from simkit.eigs_iccm import eigs_iccm, sp2cvxopt


def test_sp2cvxopt_round_trip_dense() -> None:
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    A_cvx = sp2cvxopt(A)
    assert np.allclose(np.array(A_cvx), A)


def test_eigs_iccm_returns_orthonormal_modes() -> None:
    H = sp.sparse.diags([2.0, 3.0, 4.0], format="csc")
    M = sp.sparse.identity(3, format="csc")
    k = 2
    l = 0.01

    U = eigs_iccm(H, l, k, M=M, max_iters=20, tolerance=1e-4)

    assert U.shape == (3, k)
    for i in range(k):
        norm = float(U[:, i].T @ M @ U[:, i])
        assert np.isclose(norm, 1.0, rtol=1e-3)
    if k > 1:
        orth = float(U[:, 0].T @ M @ U[:, 1])
        assert np.isclose(orth, 0.0, atol=1e-3)
