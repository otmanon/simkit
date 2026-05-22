"""Tests for ``simkit.eigs``."""

from __future__ import annotations

import numpy as np
import pytest
import scipy as sp

pytest.importorskip("cvxopt")

from simkit.eigs import eigs


def test_eigs_returns_expected_shapes() -> None:
    n = 6
    diag = np.arange(1, n + 1, dtype=float)
    A = sp.sparse.diags(diag, format="csc")
    M = sp.sparse.identity(n, format="csc")
    k = 3

    D, B = eigs(A, k=k, M=M)

    assert D.shape == (k,)
    assert B.shape == (n, k)
    assert np.all(np.isfinite(D))
    assert np.all(np.isfinite(B))
