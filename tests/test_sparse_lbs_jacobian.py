"""Tests for ``simkit.sparse_lbs_jacobian``."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

from simkit.lbs_jacobian import lbs_jacobian
from simkit.sparse_lbs_jacobian import sparse_lbs_jacobian


def _random_compact_weights(rng, n, k):
    W = rng.random((n, k))
    W[rng.random((n, k)) < 0.4] = 0.0  # compact support
    W[W.sum(axis=1) == 0, 0] = 1.0
    return W / W.sum(axis=1, keepdims=True)


@pytest.mark.parametrize("d", [2, 3])
def test_affine_frames_match_dense_lbs_jacobian(d) -> None:
    rng = np.random.default_rng(0)
    n, k = 9, 3
    X = rng.standard_normal((n, d))
    W = _random_compact_weights(rng, n, k)
    B = sparse_lbs_jacobian(X, sps.csr_matrix(W), order=1)
    assert sps.issparse(B)
    assert B.shape == (n * d, k * d * (d + 1))
    np.testing.assert_allclose(B.toarray(), lbs_jacobian(X, W), atol=1e-14)


def test_accepts_dense_weights() -> None:
    rng = np.random.default_rng(2)
    X = rng.standard_normal((6, 2))
    W = _random_compact_weights(rng, 6, 2)
    a = sparse_lbs_jacobian(X, W)
    b = sparse_lbs_jacobian(X, sps.csr_matrix(W))
    assert (a != b).nnz == 0


def test_sparsity_follows_weights() -> None:
    rng = np.random.default_rng(3)
    n, d, k = 20, 2, 5
    X = rng.standard_normal((n, d))
    W = _random_compact_weights(rng, n, k)
    B = sparse_lbs_jacobian(X, W, order=1)
    nnz_W = np.count_nonzero(W)
    # each weight non-zero contributes d * (d + 1) entries (minus exact zeros in X)
    assert B.nnz <= nnz_W * d * (d + 1)
    assert B.nnz >= nnz_W * d  # at least the constant-polynomial block


def test_point_frames_are_kron_with_identity() -> None:
    rng = np.random.default_rng(1)
    n, d, k = 7, 2, 4
    X = rng.standard_normal((n, d))
    W = _random_compact_weights(rng, n, k)
    B = sparse_lbs_jacobian(X, W, order=0)
    assert B.shape == (n * d, k * d)
    np.testing.assert_allclose(B.toarray(), np.kron(W, np.eye(d)), atol=1e-14)
    # a uniform translation of every node translates every vertex
    t = np.array([0.3, -0.7])
    u = (B @ np.tile(t, k)).reshape(-1, d)
    np.testing.assert_allclose(u, np.tile(t, (n, 1)), atol=1e-14)


def test_invalid_order_raises() -> None:
    X = np.zeros((3, 2))
    W = np.ones((3, 1))
    with pytest.raises(ValueError):
        sparse_lbs_jacobian(X, W, order=2)
