"""Tests for ``simkit.lbs_jacobian``."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

from simkit.lbs_jacobian import lbs_jacobian


def _random_compact_weights(rng, n, k):
    W = rng.random((n, k))
    W[rng.random((n, k)) < 0.4] = 0.0  # compact support
    W[W.sum(axis=1) == 0, 0] = 1.0
    return W / W.sum(axis=1, keepdims=True)


def test_lbs_jacobian_shape() -> None:
    n, d, k = 4, 3, 2
    rng = np.random.default_rng(1)
    V = rng.standard_normal((n, d))
    W = rng.random((n, k))
    W /= W.sum(axis=1, keepdims=True)

    J = lbs_jacobian(V, W)
    assert isinstance(J, np.ndarray)
    assert J.shape == (n * d, k * (d + 1) * d)


def test_dense_layout_single_frame_is_homogeneous_coordinates() -> None:
    # One frame, all weights 1: J rows are [x, y, 1] placed per component.
    V = np.array([[2.0, 3.0]])
    W = np.ones((1, 1))
    J = lbs_jacobian(V, W)
    expected = np.array([[2.0, 0.0, 3.0, 0.0, 1.0, 0.0],
                         [0.0, 2.0, 0.0, 3.0, 0.0, 1.0]])
    np.testing.assert_allclose(J, expected)


@pytest.mark.parametrize("d", [2, 3])
def test_sparse_matches_dense(d) -> None:
    rng = np.random.default_rng(0)
    n, k = 9, 3
    V = rng.standard_normal((n, d))
    W = _random_compact_weights(rng, n, k)
    Js = lbs_jacobian(V, W, sparse=True)
    assert sps.issparse(Js) and Js.format == "csr"
    assert Js.shape == (n * d, k * d * (d + 1))
    np.testing.assert_allclose(Js.toarray(), lbs_jacobian(V, W), atol=1e-14)


def test_sparse_and_dense_accept_sparse_weights() -> None:
    rng = np.random.default_rng(2)
    V = rng.standard_normal((6, 2))
    W = _random_compact_weights(rng, 6, 2)
    Ws = sps.csr_matrix(W)
    np.testing.assert_allclose(lbs_jacobian(V, Ws), lbs_jacobian(V, W), atol=1e-14)
    assert (lbs_jacobian(V, Ws, sparse=True) != lbs_jacobian(V, W, sparse=True)).nnz == 0


def test_sparsity_follows_weights() -> None:
    rng = np.random.default_rng(3)
    n, d, k = 20, 2, 5
    V = rng.standard_normal((n, d))
    W = _random_compact_weights(rng, n, k)
    J = lbs_jacobian(V, W, sparse=True)
    nnz_W = np.count_nonzero(W)
    # each weight non-zero contributes d * (d + 1) entries (minus exact zeros in V)
    assert J.nnz <= nnz_W * d * (d + 1)
    assert J.nnz >= nnz_W * d  # at least the constant-polynomial block
    assert J.nnz < np.prod(J.shape)
