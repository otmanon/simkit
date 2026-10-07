"""Tests for ``simkit.sparse_meshless_methods_basis``."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps

from simkit.lbs_affine_coordinates import lbs_affine_coordinates
from simkit.lbs_jacobian import lbs_jacobian
from simkit.sparse_meshless_methods_basis import sparse_meshless_methods_basis


def test_output_shapes_and_properties(stiff_bar_strip) -> None:
    X, T, ym = stiff_bar_strip
    W, B, labels, nodes = sparse_meshless_methods_basis(X, T, ym, n_nodes=8)
    k = len(nodes)
    assert k == 8
    assert sps.issparse(W) and W.shape == (X.shape[0], k)
    assert sps.issparse(B) and B.shape == (X.shape[0] * 2, k * 2 * 3)
    assert labels.shape == (T.shape[0],)
    assert labels.min() >= 0 and labels.max() < k
    np.testing.assert_allclose(W.sum(axis=1).A.ravel(), 1.0, atol=1e-12)
    np.testing.assert_allclose(W[nodes].toarray(), np.eye(k), atol=1e-12)


def test_affine_patch_test(stiff_bar_strip) -> None:
    X, T, ym = stiff_bar_strip
    W, B, labels, nodes = sparse_meshless_methods_basis(X, T, ym, n_nodes=8)
    k = len(nodes)
    A = np.array([[1.3, 0.2], [-0.1, 0.8]])
    t = np.array([0.5, -0.2])
    z = lbs_affine_coordinates(k, A, t)
    np.testing.assert_allclose((B @ z).reshape(-1, 2), X @ A.T + t, atol=1e-12)
    z0 = lbs_affine_coordinates(k, np.eye(2), np.zeros(2))
    np.testing.assert_allclose((B @ z0).reshape(-1, 2), X, atol=1e-12)


def test_matches_dense_lbs_jacobian(small_stiff_bar_strip) -> None:
    X, T, ym = small_stiff_bar_strip
    W, B, _, _ = sparse_meshless_methods_basis(X, T, ym, n_nodes=5)
    np.testing.assert_allclose(B.toarray(), lbs_jacobian(X, W.toarray()), atol=1e-14)


def test_return_distances_and_point_frames(small_stiff_bar_strip) -> None:
    X, T, ym = small_stiff_bar_strip
    out = sparse_meshless_methods_basis(
        X, T, ym, n_nodes=4, frame_order=0, return_distances=True
    )
    assert len(out) == 5
    W, B, labels, nodes, D = out
    assert D.shape == (4, X.shape[0])
    assert np.all(np.isfinite(D))
    np.testing.assert_allclose(D[np.arange(4), nodes], 0.0)
    assert B.shape == (X.shape[0] * 2, 4 * 2)


def test_seed_nodes_and_edge_reduce_are_forwarded(small_stiff_bar_strip) -> None:
    X, T, ym = small_stiff_bar_strip
    _, _, _, nodes = sparse_meshless_methods_basis(
        X, T, ym, n_nodes=4, seed_nodes=[0, 3], edge_reduce="min", lloyd_iters=2
    )
    assert nodes[0] == 0 and nodes[1] == 3


def test_3d_surface_mesh(small_stiff_bar_strip) -> None:
    X2, T, ym = small_stiff_bar_strip
    X = np.c_[X2, np.sin(X2[:, 0])]
    W, B, labels, nodes = sparse_meshless_methods_basis(X, T, ym, n_nodes=5)
    assert B.shape == (X.shape[0] * 3, 5 * 3 * 4)
    np.testing.assert_allclose(B.toarray(), lbs_jacobian(X, W.toarray()), atol=1e-14)


def test_deterministic(small_stiff_bar_strip) -> None:
    X, T, ym = small_stiff_bar_strip
    a = sparse_meshless_methods_basis(X, T, ym, n_nodes=5)
    b = sparse_meshless_methods_basis(X, T, ym, n_nodes=5)
    np.testing.assert_array_equal(a[3], b[3])
    assert (a[0] != b[0]).nnz == 0
    np.testing.assert_array_equal(a[2], b[2])
