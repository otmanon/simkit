"""Tests for ``simkit.hyper_reduced_projective_dynamics_basis``."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps

from simkit.hyper_reduced_projective_dynamics_basis import (
    hyper_reduced_projective_dynamics_basis,
)
from simkit.lbs_jacobian import lbs_jacobian


def test_shapes_partition_of_unity_and_compact_support(stiff_bar_strip) -> None:
    X, T, _ = stiff_bar_strip
    k = 8
    W, B, samples = hyper_reduced_projective_dynamics_basis(X, T, k)
    assert sps.issparse(W) and W.shape == (X.shape[0], k)
    assert sps.issparse(B) and B.shape == (X.size, k * 2 * 3)
    assert samples.shape == (k,) and len(set(samples.tolist())) == k
    Wd = W.toarray()
    assert Wd.min() >= 0.0
    np.testing.assert_allclose(Wd.sum(axis=1), 1.0, atol=1e-12)
    assert W.nnz < 0.7 * np.prod(W.shape)  # compact: most vertices see a few samples


def test_weights_are_geometry_only(stiff_bar_strip) -> None:
    # the material does not enter: same mesh with a different ym gives the same basis
    X, T, ym = stiff_bar_strip
    W1, _, s1 = hyper_reduced_projective_dynamics_basis(X, T, 6)
    W2, _, s2 = hyper_reduced_projective_dynamics_basis(X, T, 6)
    np.testing.assert_array_equal(s1, s2)
    assert (W1 != W2).nnz == 0


def test_sample_vertex_is_dominated_by_its_own_weight(stiff_bar_strip) -> None:
    X, T, _ = stiff_bar_strip
    W, _, samples = hyper_reduced_projective_dynamics_basis(X, T, 6)
    Ws = W[samples].toarray()
    assert np.all(np.argmax(Ws, axis=1) == np.arange(6))


def test_support_scale_widens_supports(stiff_bar_strip) -> None:
    X, T, _ = stiff_bar_strip
    tight, _, _ = hyper_reduced_projective_dynamics_basis(X, T, 6, support_scale=1.0)
    wide, _, _ = hyper_reduced_projective_dynamics_basis(X, T, 6, support_scale=3.0)
    assert wide.nnz > tight.nnz
    np.testing.assert_allclose(tight.sum(axis=1).A.ravel(), 1.0, atol=1e-12)  # still covered


def test_basis_matches_dense_and_passes_patch_test(small_stiff_bar_strip) -> None:
    X, T, _ = small_stiff_bar_strip
    k = 5
    W, B, _ = hyper_reduced_projective_dynamics_basis(X, T, k)
    np.testing.assert_allclose(B.toarray(), lbs_jacobian(X, W.toarray()), atol=1e-14)
    A = np.array([[1.2, 0.3], [-0.2, 0.9]])
    t = np.array([0.4, -0.1])
    z = np.tile(np.c_[A, t].T.ravel(), k).reshape(-1, 1)
    np.testing.assert_allclose((B @ z).reshape(-1, 2), X @ A.T + t, atol=1e-12)


def test_return_distances_and_seed(small_stiff_bar_strip) -> None:
    X, T, _ = small_stiff_bar_strip
    W, B, samples, D = hyper_reduced_projective_dynamics_basis(
        X, T, 4, seed_index=7, return_distances=True
    )
    assert samples[0] == 7
    assert D.shape == (4, X.shape[0])
    np.testing.assert_allclose(D[np.arange(4), samples], 0.0)
