"""Tests for ``simkit.lbs_affine_coordinates``."""

from __future__ import annotations

import numpy as np
import pytest

from simkit.lbs_affine_coordinates import lbs_affine_coordinates
from simkit.lbs_jacobian import lbs_jacobian


def test_shape_and_layout() -> None:
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    t = np.array([5.0, 6.0])
    z = lbs_affine_coordinates(2, A, t)
    assert z.shape == (2 * 2 * 3, 1)
    # per node: poly x -> column 0 of A, poly y -> column 1, poly 1 -> t
    per_node = [1.0, 3.0, 2.0, 4.0, 5.0, 6.0]
    np.testing.assert_allclose(z.ravel(), per_node * 2)


@pytest.mark.parametrize("d", [2, 3])
def test_reproduces_affine_map_with_dense_and_sparse_bases(d) -> None:
    rng = np.random.default_rng(4)
    n, k = 10, 3
    X = rng.standard_normal((n, d))
    W = rng.random((n, k))
    W /= W.sum(axis=1, keepdims=True)
    A = rng.standard_normal((d, d))
    t = rng.standard_normal(d)
    z = lbs_affine_coordinates(k, A, t)
    expected = X @ A.T + t
    np.testing.assert_allclose((lbs_jacobian(X, W) @ z).reshape(-1, d), expected, atol=1e-12)
    np.testing.assert_allclose((lbs_jacobian(X, W, sparse=True) @ z).reshape(-1, d), expected, atol=1e-12)


def test_identity_gives_rest_state() -> None:
    rng = np.random.default_rng(5)
    X = rng.standard_normal((8, 2))
    W = np.ones((8, 1))
    z = lbs_affine_coordinates(1, np.eye(2), np.zeros(2))
    np.testing.assert_allclose((lbs_jacobian(X, W, sparse=True) @ z).reshape(-1, 2), X, atol=1e-14)
