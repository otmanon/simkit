"""Tests for ``simkit.affine_embedding_matrix``: embedded points reproduce their rest
positions and follow affine (hence rigid) motions of the candidate vertices exactly."""

from __future__ import annotations

import numpy as np

from simkit.affine_embedding_matrix import affine_embedding_matrix


def test_affine_embedding_reproduces_points_and_affine_motion() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 3))
    Q = rng.standard_normal((5, 3))
    cand = [np.arange(40)] * 3 + [np.arange(20)] * 2
    G = affine_embedding_matrix(X, Q, cand, k=12)
    assert G.shape == (15, 120)
    assert np.allclose((G @ X.reshape(-1)).reshape(-1, 3), Q)
    A, t = rng.standard_normal((3, 3)), rng.standard_normal(3)
    assert np.allclose((G @ (X @ A.T + t).reshape(-1)).reshape(-1, 3), Q @ A.T + t)
