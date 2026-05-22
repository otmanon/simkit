"""Tests for ``simkit.subspace_rotation``."""

from __future__ import annotations

import numpy as np

from simkit.subspace_rotation import subspace_rotation


def _unit_tet() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    T = np.array([[0, 1, 2, 3]])
    return X, T


def test_subspace_rotation_matrices_are_orthogonal_per_cluster() -> None:
    X, T = _unit_tet()
    dim = X.shape[1]
    r = 6
    rng = np.random.default_rng(5)
    B = rng.standard_normal((X.shape[0] * dim, r))
    z = rng.standard_normal((r, 1)) * 0.05

    R = subspace_rotation(z, B, X, T)

    assert R.ndim == 3
    assert R.shape[1:] == (dim, dim)
    for k in range(R.shape[0]):
        Rk = R[k]
        assert np.allclose(Rk @ Rk.T, np.eye(dim), atol=1e-8)
        assert np.allclose(Rk.T @ Rk, np.eye(dim), atol=1e-8)
