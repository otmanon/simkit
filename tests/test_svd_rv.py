"""Tests for ``simkit.svd_rv``."""

from __future__ import annotations

import numpy as np

from simkit.svd_rv import svd_rv


def test_svd_rv_rotation_factor_is_orthogonal() -> None:
    rng = np.random.default_rng(4)
    F = rng.standard_normal((6, 3, 3))

    U, _S, V = svd_rv(F)

    for k in range(F.shape[0]):
        R = U[k] @ V[k].T
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)
        assert np.linalg.det(R) >= 0
