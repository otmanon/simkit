"""Tests for ``simkit.polar_svd``."""

from __future__ import annotations

import numpy as np

from simkit.polar_svd import polar_svd


def test_polar_svd_rotation_orthogonal_stretch_symmetric_psd() -> None:
    rng = np.random.default_rng(3)
    F = rng.standard_normal((5, 3, 3))
    R, S = polar_svd(F)
    for k in range(F.shape[0]):
        Rk, Sk = R[k], S[k]
        assert np.allclose(Rk @ Rk.T, np.eye(3), atol=1e-10)
        assert np.allclose(Rk.T @ Rk, np.eye(3), atol=1e-10)
        assert np.allclose(Sk, Sk.T, atol=1e-10)
        assert np.allclose(F[k], Rk @ Sk, atol=1e-10)

    A = rng.standard_normal((3, 3))
    F_spd = (A.T @ A)[None, :, :]
    _, S_spd = polar_svd(F_spd)
    assert np.all(np.linalg.eigvalsh(S_spd[0]) >= -1e-10)
