"""Tests for ``simkit.orthonormalize``."""

from __future__ import annotations

import numpy as np

from simkit.orthonormalize import orthonormalize


def test_orthonormalize_columns_are_orthonormal_wrt_identity() -> None:
    rng = np.random.default_rng(0)
    B = rng.standard_normal((8, 4))
    B_ortho = orthonormalize(B, M=None)
    gram = B_ortho.T @ B_ortho
    assert np.allclose(gram, np.eye(B_ortho.shape[1]), atol=1e-10)
