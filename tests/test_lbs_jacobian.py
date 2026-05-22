"""Tests for ``simkit.lbs_jacobian``."""

from __future__ import annotations

import numpy as np

from simkit.lbs_jacobian import lbs_jacobian


def test_lbs_jacobian_shape() -> None:
    n, d, k = 4, 3, 2
    rng = np.random.default_rng(1)
    V = rng.standard_normal((n, d))
    W = rng.random((n, k))
    W /= W.sum(axis=1, keepdims=True)

    J = lbs_jacobian(V, W)
    assert J.shape == (n * d, k * (d + 1) * d)
