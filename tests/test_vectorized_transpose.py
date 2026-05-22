"""Tests for ``simkit.vectorized_transpose``."""

from __future__ import annotations

import numpy as np

from simkit.vectorized_transpose import vectorized_transpose


def _stack_mats(mats: np.ndarray) -> np.ndarray:
    """Row-major flatten of each ``d x d`` block in the stacked layout."""
    n, d, _ = mats.shape
    vec = np.zeros(n * d * d, dtype=mats.dtype)
    for i in range(n):
        vec[i * d * d : (i + 1) * d * d] = mats[i].reshape(-1, order="C")
    return vec


def _unstack_mats(vec: np.ndarray, n: int, d: int) -> np.ndarray:
    mats = np.zeros((n, d, d), dtype=vec.dtype)
    for i in range(n):
        mats[i] = vec[i * d * d : (i + 1) * d * d].reshape(d, d, order="C")
    return mats


def test_vectorized_transpose_permutes_blocks() -> None:
    n, d = 2, 3
    T = vectorized_transpose(n, d)

    assert T.shape == (n * d * d, n * d * d)

    rng = np.random.default_rng(0)
    mats = rng.standard_normal((n, d, d))
    vec = _stack_mats(mats)
    transposed_vec = T @ vec
    transposed_mats = _unstack_mats(transposed_vec, n, d)

    assert np.allclose(transposed_mats, np.transpose(mats, (0, 2, 1)), rtol=1e-12)
