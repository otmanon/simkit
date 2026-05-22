"""Tests for ``simkit.vectorized_trace``."""

from __future__ import annotations

import numpy as np

from simkit.vectorized_trace import vectorized_trace


def _stack_mats(mats: np.ndarray) -> np.ndarray:
    """Row-major flatten of each ``d x d`` block in the stacked layout."""
    n, d, _ = mats.shape
    vec = np.zeros(n * d * d, dtype=mats.dtype)
    for i in range(n):
        vec[i * d * d : (i + 1) * d * d] = mats[i].reshape(-1, order="C")
    return vec


def test_vectorized_trace_shape_and_action() -> None:
    n, d = 3, 2
    T = vectorized_trace(n, d)

    assert T.shape == (n, n * d * d)

    mats = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[10.0, 20.0], [30.0, 40.0]],
            [[-1.0, 0.5], [0.25, 2.0]],
        ]
    )
    vec = _stack_mats(mats)
    traces = T @ vec

    expected = np.array([np.trace(mats[i]) for i in range(n)])
    assert traces.shape == (n,)
    assert np.allclose(traces, expected, rtol=1e-12)
