"""Tests for ``simkit.uniform_line``."""

from __future__ import annotations

import numpy as np

from simkit.uniform_line import uniform_line


def test_uniform_line_vertex_count_and_endpoints() -> None:
    n = 7
    X = uniform_line(n)

    assert X.shape == (n, 1)
    assert np.isclose(X[0, 0], 0.0, rtol=1e-12)
    assert np.isclose(X[-1, 0], 1.0, rtol=1e-12)


def test_uniform_line_with_simplex_connectivity() -> None:
    n = 5
    X, T = uniform_line(n, return_simplex=True)

    assert X.shape == (n, 1)
    assert T.shape == (n - 1, 2)
    assert T[0, 0] == 0
    assert T[-1, 1] == n - 1
