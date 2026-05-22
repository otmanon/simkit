"""Tests for ``simkit.fast_sandwich_transform_clustered``."""

from __future__ import annotations

import numpy as np
import pytest

from simkit.fast_sandwich_transform_clustered import fast_sandwich_transform_clustered


def _example_operators() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dim = 3
    n = 2
    m = 4
    block = dim * dim

    A = np.zeros((m, block * n))
    for elem in range(n):
        base_col = elem * block
        for i in range(m):
            A[i, base_col + i] = float(elem + 1)

    B = np.zeros((block * n, m))
    for elem in range(n):
        base_row = elem * block
        for j in range(m):
            B[base_row + j, j] = 0.5 * (elem + 1)

    labels = np.array([0, 1])
    return A, B, labels


def test_fast_sandwich_transform_clustered_output_shape() -> None:
    A, B, labels = _example_operators()
    fst = fast_sandwich_transform_clustered(A, B, labels)

    r = np.tile(np.eye(3), (2, 1, 1))
    arb = fst.eval(r)

    assert arb.shape == (A.shape[0], B.shape[1])


def test_fast_sandwich_transform_clustered_bilinear_in_rotations() -> None:
    A, B, labels = _example_operators()
    fst = fast_sandwich_transform_clustered(A, B, labels)

    r1 = np.tile(np.eye(3), (2, 1, 1))
    r2 = np.tile(
        np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        (2, 1, 1),
    )
    c1, c2 = 0.35, 0.65
    r = c1 * r1 + c2 * r2

    lhs = fst.eval(r)
    rhs = c1 * fst.eval(r1) + c2 * fst.eval(r2)

    assert np.allclose(lhs, rhs, rtol=1e-12, atol=1e-12)
