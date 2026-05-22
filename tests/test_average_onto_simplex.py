"""Tests for ``simkit.average_onto_simplex``."""

from __future__ import annotations

import numpy as np

from simkit.average_onto_simplex import average_onto_simplex


def test_average_onto_triangle_is_vertex_mean() -> None:
    A = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
    T = np.array([[0, 1, 2]])
    At = average_onto_simplex(A, T)
    assert At.shape == (1, 2)
    assert np.allclose(At[0], A.mean(axis=0))


def test_average_onto_tet_matches_manual_mean() -> None:
    A = np.arange(12, dtype=float).reshape(4, 3)
    T = np.array([[0, 1, 2, 3]])
    At = average_onto_simplex(A, T)
    assert np.allclose(At[0], np.mean(A, axis=0))


def test_average_over_multiple_simplices() -> None:
    A = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    T = np.array([[0, 1, 2], [1, 2, 3]])
    At = average_onto_simplex(A, T)
    assert At.shape == (2, 2)
    assert np.allclose(At[0], A[[0, 1, 2]].mean(axis=0))
    assert np.allclose(At[1], A[[1, 2, 3]].mean(axis=0))
