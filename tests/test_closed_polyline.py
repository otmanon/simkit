"""Tests for ``simkit.closed_polyline``."""

from __future__ import annotations

import numpy as np

from simkit.closed_polyline import closed_polyline


def test_closed_polyline_wraps_last_edge() -> None:
    V = np.random.default_rng(0).random((7, 3))
    E = closed_polyline(V)
    assert E.shape == (7, 2)
    assert np.allclose(E[:-1, 0] + 1, E[:-1, 1])
    assert E[-1, 0] == 6
    assert E[-1, 1] == 0


def test_closed_polyline_two_vertices() -> None:
    V = np.array([[0.0, 0.0], [1.0, 0.0]])
    E = closed_polyline(V)
    expected = np.array([[0, 1], [1, 0]])
    assert np.array_equal(E, expected)
