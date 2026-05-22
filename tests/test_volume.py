"""Tests for ``simkit.volume``."""

from __future__ import annotations

import numpy as np

from simkit.volume import volume


def test_volume_dispatches_edge_length() -> None:
    V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    F = np.array([[0, 1]])
    vol = volume(V, F)
    assert vol.shape == (1, 1)
    assert np.isclose(vol[0, 0], 1.0, rtol=1e-12)


def test_volume_dispatches_triangle_area() -> None:
    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    F = np.array([[0, 1, 2]])
    vol = volume(V, F)
    assert vol.shape == (1, 1)
    assert np.isclose(vol[0, 0], 0.5, rtol=1e-12)


def test_volume_dispatches_tetrahedron_volume() -> None:
    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    F = np.array([[0, 1, 2, 3]])
    vol = volume(V, F)
    assert vol.shape == (1, 1)
    assert np.isclose(vol[0, 0], 1.0 / 6.0, rtol=1e-12)
