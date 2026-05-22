"""Tests for ``simkit.normals``."""

from __future__ import annotations

import numpy as np

from simkit.normals import normals


def test_unit_triangle_normal_has_unit_magnitude() -> None:
    X = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    F = np.array([[0, 1, 2]])
    n = normals(X, F)
    assert n.shape == (1, 3)
    assert np.isclose(np.linalg.norm(n), 1.0, rtol=1e-12)
