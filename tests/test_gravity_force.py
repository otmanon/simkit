"""Tests for ``simkit.gravity_force``."""

from __future__ import annotations

import numpy as np
import pytest

from simkit.gravity_force import gravity_force


def _unit_tet() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    T = np.array([[0, 1, 2, 3]])
    return X, T


def test_gravity_force_shape_matches_vertex_dofs() -> None:
    X, T = _unit_tet()

    g = gravity_force(X, T)

    assert g.shape == X.shape
