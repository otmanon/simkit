"""Tests for ``simkit.tetrahedron_volumes``."""

from __future__ import annotations

import numpy as np

from simkit.tetrahedron_volumes import tetrahedron_volumes


def _unit_tet() -> tuple[np.ndarray, np.ndarray]:
    """Standard unit tetrahedron with volume 1/6."""
    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    T = np.array([[0, 1, 2, 3]])
    return X, T


def test_unit_tetrahedron_volume_is_one_sixth() -> None:
    X, T = _unit_tet()
    V = tetrahedron_volumes(X, T)
    assert V.shape == (1,)
    assert np.isclose(V[0], 1.0 / 6.0, rtol=1e-12)
