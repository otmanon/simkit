"""Tests for ``simkit.spectral_cubature``."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytestmark = pytest.mark.learn

from simkit.spectral_cubature import spectral_cubature


def _two_tet_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Two tetrahedra sharing a face (five vertices)."""
    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    T = np.array([[0, 1, 2, 3], [1, 4, 2, 3]])
    return X, T


def test_spectral_cubature_cluster_volumes_are_positive() -> None:
    X, T = _two_tet_mesh()
    rng = np.random.default_rng(1)
    W = rng.standard_normal((X.shape[0], 4))
    k = 2

    lI, mc = spectral_cubature(X, T, W, k)

    assert lI.shape == (k,)
    assert mc.shape == (k,)
    assert np.all(mc > 0)
    assert mc.sum() > 0
