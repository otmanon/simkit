"""Tests for ``simkit.spectral_basis_localization``."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cvxopt")
pytest.importorskip("sklearn")
pytestmark = [pytest.mark.solvers, pytest.mark.learn]

from simkit.spectral_basis_localization import spectral_basis_localization


def _unit_tet() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    T = np.array([[0, 1, 2, 3]])
    return X, T


def test_spectral_basis_localization_partition_of_unity_on_unit_tet() -> None:
    X, T = _unit_tet()
    m = 2

    Wh, cI = spectral_basis_localization(X, T, m, order=1)

    assert Wh.shape == (X.shape[0], m)
    assert cI.shape == (m,)
    assert np.allclose(Wh.sum(axis=1), 1.0, atol=1e-8)
