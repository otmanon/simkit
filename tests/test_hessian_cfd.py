"""Tests for ``simkit.hessian_cfd``."""

from __future__ import annotations

import numpy as np
import pytest

from simkit.hessian_cfd import hessian_cfd

FD_STEP = 1e-4
TOL = 5e-3


def test_hessian_cfd_quadratic_gives_two_identity() -> None:
    y = np.array([0.5, -1.0, 2.0])

    def phi(x: np.ndarray) -> np.ndarray:
        return np.sum(x**2)

    H = hessian_cfd(phi, y, FD_STEP)

    assert H.shape == (y.shape[0], y.shape[0])
    assert np.allclose(H, 2.0 * np.eye(y.shape[0]), atol=TOL)
