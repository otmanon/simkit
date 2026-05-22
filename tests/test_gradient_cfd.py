"""Tests for ``simkit.gradient_cfd``."""

from __future__ import annotations

import numpy as np
import pytest

from simkit.gradient_cfd import gradient_cfd

FD_STEP = 1e-5
TOL = 1e-4


def test_gradient_cfd_quadratic_gives_two_x() -> None:
    y = np.array([0.3, -1.2, 2.0])

    def phi(x: np.ndarray) -> np.ndarray:
        return np.sum(x**2)

    g = gradient_cfd(phi, y, FD_STEP)

    assert g.shape == y.shape
    assert np.allclose(g, 2.0 * y, atol=TOL)
