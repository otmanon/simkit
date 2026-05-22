"""Tests for ``simkit.gaussian_rbf``."""

from __future__ import annotations

import numpy as np
import pytest

from simkit.gaussian_rbf import gaussian_rbf


def test_gaussian_rbf_at_center_is_one() -> None:
    center = np.array([[1.0, 2.0, 3.0]])
    sigma = 0.5
    p = np.array([[1.0, 2.0, 3.0, sigma]])

    phi = gaussian_rbf(center, p)

    assert phi.shape == (1, 1)
    assert np.isclose(phi[0, 0], 1.0, rtol=1e-12)
