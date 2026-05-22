"""Tests for ``simkit.stretch_gradient``."""

from __future__ import annotations

import numpy as np

from simkit.stretch_gradient import stretch_gradient


def test_stretch_gradient_shape() -> None:
    dim = 3
    t = 5
    F = np.tile(np.eye(dim), (t, 1, 1))

    dSdF = stretch_gradient(F)

    assert dSdF.shape == (t, dim, dim, dim, dim)
