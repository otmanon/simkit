"""Tests for ``simkit.rotation_gradient``."""

from __future__ import annotations

import numpy as np

from simkit.rotation_gradient import rotation_gradient_F


def test_rotation_gradient_shape_on_batch_of_F() -> None:
    rng = np.random.default_rng(1)
    n = 5

    F2 = rng.standard_normal((n, 2, 2)) + np.eye(2)
    K2 = rotation_gradient_F(F2)
    assert K2.shape == (n, 4, 4)

    F3 = rng.standard_normal((n, 3, 3)) + np.eye(3)
    K3 = rotation_gradient_F(F3)
    assert K3.shape == (n, 9, 9)
