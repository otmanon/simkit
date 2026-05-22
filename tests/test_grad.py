"""Tests for ``simkit.grad``."""

from __future__ import annotations

import numpy as np
import pytest

from simkit.grad import grad


def _unit_triangle() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    F = np.array([[0, 1, 2]])
    U = X.copy()
    return X, F, U


def test_grad_on_unit_triangle_returns_expected_shapes() -> None:
    X, F, U = _unit_triangle()

    D, HXHi = grad(X, F, U)

    assert D.shape == (2, 2)
    assert HXHi.shape == (1, 3, 2)
