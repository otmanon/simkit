"""Tests for ``simkit.backtracking_line_search``."""

from __future__ import annotations

import numpy as np
import pytest

from simkit.backtracking_line_search import backtracking_line_search


def _quadratic(x: np.ndarray) -> float:
    return float(0.5 * np.dot(x, x))


def test_full_step_accepted_on_quadratic() -> None:
    x0 = np.array([2.0, -1.0])
    g = x0.copy()
    dx = -g
    t, x, fx = backtracking_line_search(_quadratic, x0, g, dx)
    assert t == pytest.approx(1.0)
    assert np.allclose(x, np.zeros_like(x0))
    assert fx == pytest.approx(0.0)


def test_armijo_condition_holds_at_returned_step() -> None:
    alpha = 0.01
    x0 = np.array([1.5])
    g = np.array([2.0 * x0[0]])
    dx = np.array([-1.0])

    def stiff_quartic(x: np.ndarray) -> float:
        return float(x[0] ** 4)

    t, x, fx = backtracking_line_search(
        stiff_quartic, x0, g, dx, alpha=alpha, beta=0.5, max_iter=50
    )
    fx0 = stiff_quartic(x0)
    assert fx <= fx0 + alpha * t * (g.T @ dx) + 1e-12
    assert np.allclose(x, x0 + t * dx)


def test_zero_step_when_no_progress_within_max_iter() -> None:
    x0 = np.array([0.0])
    g = np.array([1.0])
    dx = np.array([1.0])  # ascent direction; Armijo never satisfied

    t, x, fx = backtracking_line_search(
        _quadratic, x0, g, dx, max_iter=5, beta=0.5
    )
    assert t == 0.0
    assert np.allclose(x, x0)
    assert fx == pytest.approx(_quadratic(x0))
