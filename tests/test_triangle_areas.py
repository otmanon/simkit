"""Tests for ``simkit.triangle_areas``."""

from __future__ import annotations

import numpy as np

from simkit.gradient_cfd import gradient_cfd
from simkit.triangle_areas import (
    triangle_area_element,
    triangle_area_gradient_element,
)

FD_STEP = 1e-6
GRAD_TOL = 1e-5


def _right_triangle() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit right triangle in the xy plane."""
    x0 = np.array([[0.0, 0.0, 0.0]])
    x1 = np.array([[1.0, 0.0, 0.0]])
    x2 = np.array([[0.0, 1.0, 0.0]])
    return x0, x1, x2


def test_triangle_area_element_is_half() -> None:
    x0, x1, x2 = _right_triangle()
    area = triangle_area_element(x0, x1, x2)
    assert area.shape == (1,)
    assert np.isclose(area[0], 0.5, rtol=1e-12)


def test_triangle_area_gradient_element_matches_fd() -> None:
    x0, x1, x2 = _right_triangle()
    y = np.concatenate([x0, x1, x2], axis=1).flatten()

    def area_flat(coords: np.ndarray) -> np.ndarray:
        c = coords.reshape(1, 9)
        return triangle_area_element(c[:, :3], c[:, 3:6], c[:, 6:]).flatten()

    g_fd = gradient_cfd(area_flat, y, FD_STEP).reshape(1, 9)
    g = triangle_area_gradient_element(x0, x1, x2)
    assert g.shape == (1, 9)
    assert np.allclose(g, g_fd, atol=GRAD_TOL)
