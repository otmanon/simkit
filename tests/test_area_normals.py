"""Tests for ``simkit.area_normals``."""

from __future__ import annotations

import numpy as np
import pytest

from simkit.area_normals import (
    area_normal_element,
    area_normal_gradient_element,
    area_normal_hessian_element,
)
from simkit.gradient_cfd import gradient_cfd
from simkit.hessian_cfd import hessian_cfd

FD_STEP = 1e-6
GRAD_TOL = 1e-5
HESS_TOL = 1e-4


def _right_triangle() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit right triangle in the xy plane."""
    x0 = np.array([[0.0, 0.0, 0.0]])
    x1 = np.array([[1.0, 0.0, 0.0]])
    x2 = np.array([[0.0, 1.0, 0.0]])
    return x0, x1, x2


def test_area_normal_magnitude_is_twice_triangle_area() -> None:
    x0, x1, x2 = _right_triangle()
    n = area_normal_element(x0, x1, x2)
    assert n.shape == (1, 3)
    assert np.isclose(np.linalg.norm(n), 1.0, rtol=1e-12)
    assert np.isclose(n[0, 2], -1.0, rtol=1e-12)


def test_area_normal_gradient_matches_fd() -> None:
    x0, x1, x2 = _right_triangle()
    y = np.concatenate([x0, x1, x2], axis=1).flatten()

    def normal_flat(coords: np.ndarray) -> np.ndarray:
        c = coords.reshape(1, 9)
        return area_normal_element(c[:, :3], c[:, 3:6], c[:, 6:]).flatten()

    g_fd = gradient_cfd(normal_flat, y, FD_STEP).reshape(3, 9)
    g = area_normal_gradient_element(x0, x1, x2)[0]
    assert np.allclose(g, g_fd, atol=GRAD_TOL)


def test_area_normal_hessian_is_constant_and_matches_fd() -> None:
    x0, x1, x2 = _right_triangle()
    y = np.concatenate([x0, x1, x2], axis=1).flatten()

    def normal_flat(coords: np.ndarray) -> np.ndarray:
        c = coords.reshape(1, 9)
        return area_normal_element(c[:, :3], c[:, 3:6], c[:, 6:]).flatten()

    h_fd = hessian_cfd(normal_flat, y, FD_STEP).reshape(3, 9, 9)
    h = area_normal_hessian_element(x0, x1, x2)[0]
    assert np.allclose(h, h_fd, atol=HESS_TOL)

    # Hessian does not depend on geometry; only the batch size matters.
    x0b = np.vstack([x0, x0 + 0.3])
    x1b = np.vstack([x1, x1 + 0.1])
    x2b = np.vstack([x2, x2 - 0.2])
    hb = area_normal_hessian_element(x0b, x1b, x2b)
    assert hb.shape == (2, 3, 9, 9)
    assert np.allclose(hb[0], hb[1])
