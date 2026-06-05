"""Tests for ``simkit.outline_areas``.

Covers three things:

* signed areas of meshes with analytically known areas (triangle, square,
  regular polygon approximating a circle), including the sign flip under
  reversed orientation;
* the per-outline gradient against a central finite difference;
* the per-outline (constant) Hessian against a central finite difference.
"""

from __future__ import annotations

import numpy as np

from simkit.closed_polyline import closed_polyline
from simkit.gradient_cfd import gradient_cfd
from simkit.hessian_cfd import hessian_cfd
from simkit.outline_areas import (
    outline_areas,
    outline_areas_gradient,
    outline_areas_hessian,
    outline_selection_matrices,
)

FD_STEP = 1e-6
GRAD_TOL = 1e-6
HESS_TOL = 1e-4


def _regular_polygon(n: int, r: float = 1.0) -> np.ndarray:
    """``n`` vertices evenly spaced counter-clockwise on a circle of radius ``r``."""
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


# ---------------------------------------------------------------------------
# Known signed areas.
# ---------------------------------------------------------------------------
def test_unit_square_area_is_one() -> None:
    X = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    E = closed_polyline(X)

    areas = outline_areas(X, [E])

    assert areas.shape == (1,)
    assert np.isclose(areas[0], 1.0, rtol=1e-12)


def test_right_triangle_area_is_half() -> None:
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    E = closed_polyline(X)

    areas = outline_areas(X, [E])

    assert np.isclose(areas[0], 0.5, rtol=1e-12)


def test_scaled_quad_area() -> None:
    # 3-wide, 2-tall axis-aligned rectangle -> area 6.
    X = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 2.0], [0.0, 2.0]])
    E = closed_polyline(X)

    assert np.isclose(outline_areas(X, [E])[0], 6.0, rtol=1e-12)


def test_regular_polygon_approximates_circle() -> None:
    # Area of a regular n-gon inscribed in radius r is (1/2) n r^2 sin(2pi/n),
    # which converges to pi r^2 as n grows.
    r = 1.3
    n = 2000
    X = _regular_polygon(n, r)
    E = closed_polyline(X)

    area = outline_areas(X, [E])[0]
    exact_ngon = 0.5 * n * r ** 2 * np.sin(2.0 * np.pi / n)

    assert np.isclose(area, exact_ngon, rtol=1e-12)
    assert np.isclose(area, np.pi * r ** 2, rtol=1e-5)


def test_orientation_sets_sign() -> None:
    X = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    E_ccw = closed_polyline(X)
    E_cw = E_ccw[::-1, ::-1]  # reverse traversal -> clockwise loop

    ccw = outline_areas(X, [E_ccw])[0]
    cw = outline_areas(X, [E_cw])[0]

    assert ccw > 0.0
    assert cw < 0.0
    assert np.isclose(ccw, -cw, rtol=1e-12)


def test_multiple_chambers_returns_one_area_each() -> None:
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    tri = np.array([[5.0, 0.0], [7.0, 0.0], [5.0, 2.0]])
    X = np.vstack([square, tri])
    E0 = closed_polyline(square)              # vertices 0..3
    E1 = closed_polyline(tri) + 4             # vertices 4..6

    areas = outline_areas(X, [E0, E1])

    assert areas.shape == (2,)
    assert np.isclose(areas[0], 1.0, rtol=1e-12)
    assert np.isclose(areas[1], 2.0, rtol=1e-12)  # base 2, height 2 -> area 2


# ---------------------------------------------------------------------------
# Finite-difference checks of the gradient and Hessian.
# ---------------------------------------------------------------------------
def _wonky_pentagon() -> np.ndarray:
    """A non-degenerate, irregular CCW polygon to exercise the derivatives."""
    return np.array(
        [
            [0.0, 0.0],
            [2.0, -0.3],
            [2.4, 1.1],
            [1.0, 2.0],
            [-0.5, 1.2],
        ]
    )


def test_gradient_matches_finite_difference() -> None:
    X = _wonky_pentagon()
    E = closed_polyline(X)

    # All vertices belong to the single outline, so the local interleaved DOFs
    # [x0, y0, x1, y1, ...] coincide with X.flatten().
    def area_flat(z: np.ndarray) -> np.ndarray:
        return outline_areas(z.reshape(-1, 2), [E])

    z = X.flatten()
    g_fd = gradient_cfd(area_flat, z, FD_STEP).reshape(1, -1)
    g = outline_areas_gradient(X, [E])[0]

    assert g.shape == (1, 2 * X.shape[0])
    assert np.allclose(g, g_fd, atol=GRAD_TOL)


def test_hessian_matches_finite_difference() -> None:
    X = _wonky_pentagon()
    E = closed_polyline(X)

    def area_scalar(z: np.ndarray) -> np.ndarray:
        return outline_areas(z.reshape(-1, 2), [E])

    z = X.flatten()
    H_fd = hessian_cfd(area_scalar, z, 1e-3).reshape(2 * X.shape[0], 2 * X.shape[0])
    H = outline_areas_hessian(X, [E])[0]

    assert H.shape == (2 * X.shape[0], 2 * X.shape[0])
    assert np.allclose(H, H.T, atol=1e-12)          # symmetric
    assert np.allclose(H, H_fd, atol=HESS_TOL)


def test_hessian_is_constant() -> None:
    # Signed area is bilinear, so the Hessian must not depend on the positions.
    E = closed_polyline(_wonky_pentagon())
    H0 = outline_areas_hessian(_wonky_pentagon(), [E])[0]
    H1 = outline_areas_hessian(_wonky_pentagon() * 3.0 + 1.0, [E])[0]

    assert np.allclose(H0, H1, atol=1e-12)


# ---------------------------------------------------------------------------
# Selection matrices.
# ---------------------------------------------------------------------------
def test_selection_matrix_gathers_outline_vertices() -> None:
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    tri = np.array([[5.0, 0.0], [7.0, 0.0], [5.0, 2.0]])
    X = np.vstack([square, tri])
    E1 = closed_polyline(tri) + 4

    S = outline_selection_matrices([E1], X.shape[0])[0]

    assert S.shape == (3, X.shape[0])
    # Gathers exactly the triangle's vertices in np.unique order (4, 5, 6).
    assert np.allclose(S @ X, X[4:7])
