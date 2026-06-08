"""Tests for ``simkit.winding_number`` in 2D and 3D."""

from __future__ import annotations

import numpy as np
import pytest

from simkit.combine_meshes import combine_meshes
from simkit.winding_number import winding_number


# ---------------------------------------------------------------------------
# Mesh builders.
# ---------------------------------------------------------------------------
def _ccw_square() -> tuple[np.ndarray, np.ndarray]:
    """Unit square wound counter-clockwise."""
    V = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    E = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    return V, E


def _regular_polygon(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Regular ``n``-gon on the unit circle, wound counter-clockwise."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    V = np.stack([np.cos(t), np.sin(t)], axis=1)
    E = np.stack([np.arange(n), (np.arange(n) + 1) % n], axis=1)
    return V, E


def _outward_cube() -> tuple[np.ndarray, np.ndarray]:
    """Unit cube triangulated with consistently outward-facing normals."""
    V = np.array(
        [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ],
        dtype=float,
    )
    F = np.array(
        [
            [0, 3, 2], [0, 2, 1],  # z = 0  (normal -z)
            [4, 5, 6], [4, 6, 7],  # z = 1  (normal +z)
            [0, 1, 5], [0, 5, 4],  # y = 0  (normal -y)
            [2, 3, 7], [2, 7, 6],  # y = 1  (normal +y)
            [1, 2, 6], [1, 6, 5],  # x = 1  (normal +x)
            [0, 4, 7], [0, 7, 3],  # x = 0  (normal -x)
        ]
    )
    return V, F


# ---------------------------------------------------------------------------
# 2D.
# ---------------------------------------------------------------------------
def test_2d_inside_is_one_outside_is_zero() -> None:
    V, E = _ccw_square()
    Q = np.array([[0.5, 0.5], [0.25, 0.75], [2.0, 2.0], [-1.0, 0.5]])
    w = winding_number(Q, V, E)
    assert np.allclose(w, [1.0, 1.0, 0.0, 0.0], atol=1e-9)


def test_2d_orientation_flip_negates_interior() -> None:
    V, E = _ccw_square()
    w_ccw = winding_number(np.array([[0.5, 0.5]]), V, E)
    w_cw = winding_number(np.array([[0.5, 0.5]]), V, E[:, ::-1])
    assert np.allclose(w_ccw, 1.0)
    assert np.allclose(w_cw, -1.0)


def test_2d_polygon_interior_and_exterior() -> None:
    V, E = _regular_polygon(64)
    inside = np.array([[0.0, 0.0], [0.3, -0.2]])      # within unit circle
    outside = np.array([[2.0, 0.0], [0.0, -1.5]])     # beyond it
    assert np.allclose(winding_number(inside, V, E), 1.0, atol=1e-6)
    assert np.allclose(winding_number(outside, V, E), 0.0, atol=1e-6)


def test_2d_returns_one_value_per_query() -> None:
    V, E = _ccw_square()
    Q = np.random.default_rng(0).random((17, 2))
    assert winding_number(Q, V, E).shape == (17,)


def test_2d_combined_disjoint_loops_sum_independently() -> None:
    # Two disjoint CCW squares: each interior point is inside exactly one loop.
    V0, E0 = _ccw_square()
    V1, E1 = _ccw_square()
    V1 = V1 + np.array([5.0, 0.0])
    V, E = combine_meshes([V0, V1], [E0, E1])
    Q = np.array([[0.5, 0.5], [5.5, 0.5], [2.5, 0.5]])
    w = winding_number(Q, V, E)
    assert np.allclose(w, [1.0, 1.0, 0.0], atol=1e-9)


# ---------------------------------------------------------------------------
# 3D.
# ---------------------------------------------------------------------------
def test_3d_inside_is_one_outside_is_zero() -> None:
    V, F = _outward_cube()
    Q = np.array([[0.5, 0.5, 0.5], [0.1, 0.1, 0.1], [5.0, 5.0, 5.0], [-1.0, 0.5, 0.5]])
    w = winding_number(Q, V, F)
    assert np.allclose(w, [1.0, 1.0, 0.0, 0.0], atol=1e-9)


def test_3d_orientation_flip_negates_interior() -> None:
    V, F = _outward_cube()
    center = np.array([[0.5, 0.5, 0.5]])
    assert np.allclose(winding_number(center, V, F), 1.0)
    assert np.allclose(winding_number(center, V, F[:, ::-1]), -1.0)


def test_3d_returns_one_value_per_query() -> None:
    V, F = _outward_cube()
    Q = np.random.default_rng(1).random((23, 3))
    assert winding_number(Q, V, F).shape == (23,)


def test_3d_combined_disjoint_cubes_sum_independently() -> None:
    V0, F0 = _outward_cube()
    V1, F1 = _outward_cube()
    V1 = V1 + np.array([10.0, 0.0, 0.0])
    V, F = combine_meshes([V0, V1], [F0, F1])
    Q = np.array([[0.5, 0.5, 0.5], [10.5, 0.5, 0.5], [5.0, 0.5, 0.5]])
    w = winding_number(Q, V, F)
    assert np.allclose(w, [1.0, 1.0, 0.0], atol=1e-9)


# ---------------------------------------------------------------------------
# Input validation.
# ---------------------------------------------------------------------------
def test_bad_vertex_dimension_raises() -> None:
    with pytest.raises(ValueError, match="2D or 3D"):
        winding_number(np.zeros((1, 4)), np.zeros((3, 4)), np.array([[0, 1, 2]]))


def test_2d_wrong_connectivity_width_raises() -> None:
    V = np.zeros((3, 2))
    with pytest.raises(ValueError, match="edges with 2 columns"):
        winding_number(np.zeros((1, 2)), V, np.array([[0, 1, 2]]))


def test_3d_wrong_connectivity_width_raises() -> None:
    V = np.zeros((3, 3))
    with pytest.raises(ValueError, match="triangles with 3 columns"):
        winding_number(np.zeros((1, 3)), V, np.array([[0, 1]]))
