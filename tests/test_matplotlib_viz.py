"""Tests for ``simkit.matplotlib``.

The drawing helpers had no test file at all, so a rename in the numerics layer
could break every plotting demo silently. These run on the headless Agg
backend and check the lifecycle each class shares: construct, update positions,
remove.
"""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")  # pip install 'simkit[viz]'
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

from simkit.matplotlib import (  # noqa: E402
    Curve,
    Frame,
    PointCloud,
    TriangleMesh,
    VectorField,
)


@pytest.fixture(autouse=True)
def _clean_figure():
    """Each test draws into its own figure and closes it."""
    fig = plt.figure()
    plt.gca().set_xlim(-2, 2)
    plt.gca().set_ylim(-2, 2)
    yield
    plt.close(fig)


X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
T = np.array([[0, 1, 2], [1, 3, 2]])
E = np.array([[0, 1], [1, 3], [3, 2], [2, 0]])


# --------------------------------------------------------------------------- #
# Exports
# --------------------------------------------------------------------------- #
def test_all_drawing_classes_are_exported():
    import simkit.matplotlib as m

    for name in ("TriangleMesh", "PointCloud", "Curve", "Frame", "VectorField"):
        assert isinstance(getattr(m, name), type), name


def test_color_constants_are_exported():
    import simkit.matplotlib as m

    # colors.py is star-imported; at least the ones the classes default to.
    for name in ("light_blue", "yellow", "black"):
        assert hasattr(m, name), name


# --------------------------------------------------------------------------- #
# Shared lifecycle: construct -> update -> remove
# --------------------------------------------------------------------------- #
def test_triangle_mesh_lifecycle():
    mesh = TriangleMesh(X, T)
    assert mesh.pc in plt.gca().collections

    moved = X + np.array([0.5, -0.25])
    mesh.update_vertex_positions(moved)
    assert np.allclose(mesh.X, moved)

    mesh.remove()
    assert mesh.pc not in plt.gca().collections


def test_curve_lifecycle():
    curve = Curve(X, E)
    assert curve.lc in plt.gca().collections

    curve.update_vertex_positions(2.0 * X)
    assert np.allclose(curve.X, 2.0 * X)

    curve.remove()
    # Curve.remove() deletes its handle outright, unlike TriangleMesh which
    # keeps `self.pc`. Pinning the behaviour as it stands.
    assert not hasattr(curve, "lc")


def test_point_cloud_lifecycle():
    pc = PointCloud(X)
    assert np.allclose(pc.sc.get_offsets(), X)

    moved = X + 1.0
    pc.update_vertex_positions(moved)
    assert np.allclose(pc.sc.get_offsets(), moved)

    pc.remove()


def test_vector_field_lifecycle():
    V = np.tile(np.array([0.0, 1.0]), (X.shape[0], 1))
    vf = VectorField(X, V)

    vf.update_vector_field(X + 0.1, 2.0 * V)
    assert np.allclose(vf.V, 2.0 * V)

    vf.remove()


def test_frame_lifecycle():
    A = np.hstack([np.eye(2), np.array([[0.5], [0.5]])])
    frame = Frame(A)

    rotated = np.hstack([
        np.array([[0.0, -1.0], [1.0, 0.0]]),
        np.array([[1.0], [0.0]]),
    ])
    frame.update_frame(rotated)
    assert np.allclose(frame.A, rotated)

    frame.remove()


# --------------------------------------------------------------------------- #
# Behaviour
# --------------------------------------------------------------------------- #
def test_triangle_mesh_draws_one_polygon_per_triangle():
    mesh = TriangleMesh(X, T)
    assert len(mesh.pc.get_paths()) == T.shape[0]
    mesh.remove()


def test_curve_draws_one_segment_per_edge():
    curve = Curve(X, E)
    assert len(curve.lc.get_segments()) == E.shape[0]
    curve.remove()


def test_update_does_not_mutate_the_caller_array():
    mesh = TriangleMesh(X, T)
    original = X.copy()
    mesh.update_vertex_positions(X + 1.0)
    assert np.array_equal(X, original)
    mesh.remove()
