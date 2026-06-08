"""Tests for ``simkit.combine_meshes``."""

from __future__ import annotations

import numpy as np
import pytest

from simkit.combine_meshes import combine_meshes


def test_two_edge_meshes_offsets_indices() -> None:
    V0 = np.array([[0.0, 0.0], [1.0, 0.0]])
    V1 = np.array([[0.0, 1.0], [1.0, 1.0]])
    E0 = np.array([[0, 1]])
    E1 = np.array([[0, 1]])
    V, E = combine_meshes([V0, V1], [E0, E1])
    assert np.array_equal(V, np.vstack([V0, V1]))
    # Second mesh's edge indices are shifted by the first mesh's 2 vertices.
    assert np.array_equal(E, np.array([[0, 1], [2, 3]]))


def test_three_meshes_running_offsets() -> None:
    # Meshes with differing vertex/edge counts -> offsets are a running total.
    V_list = [
        np.zeros((3, 2)),
        np.zeros((2, 2)),
        np.zeros((4, 2)),
    ]
    E_list = [
        np.array([[0, 1], [1, 2]]),  # 2 edges
        np.array([[0, 1]]),          # 1 edge
        np.array([[0, 1], [2, 3]]),  # 2 edges
    ]
    V, E = combine_meshes(V_list, E_list)
    assert V.shape == (9, 2)
    expected = np.array(
        [
            [0, 1], [1, 2],          # offset 0
            [3, 4],                  # offset 3
            [5, 6], [7, 8],          # offset 5
        ]
    )
    assert np.array_equal(E, expected)


def test_3d_triangle_meshes() -> None:
    # Same offsetting logic must hold for 3D vertices and triangle faces.
    V0 = np.random.default_rng(0).random((3, 3))
    V1 = np.random.default_rng(1).random((3, 3))
    F0 = np.array([[0, 1, 2]])
    F1 = np.array([[0, 1, 2]])
    V, F = combine_meshes([V0, V1], [F0, F1])
    assert V.shape == (6, 3)
    assert np.array_equal(F, np.array([[0, 1, 2], [3, 4, 5]]))


def test_single_mesh_is_unchanged() -> None:
    V0 = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    F0 = np.array([[0, 1, 2]])
    V, F = combine_meshes([V0], [F0])
    assert np.array_equal(V, V0)
    assert np.array_equal(F, F0)


def test_combined_connectivity_indexes_into_combined_vertices() -> None:
    # The whole point: combined faces should address the combined vertex array
    # and recover the same per-mesh corner positions.
    V0 = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    V1 = np.array([[5.0, 5.0], [7.0, 5.0], [5.0, 7.0]])
    F0 = np.array([[0, 1, 2]])
    F1 = np.array([[0, 2, 1]])
    V, F = combine_meshes([V0, V1], [F0, F1])
    assert np.allclose(V[F[0]], V0[F0[0]])
    assert np.allclose(V[F[1]], V1[F1[0]])


def test_mismatched_list_lengths_raise() -> None:
    with pytest.raises(ValueError, match="vertex arrays but"):
        combine_meshes([np.zeros((2, 2))], [np.zeros((1, 2), int), np.zeros((1, 2), int)])


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="at least one mesh"):
        combine_meshes([], [])
