"""Tests for ``simkit.linear_to_quadratic_elements``."""

from __future__ import annotations

import numpy as np
import pytest

from simkit.edges import edges
from simkit.linear_to_quadratic_elements import linear_to_quadratic_elements


def _two_element_mesh(dim: int):
    """A small mesh with two simplices sharing an interior face/edge."""
    if dim == 2:
        # Two triangles sharing edge (0, 2).
        X = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        T = np.array([[0, 1, 2], [0, 2, 3]])
    elif dim == 3:
        # Two tets sharing the triangular face (0, 1, 2).
        X = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )
        T = np.array([[0, 1, 2, 3], [0, 2, 1, 4]])
    else:
        raise ValueError(dim)
    return X, T


@pytest.mark.parametrize("dim", [2, 3])
def test_shapes_and_node_count(dim: int) -> None:
    X, T = _two_element_mesh(dim)
    V2, T2 = linear_to_quadratic_elements(X, T)

    n_nodes = 6 if dim == 2 else 10
    n_edges = edges(T).shape[0]

    # One new vertex per unique edge; element gains midpoint columns.
    assert V2.shape == (X.shape[0] + n_edges, dim)
    assert T2.shape == (T.shape[0], n_nodes)

    # Corner columns are copied verbatim.
    assert np.array_equal(T2[:, : dim + 1], T)


@pytest.mark.parametrize("dim", [2, 3])
def test_midpoints_lie_at_edge_midpoints(dim: int) -> None:
    X, T = _two_element_mesh(dim)
    V2, T2 = linear_to_quadratic_elements(X, T)

    s = dim + 1
    # Local edge ordering used by linear_to_quadratic_elements.
    if dim == 2:
        local = [(0, 1), (1, 2), (2, 0)]
    else:
        local = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    for e in range(T.shape[0]):
        for k, (a, b) in enumerate(local):
            mid_node = T2[e, s + k]
            expected = 0.5 * (X[T[e, a]] + X[T[e, b]])
            assert np.allclose(V2[mid_node], expected)


@pytest.mark.parametrize("dim", [2, 3])
def test_shared_edge_node_is_shared(dim: int) -> None:
    X, T = _two_element_mesh(dim)
    V2, T2 = linear_to_quadratic_elements(X, T)

    # The two elements share an edge (in 2D the (0,2) edge; in 3D both faces
    # contain the (0,1), (1,2) and (0,2) edges). Any shared edge must resolve to
    # the same midpoint node in both elements -> the unique node count matches
    # the unique edge count, and no midpoint vertex is duplicated.
    n_edges = edges(T).shape[0]
    midpoint_nodes = T2[:, dim + 1 :]
    assert np.unique(midpoint_nodes).size == n_edges

    # Concretely: the (0, 2) edge midpoint is shared between both elements.
    mid_02 = 0.5 * (X[0] + X[2])
    rows = np.where(np.all(np.isclose(V2, mid_02), axis=1))[0]
    assert rows.size == 1  # only one vertex sits there
