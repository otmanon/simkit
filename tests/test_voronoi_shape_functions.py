"""Tests for ``simkit.voronoi_shape_functions``."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps

from simkit.compliance_distances import compliance_distances
from simkit.compliance_graph import compliance_graph
from simkit.compliance_node_sampling import compliance_node_sampling
from simkit.voronoi_shape_functions import voronoi_shape_functions


def test_partition_of_unity_nonnegative_and_interpolating(stiff_bar_strip) -> None:
    X, T, ym = stiff_bar_strip
    G = compliance_graph(X, T, ym)
    nodes = compliance_node_sampling(X, G, 8)
    D = compliance_distances(G, nodes)
    W = voronoi_shape_functions(D, nodes)
    assert sps.issparse(W) and W.shape == (X.shape[0], 8)
    Wd = W.toarray()
    assert Wd.min() >= 0.0
    np.testing.assert_allclose(Wd.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(Wd[nodes], np.eye(8), atol=1e-12)


def test_compact_support_and_widening(stiff_bar_strip) -> None:
    X, T, ym = stiff_bar_strip
    G = compliance_graph(X, T, ym)
    nodes = compliance_node_sampling(X, G, 8)
    D = compliance_distances(G, nodes)
    tight = voronoi_shape_functions(D, nodes, support_scale=1.0)
    wide = voronoi_shape_functions(D, nodes, support_scale=3.0)
    # tents are compact: most vertices see only a few nodes
    assert tight.nnz < 0.5 * np.prod(tight.shape)
    # widening the support can only add non-zeros
    assert wide.nnz >= tight.nnz


def test_tent_is_half_at_midpoint_between_two_nodes() -> None:
    # Two nodes on a uniform strip: at the vertex half-way (in compliance
    # distance) between them both tents are 0.5 and normalise to 0.5 / 0.5.
    X = np.c_[np.linspace(0, 1, 5), np.zeros(5)]
    X = np.vstack([X, X + [0.0, 1.0]])
    T = np.array([[i, i + 1, i + 6] for i in range(4)] + [[i, i + 6, i + 5] for i in range(4)])
    G = compliance_graph(X, T, np.ones(T.shape[0]))
    nodes = np.array([0, 4])
    D = compliance_distances(G, nodes)
    W = voronoi_shape_functions(D, nodes, support_scale=1.0).toarray()
    np.testing.assert_allclose(W[2], [0.5, 0.5], atol=1e-12)
    np.testing.assert_allclose(W[1], [0.75, 0.25], atol=1e-12)


def test_vertices_outside_all_supports_fall_back_to_owner() -> None:
    # hand-built distances: vertex 2 is far from both nodes (beyond R = 1.0)
    D = np.array([[0.0, 1.0, 5.0], [1.0, 0.0, 6.0]])
    nodes = np.array([0, 1])
    W = voronoi_shape_functions(D, nodes, support_scale=1.0).toarray()
    np.testing.assert_allclose(W[2], [1.0, 0.0])
    np.testing.assert_allclose(W.sum(axis=1), 1.0)
