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


def test_tent_is_linear_between_two_nodes() -> None:
    # Two nodes on a uniform strip: the blend is linear in compliance
    # distance, 0.5 / 0.5 half-way and 0.75 / 0.25 a quarter of the way.
    X = np.c_[np.linspace(0, 1, 5), np.zeros(5)]
    X = np.vstack([X, X + [0.0, 1.0]])
    T = np.array([[i, i + 1, i + 6] for i in range(4)] + [[i, i + 6, i + 5] for i in range(4)])
    G = compliance_graph(X, T, np.ones(T.shape[0]))
    nodes = np.array([0, 4])
    D = compliance_distances(G, nodes)
    W = voronoi_shape_functions(D, nodes, support_scale=1.0).toarray()
    np.testing.assert_allclose(W[2], [0.5, 0.5], atol=1e-12)
    np.testing.assert_allclose(W[1], [0.75, 0.25], atol=1e-12)


def test_no_vertex_is_uncovered_and_nearest_node_dominates() -> None:
    # hand-built distances: vertex 2 is beyond both per-node radii (1.0). The
    # radius widens to d1 + d2 = 11 so it is still covered, continuously,
    # favouring the nearer node.
    D = np.array([[0.0, 1.0, 5.0], [1.0, 0.0, 6.0]])
    nodes = np.array([0, 1])
    W = voronoi_shape_functions(D, nodes, support_scale=1.0).toarray()
    np.testing.assert_allclose(W.sum(axis=1), 1.0)
    np.testing.assert_allclose(W[2], [6 / 11, 5 / 11])
    assert W[2, 0] >= 0.5


def test_weights_are_continuous_across_voronoi_frontier() -> None:
    # Three nodes on a uniform line at x = 0, 1 and 1.4 (unequal spacing);
    # sweep the vertices across the frontier between nodes 0 and 1.
    xs = np.linspace(0.0, 1.4, 281)
    node_x = np.array([0.0, 1.0, 1.4])
    nodes = np.array([0, 200, 280])
    D = np.abs(xs[None, :] - node_x[:, None])  # (3, n)
    W = voronoi_shape_functions(D, nodes, support_scale=1.0).toarray()
    np.testing.assert_allclose(W.sum(axis=1), 1.0)
    # bounded slope: the biggest step between neighbouring samples (0.005 apart)
    assert np.abs(np.diff(W, axis=0)).max() < 0.03
    np.testing.assert_allclose(W[nodes], np.eye(3), atol=1e-12)
    # equal weights on the frontier between nodes 0 and 1, although node 1's
    # own radius (0.4, to node 2) is shorter than the half-way distance; node 2
    # is just within reach there (0.9 < 1.0) and takes a small share
    np.testing.assert_allclose(W[100, 0], W[100, 1], atol=1e-12)
    assert 0.45 < W[100, 0] < 0.5 and W[100, 2] < 0.1
    # linear w_0 = 1 - x while node 2 is out of reach (x < 0.4)
    np.testing.assert_allclose(W[:80, 0], 1 - xs[:80], atol=1e-12)


def test_single_node_gets_all_the_weight() -> None:
    D = np.array([[0.0, 3.0, 7.0]])
    W = voronoi_shape_functions(D, np.array([0])).toarray()
    np.testing.assert_allclose(W, np.ones((3, 1)))
