"""Tests for ``simkit.voronoi_labels``."""

from __future__ import annotations

import numpy as np

from simkit.compliance_distances import compliance_distances
from simkit.compliance_graph import compliance_graph
from simkit.compliance_node_sampling import compliance_node_sampling
from simkit.voronoi_labels import voronoi_labels


def test_shapes_ranges_and_node_ownership(stiff_bar_strip) -> None:
    X, T, ym = stiff_bar_strip
    G = compliance_graph(X, T, ym)
    nodes = compliance_node_sampling(X, G, 6)
    D = compliance_distances(G, nodes)
    vl, fl = voronoi_labels(D, T)
    assert vl.shape == (X.shape[0],) and fl.shape == (T.shape[0],)
    assert vl.min() >= 0 and vl.max() < 6
    assert fl.min() >= 0 and fl.max() < 6
    # every node vertex belongs to its own region
    np.testing.assert_array_equal(vl[nodes], np.arange(6))


def test_face_label_minimises_summed_corner_distance() -> None:
    # 2 nodes, 4 vertices, 2 triangles; hand-built distance table
    D = np.array(
        [[0.0, 1.0, 5.0, 5.0],  # node 0 close to vertices 0, 1
         [5.0, 5.0, 0.0, 1.0]],  # node 1 close to vertices 2, 3
    )
    T = np.array([[0, 1, 2], [1, 2, 3]])
    vl, fl = voronoi_labels(D, T)
    np.testing.assert_array_equal(vl, [0, 0, 1, 1])
    np.testing.assert_array_equal(fl, [0, 1])
