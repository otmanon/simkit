"""Tests for ``simkit.compliance_node_sampling``."""

from __future__ import annotations

import numpy as np

from simkit.compliance_graph import compliance_graph
from simkit.compliance_node_sampling import compliance_node_sampling


def test_count_and_uniqueness(stiff_bar_strip) -> None:
    X, T, ym = stiff_bar_strip
    G = compliance_graph(X, T, ym)
    nodes = compliance_node_sampling(X, G, 8, lloyd_iters=4)
    assert nodes.shape == (8,)
    assert nodes.dtype.kind == "i"
    assert len(set(nodes.tolist())) == 8


def test_seed_nodes_are_kept_first_and_fixed(stiff_bar_strip) -> None:
    X, T, ym = stiff_bar_strip
    G = compliance_graph(X, T, ym)
    nodes = compliance_node_sampling(X, G, 5, seed_nodes=[0, 3], lloyd_iters=4)
    assert nodes.shape == (5,)
    assert nodes[0] == 0 and nodes[1] == 3
    # duplicate seeds collapse to one
    nodes = compliance_node_sampling(X, G, 4, seed_nodes=[7, 7], lloyd_iters=0)
    assert nodes[0] == 7 and len(set(nodes.tolist())) == 4


def test_seed_index_is_honoured_without_relaxation(stiff_bar_strip) -> None:
    X, T, ym = stiff_bar_strip
    G = compliance_graph(X, T, ym)
    assert compliance_node_sampling(X, G, 3, seed_index=7, lloyd_iters=0)[0] == 7
    # default seed: vertex with the smallest first coordinate
    assert compliance_node_sampling(X, G, 1, lloyd_iters=0)[0] == int(np.argmin(X[:, 0]))


def test_adapts_to_material(stiff_bar_strip) -> None:
    # The stiff bar is "small" in the compliance metric, so farthest-point
    # sampling should put most nodes in the soft ends.
    X, T, ym = stiff_bar_strip
    G = compliance_graph(X, T, ym)
    nodes = compliance_node_sampling(X, G, 8)
    in_bar = (X[nodes, 0] > 1.0) & (X[nodes, 0] < 2.0)
    assert in_bar.sum() < (~in_bar).sum()


def test_deterministic(stiff_bar_strip) -> None:
    X, T, ym = stiff_bar_strip
    G = compliance_graph(X, T, ym)
    a = compliance_node_sampling(X, G, 6)
    b = compliance_node_sampling(X, G, 6)
    np.testing.assert_array_equal(a, b)
