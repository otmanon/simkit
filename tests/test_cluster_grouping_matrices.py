"""Tests for ``simkit.cluster_grouping_matrices``."""

from __future__ import annotations

import numpy as np

from simkit.cluster_grouping_matrices import cluster_grouping_matrices


def _two_tet_mesh():
    X = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
    )
    T = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    return X, T


def test_cluster_grouping_sums_elements() -> None:
    X, T = _two_tet_mesh()
    labels = np.array([0, 1])
    G, Gm = cluster_grouping_matrices(labels, X, T)
    values = np.array([[1.0], [2.0]])
    summed = G @ values
    assert np.allclose(summed.flatten(), values.flatten())


def test_cluster_grouping_mass_average_preserves_constants() -> None:
    X, T = _two_tet_mesh()
    labels = np.array([0, 0])
    G, Gm = cluster_grouping_matrices(labels, X, T)
    constant = np.array([[3.5], [3.5]])
    averaged = Gm @ constant
    assert np.allclose(averaged, [[3.5]])


def test_cluster_grouping_return_mass() -> None:
    X, T = _two_tet_mesh()
    labels = np.array([0, 1])
    G, Gm, mc, mt, f = cluster_grouping_matrices(labels, X, T, return_mass=True)
    assert mc.shape == (2,)
    assert mt.shape == (2, 1)
    assert f.shape == (2,)
    assert np.allclose(f, mt[:, 0] / mc[labels])
