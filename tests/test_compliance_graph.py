"""Tests for ``simkit.compliance_graph``."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

from simkit.compliance_graph import compliance_graph


def test_single_triangle_weights_are_length_times_compliance() -> None:
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    T = np.array([[0, 1, 2]])
    G = compliance_graph(X, T, np.array([2.0]))  # compliance 0.5
    assert sps.issparse(G) and G.shape == (3, 3)
    Gd = G.toarray()
    np.testing.assert_allclose(Gd, Gd.T)
    np.testing.assert_allclose(Gd[0, 1], 1.0 * 0.5)
    np.testing.assert_allclose(Gd[0, 2], 2.0 * 0.5)
    np.testing.assert_allclose(Gd[1, 2], np.sqrt(5.0) * 0.5)
    assert np.all(np.diag(Gd) == 0)


def test_accepts_column_vector_youngs_modulus() -> None:
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    T = np.array([[0, 1, 2]])
    a = compliance_graph(X, T, np.array([4.0]))
    b = compliance_graph(X, T, np.array([[4.0]]))
    assert (a != b).nnz == 0
    assert a[0, 1] == pytest.approx(0.25)


def test_edge_reduce_modes_on_shared_edge() -> None:
    # two triangles share edge (1, 2); compliances 1 and 1/3
    X = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    T = np.array([[0, 1, 2], [1, 3, 2]])
    ym = np.array([1.0, 3.0])
    shared = (1, 2)
    assert compliance_graph(X, T, ym, reduce="mean")[shared] == pytest.approx((1 + 1 / 3) / 2)
    assert compliance_graph(X, T, ym, reduce="min")[shared] == pytest.approx(1 / 3)
    assert compliance_graph(X, T, ym, reduce="max")[shared] == pytest.approx(1.0)
    # an unshared edge is unaffected by the reduce mode
    for mode in ("mean", "min", "max"):
        assert compliance_graph(X, T, ym, reduce=mode)[0, 1] == pytest.approx(1.0)


def test_unknown_reduce_raises() -> None:
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    T = np.array([[0, 1, 2]])
    with pytest.raises(ValueError):
        compliance_graph(X, T, np.array([1.0]), reduce="median")


def test_stiff_elements_give_small_weights(stiff_bar_strip) -> None:
    X, T, ym = stiff_bar_strip
    G = compliance_graph(X, T, ym).tocoo()
    mid = 0.5 * (X[G.row] + X[G.col])
    in_bar = (mid[:, 0] > 1.05) & (mid[:, 0] < 1.95)
    assert G.data[in_bar].max() < 1e-3 * G.data[~in_bar].min()
