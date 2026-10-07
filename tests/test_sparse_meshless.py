"""Tests for ``simkit.sparse_meshless``."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

from simkit.lbs_jacobian import lbs_jacobian
from simkit.sparse_meshless import (
    compliance_distances,
    compliance_graph,
    element_compliance,
    lbs_affine_coordinates,
    sample_nodes,
    sparse_lbs_jacobian,
    sparse_meshless_methods_basis,
    voronoi_labels,
    voronoi_shape_functions,
)


def _grid_mesh(nx: int = 21, ny: int = 7, length: float = 3.0):
    """Regular triangulated strip ``[0, length] x [0, 1]``."""
    xs, ys = np.meshgrid(np.linspace(0, length, nx), np.linspace(0, 1, ny), indexing="ij")
    X = np.c_[xs.ravel(), ys.ravel()]
    idx = np.arange(nx * ny).reshape(nx, ny)
    T = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
            T += [[a, b, c], [a, c, d]]
    return X, np.array(T)


def _stiff_bar_ym(X, T, lo=1.0, hi=2.0, stiff=1e4, soft=1.0):
    """Soft strip with a stiff bar in the middle third."""
    cen = X[T].mean(axis=1)
    return np.where((cen[:, 0] > lo) & (cen[:, 0] < hi), stiff, soft)


def test_element_compliance_is_inverse_youngs_modulus() -> None:
    ym = np.array([[1.0], [4.0], [0.5]])
    np.testing.assert_allclose(element_compliance(ym), [1.0, 0.25, 2.0])


def test_compliance_graph_symmetric_and_weighted_by_length_and_compliance() -> None:
    # single triangle, uniform material
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    T = np.array([[0, 1, 2]])
    G = compliance_graph(X, T, np.array([2.0]))
    assert sps.issparse(G) and G.shape == (3, 3)
    Gd = G.toarray()
    np.testing.assert_allclose(Gd, Gd.T)
    np.testing.assert_allclose(Gd[0, 1], 1.0 * 0.5)
    np.testing.assert_allclose(Gd[0, 2], 2.0 * 0.5)
    np.testing.assert_allclose(Gd[1, 2], np.sqrt(5.0) * 0.5)


def test_compliance_graph_edge_reduce_modes() -> None:
    # two triangles share edge (1, 2); compliances 1 and 1/3
    X = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    T = np.array([[0, 1, 2], [1, 3, 2]])
    ym = np.array([1.0, 3.0])
    shared = (1, 2)
    assert compliance_graph(X, T, ym, reduce="mean")[shared] == pytest.approx((1 + 1 / 3) / 2)
    assert compliance_graph(X, T, ym, reduce="min")[shared] == pytest.approx(1 / 3)
    assert compliance_graph(X, T, ym, reduce="max")[shared] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        compliance_graph(X, T, ym, reduce="median")


def test_compliance_distances_stiff_material_is_cheap_to_cross() -> None:
    X, T = _grid_mesh()
    ym = _stiff_bar_ym(X, T)
    G = compliance_graph(X, T, ym)
    # source at the left end; distances measured along the bottom row
    src = int(np.argmin(np.linalg.norm(X - [0.0, 0.0], axis=1)))
    D = compliance_distances(G, src)
    assert D.shape == (1, X.shape[0])
    before = int(np.argmin(np.linalg.norm(X - [1.0, 0.0], axis=1)))
    after = int(np.argmin(np.linalg.norm(X - [2.0, 0.0], axis=1)))
    # crossing the stiff bar (length 1) costs ~1e-4, crossing soft (length 1) ~1
    assert D[0, after] - D[0, before] < 1e-2
    # soft material: compliance distance along the bottom row equals arclength
    assert D[0, before] == pytest.approx(X[before, 0], rel=1e-6)


def test_sample_nodes_count_uniqueness_and_seed_handling() -> None:
    X, T = _grid_mesh()
    G = compliance_graph(X, T, _stiff_bar_ym(X, T))
    nodes = sample_nodes(X, G, 8, lloyd_iters=4)
    assert nodes.shape == (8,)
    assert len(set(nodes.tolist())) == 8

    seeded = sample_nodes(X, G, 5, seed_nodes=[0, 3], lloyd_iters=4)
    assert seeded.shape == (5,)
    assert seeded[0] == 0 and seeded[1] == 3  # fixed seeds stay put

    # explicit first seed is honoured when no seed_nodes are given
    assert sample_nodes(X, G, 3, seed_index=7, lloyd_iters=0)[0] == 7


def test_sample_nodes_adapts_to_material() -> None:
    # The stiff bar is "small" in the compliance metric, so farthest-point
    # sampling should put most nodes in the soft ends.
    X, T = _grid_mesh()
    G = compliance_graph(X, T, _stiff_bar_ym(X, T))
    nodes = sample_nodes(X, G, 8)
    in_bar = (X[nodes, 0] > 1.0) & (X[nodes, 0] < 2.0)
    assert in_bar.sum() < (~in_bar).sum()


def test_voronoi_labels_shapes_and_ranges() -> None:
    X, T = _grid_mesh()
    G = compliance_graph(X, T, _stiff_bar_ym(X, T))
    nodes = sample_nodes(X, G, 6)
    D = compliance_distances(G, nodes)
    vl, fl = voronoi_labels(D, T)
    assert vl.shape == (X.shape[0],) and fl.shape == (T.shape[0],)
    assert vl.min() >= 0 and vl.max() < 6
    assert fl.min() >= 0 and fl.max() < 6
    # every node vertex belongs to its own region
    np.testing.assert_array_equal(vl[nodes], np.arange(6))


def test_voronoi_shape_functions_partition_of_unity_and_interpolation() -> None:
    X, T = _grid_mesh()
    G = compliance_graph(X, T, _stiff_bar_ym(X, T))
    nodes = sample_nodes(X, G, 8)
    D = compliance_distances(G, nodes)
    W = voronoi_shape_functions(D, nodes)
    assert sps.issparse(W) and W.shape == (X.shape[0], 8)
    Wd = W.toarray()
    assert Wd.min() >= 0.0
    np.testing.assert_allclose(Wd.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(Wd[nodes], np.eye(8), atol=1e-12)


def test_voronoi_shape_functions_tent_value_at_node_midpoint() -> None:
    # Two nodes on a uniform line: at the point half-way (in compliance
    # distance) between them, both tents are 0.5 and normalise to 0.5 / 0.5.
    X = np.c_[np.linspace(0, 1, 5), np.zeros(5)]
    X = np.vstack([X, X + [0.0, 1.0]])
    T = np.array([[i, i + 1, i + 6] for i in range(4)] + [[i, i + 6, i + 5] for i in range(4)])
    G = compliance_graph(X, T, np.ones(T.shape[0]))
    nodes = np.array([0, 4])
    D = compliance_distances(G, nodes)
    W = voronoi_shape_functions(D, nodes, support_scale=1.0).toarray()
    np.testing.assert_allclose(W[2], [0.5, 0.5], atol=1e-12)


def test_sparse_lbs_jacobian_matches_dense_affine_layout() -> None:
    rng = np.random.default_rng(0)
    for d in (2, 3):
        n, k = 9, 3
        X = rng.standard_normal((n, d))
        W = rng.random((n, k))
        W[rng.random((n, k)) < 0.4] = 0.0  # compact support
        W /= W.sum(axis=1, keepdims=True)
        B = sparse_lbs_jacobian(X, sps.csr_matrix(W), order=1)
        assert sps.issparse(B)
        assert B.shape == (n * d, k * d * (d + 1))
        np.testing.assert_allclose(B.toarray(), lbs_jacobian(X, W), atol=1e-14)


def test_sparse_lbs_jacobian_point_frames() -> None:
    rng = np.random.default_rng(1)
    n, d, k = 7, 2, 4
    X = rng.standard_normal((n, d))
    W = rng.random((n, k))
    W /= W.sum(axis=1, keepdims=True)
    B = sparse_lbs_jacobian(X, W, order=0)
    assert B.shape == (n * d, k * d)
    np.testing.assert_allclose(B.toarray(), np.kron(W, np.eye(d)), atol=1e-14)
    with pytest.raises(ValueError):
        sparse_lbs_jacobian(X, W, order=2)


def test_affine_patch_test_through_full_pipeline() -> None:
    X, T = _grid_mesh()
    ym = _stiff_bar_ym(X, T)
    W, B, labels, nodes = sparse_meshless_methods_basis(X, T, ym, n_nodes=8)
    k = len(nodes)
    assert W.shape == (X.shape[0], k)
    assert B.shape == (X.shape[0] * 2, k * 2 * 3)
    assert labels.shape == (T.shape[0],)

    A = np.array([[1.3, 0.2], [-0.1, 0.8]])
    t = np.array([0.5, -0.2])
    z = lbs_affine_coordinates(k, A, t)
    assert z.shape == (k * 6, 1)
    x = (B @ z).reshape(-1, 2)
    np.testing.assert_allclose(x, X @ A.T + t, atol=1e-12)

    # identity / zero translation reproduces the rest state
    z0 = lbs_affine_coordinates(k, np.eye(2), np.zeros(2))
    np.testing.assert_allclose((B @ z0).reshape(-1, 2), X, atol=1e-12)


def test_basis_return_distances_and_point_frames() -> None:
    X, T = _grid_mesh(nx=11, ny=4)
    ym = _stiff_bar_ym(X, T)
    out = sparse_meshless_methods_basis(
        X, T, ym, n_nodes=4, frame_order=0, return_distances=True
    )
    assert len(out) == 5
    W, B, labels, nodes, D = out
    assert D.shape == (4, X.shape[0])
    assert np.all(np.isfinite(D))
    np.testing.assert_allclose(D[np.arange(4), nodes], 0.0)
    assert B.shape == (X.shape[0] * 2, 4 * 2)


def test_basis_on_3d_surface_mesh() -> None:
    X2, T = _grid_mesh(nx=11, ny=4)
    X = np.c_[X2, np.sin(X2[:, 0])]
    ym = _stiff_bar_ym(X2, T)
    W, B, labels, nodes = sparse_meshless_methods_basis(X, T, ym, n_nodes=5)
    assert B.shape == (X.shape[0] * 3, 5 * 3 * 4)
    np.testing.assert_allclose(B.toarray(), lbs_jacobian(X, W.toarray()), atol=1e-14)


def test_basis_is_deterministic() -> None:
    X, T = _grid_mesh(nx=11, ny=4)
    ym = _stiff_bar_ym(X, T)
    a = sparse_meshless_methods_basis(X, T, ym, n_nodes=5)
    b = sparse_meshless_methods_basis(X, T, ym, n_nodes=5)
    np.testing.assert_array_equal(a[3], b[3])
    assert (a[0] != b[0]).nnz == 0
