"""Tests for ``boundary_edges`` and ``wedge_map``.

Both are small topology helpers that had no by-name test file, and both are
used by the bending/shell code, where a wrong orientation or a wrong gather
ordering shows up as a plausible-looking but incorrect energy.
"""

from __future__ import annotations

import numpy as np
import pytest

from simkit import boundary_edges, wedge_map


# --------------------------------------------------------------------------- #
# boundary_edges
# --------------------------------------------------------------------------- #
def test_single_triangle_is_all_boundary():
    F = np.array([[0, 1, 2]])
    E = boundary_edges(F)
    assert E.shape == (3, 2)
    assert {tuple(e) for e in E} == {(0, 1), (1, 2), (2, 0)}


def test_shared_edge_is_excluded():
    """Two triangles sharing edge (1, 2): four boundary edges, not six."""
    F = np.array([[0, 1, 2], [1, 3, 2]])
    E = boundary_edges(F)
    assert E.shape == (4, 2)
    undirected = {frozenset(e) for e in E}
    assert frozenset((1, 2)) not in undirected
    assert undirected == {frozenset((0, 1)), frozenset((2, 0)),
                          frozenset((1, 3)), frozenset((3, 2))}


def test_boundary_edges_keep_triangle_winding():
    """Orientation is preserved so the result can be drawn as an outline.

    For a consistently counter-clockwise mesh, walking the returned edges
    head-to-tail must form a closed loop.
    """
    F = np.array([[0, 1, 2], [1, 3, 2]])
    E = boundary_edges(F)

    successor = {int(a): int(b) for a, b in E}
    assert len(successor) == len(E), "each vertex starts exactly one edge"

    start = int(E[0, 0])
    seen, v = [], start
    for _ in range(len(E)):
        seen.append(v)
        v = successor[v]
    assert v == start, "boundary must close back on itself"
    assert len(set(seen)) == len(E)


def test_closed_mesh_has_no_boundary():
    """A tetrahedron's surface: every edge is shared by two triangles."""
    F = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    assert boundary_edges(F).shape[0] == 0


def test_grid_boundary_count():
    """A 3x3 vertex grid (4 quads, 8 triangles) has 8 boundary edges."""
    n = 3
    idx = np.arange(n * n).reshape(n, n)
    tris = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = idx[i, j], idx[i, j + 1], idx[i + 1, j], idx[i + 1, j + 1]
            tris += [[a, b, c], [b, d, c]]
    E = boundary_edges(np.array(tris))
    assert E.shape[0] == 4 * (n - 1)


def test_result_is_a_subset_of_the_input_directed_edges():
    F = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4]])
    directed = set()
    for f in F:
        directed |= {(f[0], f[1]), (f[1], f[2]), (f[2], f[0])}
    for e in boundary_edges(F):
        assert tuple(e) in directed


# --------------------------------------------------------------------------- #
# wedge_map
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [3, 4])
def test_shape_and_single_unit_entry_per_row(k):
    rng = np.random.default_rng(0)
    nv = 9
    C = rng.integers(0, nv, size=(5, k))
    M = wedge_map(C, nv)

    assert M.shape == (C.shape[0] * k, nv)
    rows = np.asarray(M.sum(axis=1)).ravel()
    assert np.all(rows == 1), "exactly one gather entry per stacked row"
    assert M.nnz == C.size


@pytest.mark.parametrize("k", [3, 4])
def test_gathers_hinge_vertices_in_stacked_order(k):
    """Row ``k*e + j`` must select vertex ``C[e, j]``."""
    rng = np.random.default_rng(1)
    nv = 7
    C = rng.integers(0, nv, size=(4, k))
    M = wedge_map(C, nv)

    x = rng.standard_normal((nv, 1))
    gathered = (M @ x).reshape(-1)
    expected = x[C.reshape(-1), 0]
    assert np.allclose(gathered, expected)


def test_gathering_vertex_positions_reproduces_the_hinge_stencil():
    C = np.array([[0, 1, 2], [2, 3, 0]])
    nv = 4
    V = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    M = wedge_map(C, nv)

    stacked = (M @ V).reshape(C.shape[0], C.shape[1], 2)
    for e in range(C.shape[0]):
        for j in range(C.shape[1]):
            assert np.allclose(stacked[e, j], V[C[e, j]])


def test_transpose_scatters_and_accumulates():
    """``M.T`` is the scatter/accumulate adjoint used to assemble forces."""
    C = np.array([[0, 1, 2], [2, 1, 0]])
    nv = 3
    M = wedge_map(C, nv)

    ones = np.ones((C.size, 1))
    counts = np.asarray((M.T @ ones)).ravel()
    expected = np.bincount(C.reshape(-1), minlength=nv).astype(float)
    assert np.allclose(counts, expected)


def test_unreferenced_vertices_give_empty_columns():
    C = np.array([[0, 1, 2]])
    M = wedge_map(C, 5)
    col_sums = np.asarray(M.sum(axis=0)).ravel()
    assert np.allclose(col_sums[3:], 0.0)
