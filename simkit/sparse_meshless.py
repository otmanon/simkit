"""Sparse meshless models of heterogeneous deformable solids (Faure et al. 2011).

Implementation of the SIGGRAPH 2011 paper *"Sparse Meshless Models of Complex
Deformable Solids"* (Faure, Gilles, Bousquet, Pai), adapted from the paper's
voxel-grid setting to a **triangle mesh** carrying a per-element Young's modulus.

The paper's key idea: instead of geometry-only kernels (RBFs, barycentric
coordinates), build *material-aware* shape functions so that a **sparse** set of
control frames can resolve a heterogeneous stiffness map. Two points connected by
**stiff** material move together (rigidly); points connected by **compliant**
material can move very differently. This is captured by a **compliance distance**
-- a shortest-path metric where each step costs ``compliance x length`` with
``compliance = 1 / E``. Stiff regions shrink to near-points in this metric (a
rigid body collapses to a single point), soft regions stretch out.

Everything downstream flows from that one metric:

* :func:`compliance_graph`        -- the weighted mesh-edge graph (``c * len``).
* :func:`compliance_distances`    -- Dijkstra shortest paths on that graph.
* :func:`sample_nodes`            -- farthest-point sampling + Lloyd relaxation in
  the compliance metric (Section 4.4). Naturally seeds *more* nodes in soft
  regions, *fewer* in stiff ones -- the adaptivity the method buys.
* :func:`voronoi_labels`          -- the Voronoi partition (which region / node
  each vertex and each triangle belongs to).
* :func:`voronoi_shape_functions` -- the material-aware, partition-of-unity,
  interpolating, compact-support **skinning weights** ``W`` (Section 4.3), built
  as clamped-linear "tents" in the compliance metric so they reproduce the
  paper's ideal linear-in-compliance-distance shape functions (Fig. 5e).
* :func:`sparse_lbs_jacobian`     -- assembles ``W`` into a **sparse** linear
  blend skinning subspace ``B`` (affine or point frames, Section 3), kept sparse
  so a reduced simulation can exploit the block structure.

Primary entry point: :func:`sparse_meshless_methods_basis`, which returns a
:class:`SparseMeshlessModel` bundling the weights ``W``, the per-cell Voronoi
labels, the sparse subspace ``B`` and everything a subspace simulation needs
(node placement, full-integration cubature weights, per-element compliance).

Notes on the adaptation
------------------------
The original paper computes an approximate compliance-distance field on a
**voxel grid** with Dijkstra, and smooths kernels by recursively subdividing
Voronoi iso-surfaces. Here the "voxels" are the mesh vertices and the metric
lives on mesh edges; the recursive iso-surface smoothing is replaced by the
equivalent closed-form clamped-linear tent (value ``1`` at the node, ``0.5`` at
the Voronoi frontier, ``0`` at the neighbouring node), which is exactly what the
iso-surface interpolation converges to and keeps material interfaces sharp.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.csgraph as csgraph


# =========================================================================== #
# Compliance metric                                                            #
# =========================================================================== #
def element_compliance(ym: np.ndarray) -> np.ndarray:
    """Per-element compliance ``c = 1 / E`` from Young's modulus.

    Parameters
    ----------
    ym : np.ndarray (t,) or (t, 1)
        Per-element Young's modulus (heterogeneous material).

    Returns
    -------
    c : np.ndarray (t,)
        Per-element compliance. Stiff elements have small compliance.
    """
    ym = np.asarray(ym, dtype=float).reshape(-1)
    return 1.0 / ym


def compliance_graph(
    X: np.ndarray,
    T: np.ndarray,
    ym: np.ndarray,
    reduce: str = "mean",
) -> sp.sparse.csr_matrix:
    """Weighted mesh-edge graph for the compliance-distance metric.

    Each undirected mesh edge ``(i, j)`` is given weight
    ``||X_i - X_j|| * c_ij`` where ``c_ij`` is the compliance of the material the
    edge passes through, aggregated from the edge's incident triangles. Crossing
    stiff material is therefore *cheap* and crossing soft material is *expensive*,
    exactly the metric the paper's compliance distance induces.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Vertex positions.
    T : np.ndarray (t, 3)
        Triangle connectivity.
    ym : np.ndarray (t,) or (t, 1)
        Per-element Young's modulus.
    reduce : {"mean", "min", "max"}, optional
        How to combine the compliances of the triangles incident to a shared
        edge. ``"mean"`` (default) averages them (a symmetric, interface-sharp
        choice); ``"min"`` follows the paper's "stiffest path" reading literally;
        ``"max"`` is the most conservative (softest incident material wins).

    Returns
    -------
    G : scipy.sparse.csr_matrix (n, n)
        Symmetric non-negative edge-weight matrix suitable for
        :func:`scipy.sparse.csgraph.dijkstra`.
    """
    X = np.asarray(X, dtype=float)
    T = np.asarray(T)
    n = X.shape[0]
    c = element_compliance(ym)

    # All three directed edges of every triangle, canonicalised to (min, max).
    E = np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    E = np.sort(E, axis=1)
    c_rep = np.tile(c, 3)  # compliance of the triangle each edge instance came from

    Eu, inv = np.unique(E, axis=0, return_inverse=True)
    inv = inv.reshape(-1)

    if reduce == "mean":
        sums = np.bincount(inv, weights=c_rep, minlength=Eu.shape[0])
        cnts = np.bincount(inv, minlength=Eu.shape[0])
        c_edge = sums / np.maximum(cnts, 1)
    elif reduce == "min":
        c_edge = np.full(Eu.shape[0], np.inf)
        np.minimum.at(c_edge, inv, c_rep)
    elif reduce == "max":
        c_edge = np.zeros(Eu.shape[0])
        np.maximum.at(c_edge, inv, c_rep)
    else:
        raise ValueError(f"unknown reduce={reduce!r}")

    lengths = np.linalg.norm(X[Eu[:, 0]] - X[Eu[:, 1]], axis=1)
    w = lengths * c_edge

    G = sp.sparse.coo_matrix((w, (Eu[:, 0], Eu[:, 1])), shape=(n, n))
    G = G + G.T  # symmetric / undirected
    return G.tocsr()


def compliance_distances(
    G: sp.sparse.spmatrix, sources: Sequence[int]
) -> np.ndarray:
    """Shortest-path (compliance) distances from ``sources`` to every vertex.

    Parameters
    ----------
    G : scipy.sparse matrix (n, n)
        Compliance-edge graph from :func:`compliance_graph`.
    sources : sequence of int
        Source vertex indices.

    Returns
    -------
    D : np.ndarray (len(sources), n)
        ``D[a, v]`` is the compliance distance from ``sources[a]`` to vertex
        ``v`` (``inf`` if unreachable).
    """
    sources = np.atleast_1d(np.asarray(sources, dtype=int))
    D = csgraph.dijkstra(G, directed=False, indices=sources)
    return np.atleast_2d(D)


# =========================================================================== #
# Node distribution (Section 4.4): farthest-point sampling + Lloyd relaxation  #
# =========================================================================== #
def sample_nodes(
    X: np.ndarray,
    G: sp.sparse.spmatrix,
    n_nodes: int,
    seed_index: Optional[int] = None,
    seed_nodes: Optional[Sequence[int]] = None,
    lloyd_iters: int = 8,
) -> np.ndarray:
    """Distribute control nodes uniformly in the compliance metric.

    Farthest-point sampling in compliance distance (so soft regions, which are
    "large" in the metric, receive more nodes) followed by Lloyd relaxation
    (each node re-centred within its Voronoi region). Pre-seeded nodes -- e.g. in
    rigid parts, per the paper's suggestion to avoid linear-blend-skinning
    artifacts -- are kept fixed during relaxation.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Vertex positions (used only for the Euclidean re-centring in Lloyd).
    G : scipy.sparse matrix (n, n)
        Compliance-edge graph.
    n_nodes : int
        Total number of control nodes to place (including any seeds).
    seed_index : int, optional
        First farthest-point seed. Defaults to the vertex with the smallest
        first coordinate (deterministic).
    seed_nodes : sequence of int, optional
        Vertex indices to pre-place and hold fixed during Lloyd relaxation.
    lloyd_iters : int, optional
        Number of Lloyd relaxation passes.

    Returns
    -------
    nodes : np.ndarray (n_nodes,)
        Vertex indices of the control nodes.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    fixed = list(dict.fromkeys(int(s) for s in (seed_nodes or [])))
    nodes = list(fixed)

    if not nodes:
        s0 = int(np.argmin(X[:, 0])) if seed_index is None else int(seed_index)
        nodes.append(s0)

    # --- farthest-point sampling in the compliance metric ------------------
    dmin = np.full(n, np.inf)
    for s in nodes:
        dmin = np.minimum(dmin, compliance_distances(G, s)[0])
    while len(nodes) < n_nodes:
        finite = np.where(np.isfinite(dmin), dmin, -np.inf)
        idx = int(np.argmax(finite))
        if idx in nodes:  # degenerate (disconnected) mesh -- stop early
            break
        nodes.append(idx)
        dmin = np.minimum(dmin, compliance_distances(G, idx)[0])

    nodes = np.array(nodes, dtype=int)
    fixed_set = set(fixed)

    # --- Lloyd relaxation: re-centre each free node in its region ----------
    for _ in range(max(0, lloyd_iters)):
        D = compliance_distances(G, nodes)          # (k, n)
        labels = np.argmin(D, axis=0)
        moved = False
        for a in range(len(nodes)):
            if int(nodes[a]) in fixed_set:
                continue
            members = np.where(labels == a)[0]
            if members.size == 0:
                continue
            centroid = X[members].mean(axis=0)
            # snap the region's Euclidean centroid to its nearest member vertex
            j = members[int(np.argmin(np.linalg.norm(X[members] - centroid, axis=1)))]
            if j != nodes[a]:
                nodes[a] = j
                moved = True
        if not moved:
            break

    return nodes


# =========================================================================== #
# Voronoi partition + material-aware shape functions (Section 4.3)             #
# =========================================================================== #
def voronoi_labels(D_nodes: np.ndarray, T: np.ndarray):
    """Voronoi region label for every vertex and every triangle.

    Parameters
    ----------
    D_nodes : np.ndarray (k, n)
        Compliance distances from each of the ``k`` nodes to every vertex.
    T : np.ndarray (t, 3)
        Triangle connectivity.

    Returns
    -------
    vertex_labels : np.ndarray (n,)
        Index of the nearest (in compliance distance) node for each vertex.
    face_labels : np.ndarray (t,)
        Voronoi region of each triangle (cell), the node minimising the summed
        compliance distance to the triangle's three corners.
    """
    vertex_labels = np.argmin(D_nodes, axis=0)
    tri_cost = D_nodes[:, T].sum(axis=2)            # (k, t)
    face_labels = np.argmin(tri_cost, axis=0)
    return vertex_labels, face_labels


def voronoi_shape_functions(
    D_nodes: np.ndarray,
    nodes: np.ndarray,
    support_scale: float = 1.25,
) -> sp.sparse.csr_matrix:
    """Material-aware skinning weights ``W`` (the paper's Voronoi kernels).

    Each node ``a`` gets a clamped-linear "tent" in the compliance metric,

        ``w_a(v) = max(0, 1 - d_a(v) / R_a)`` ,

    with support radius ``R_a = support_scale * (distance to nearest other
    node)``. At ``support_scale = 1`` the tent is exactly ``1`` at the node,
    ``0.5`` at the Voronoi frontier (half-way to the nearest neighbour) and ``0``
    at that neighbour -- the paper's ideal, linear-in-compliance-distance shape
    function. The tents are then normalised to a partition of unity. Because the
    metric makes crossing soft material expensive, a node's support does not leak
    across a stiff bone into the flesh beyond it, giving the sharp interfaces the
    method is designed for.

    Interpolation (``w_a`` = 1 at node ``a``, 0 at every other node) is enforced
    exactly at the node vertices so the weights are suitable for Dirichlet
    handles.

    Parameters
    ----------
    D_nodes : np.ndarray (k, n)
        Compliance distances from each node to every vertex.
    nodes : np.ndarray (k,)
        Node vertex indices (rows/cols of ``D_nodes[:, nodes]`` are the
        node-to-node distances).
    support_scale : float, optional
        Multiplies each node's support radius. ``1.0`` gives compact,
        exactly-interpolating tents; ``>1`` widens supports for smoother overlap
        (the analogue of using more Voronoi sub-divisions in the paper).

    Returns
    -------
    W : scipy.sparse.csr_matrix (n, k)
        Non-negative skinning weights whose rows sum to 1.
    """
    D_nodes = np.asarray(D_nodes, dtype=float)
    k, n = D_nodes.shape
    nodes = np.asarray(nodes, dtype=int)

    # nearest-other-node compliance distance -> per-node support radius
    Dnn = D_nodes[:, nodes].copy()                  # (k, k)
    np.fill_diagonal(Dnn, np.inf)
    R = support_scale * np.min(Dnn, axis=1)         # (k,)
    R = np.where(np.isfinite(R) & (R > 0), R, 1.0)

    raw = np.maximum(0.0, 1.0 - D_nodes / R[:, None])   # (k, n)
    raw[~np.isfinite(D_nodes)] = 0.0
    W = raw.T                                            # (n, k)

    # Holes (a vertex beyond every node's support) fall back to their owner: a
    # rigid extension that keeps the partition of unity and the sharp interface.
    row_sum = W.sum(axis=1)
    empty = row_sum <= 1e-12
    if np.any(empty):
        owner = np.argmin(D_nodes, axis=0)
        W[empty, :] = 0.0
        W[empty, owner[empty]] = 1.0
        row_sum = W.sum(axis=1)

    W = W / row_sum[:, None]

    # Exact interpolation at the node vertices (clean Dirichlet handles).
    W[nodes, :] = 0.0
    W[nodes, np.arange(k)] = 1.0

    W[W < 1e-9] = 0.0
    return sp.sparse.csr_matrix(W)


# =========================================================================== #
# Sparse linear blend skinning subspace (Section 3)                            #
# =========================================================================== #
def sparse_lbs_jacobian(
    X: np.ndarray, W: sp.sparse.spmatrix, order: int = 1
) -> sp.sparse.csr_matrix:
    """Sparse LBS subspace ``B`` such that ``x = B z + x0`` (or ``u = B z``).

    Assembles the linear-blend-skinning Jacobian of the (compact-support) weights
    ``W`` into a **sparse** basis, preserving the block structure that makes a
    reduced simulation cheap. For affine frames the layout matches SimKit's dense
    :func:`simkit.lbs_jacobian` exactly (same DOF ordering), so it is a drop-in
    replacement that additionally keeps the matrix sparse.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Rest vertex positions.
    W : scipy.sparse matrix (n, k)
        Skinning weights over ``k`` nodes.
    order : {0, 1}, optional
        Frame type. ``1`` (default) = affine frames: each node carries a
        ``dim x (dim + 1)`` transform, ``dim * (dim + 1)`` DOFs, and can stretch,
        shear, bend and rotate its neighbourhood. ``0`` = point/translation
        frames: ``dim`` DOFs per node.

    Returns
    -------
    B : scipy.sparse.csr_matrix (n*dim, r)
        Reduced subspace. ``r = k * dim * (dim + 1)`` for affine frames,
        ``r = k * dim`` for point frames. The map is on *positions*; subtract the
        rest state to use it on displacements.
    """
    X = np.asarray(X, dtype=float)
    n, dim = X.shape
    Wc = sp.sparse.csr_matrix(W).tocoo()
    i, kk, w = Wc.row, Wc.col, Wc.data
    k = W.shape[1]

    if order == 0:
        # u_v = sum_k W[v,k] * t_k   ->   B = kron(W, I_dim)
        rows = (i[:, None] * dim + np.arange(dim)[None, :]).ravel()
        cols = (kk[:, None] * dim + np.arange(dim)[None, :]).ravel()
        data = np.repeat(w, dim)
        r = k * dim
        return sp.sparse.coo_matrix((data, (rows, cols)), shape=(n * dim, r)).tocsr()

    if order != 1:
        raise ValueError("only order 0 (point) and 1 (affine) frames are supported")

    # Affine frames: per node the polynomial basis is [x, y, (z,) 1].
    p = dim + 1
    V1 = np.hstack([X, np.ones((n, 1))])            # (n, dim+1)
    b = np.arange(dim)

    val = w[:, None] * V1[i]                         # (nnz, dim+1): W * [x,y,1]
    # rows: vertex i, spatial component b  (independent of the poly index)
    rows = np.broadcast_to((i[:, None] * dim + b[None, :])[:, None, :],
                           (w.shape[0], p, dim))
    # cols: DOF ((node*p + poly)*dim + b)
    poly = np.arange(p)
    cols = ((kk[:, None] * p + poly[None, :])[:, :, None] * dim + b[None, None, :])
    data = np.broadcast_to(val[:, :, None], (w.shape[0], p, dim))

    r = k * p * dim
    B = sp.sparse.coo_matrix(
        (data.ravel(), (rows.ravel(), cols.ravel())), shape=(n * dim, r)
    )
    return B.tocsr()


def lbs_affine_coordinates(
    n_nodes: int, A: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Reduced coordinates ``z`` that make every affine frame the SAME map.

    Because the skinning weights form a partition of unity, setting every control
    frame to the identical affine transform ``x -> A x + t`` makes the linear
    blend reproduce that transform *exactly* on the whole mesh: ``B z = A x + t``.
    This is the tool behind the **linear-precision / patch test** -- a subspace
    that reproduces every rigid and affine field is a necessary correctness
    property (rigid motions cost zero elastic energy, uniform stretch gives a
    uniform deformation gradient). ``A = I, t = 0`` returns the rest-state ``z``.

    The DOF ordering matches :func:`sparse_lbs_jacobian` (``order = 1``) and
    SimKit's :func:`simkit.lbs_jacobian`.

    Parameters
    ----------
    n_nodes : int
        Number of control frames ``k``.
    A : np.ndarray (dim, dim)
        Linear part of the affine transform.
    t : np.ndarray (dim,)
        Translation part.

    Returns
    -------
    z : np.ndarray (k * dim * (dim + 1), 1)
        Reduced coordinates realising ``x -> A x + t`` in the LBS subspace.
    """
    A = np.asarray(A, dtype=float)
    t = np.asarray(t, dtype=float).reshape(-1)
    d = A.shape[0]
    p = d + 1
    M = np.hstack([A, t.reshape(d, 1)])             # (d, d+1): [A | t]
    z = np.zeros((n_nodes * p * d, 1))
    for kk in range(n_nodes):
        for pp in range(p):
            for a in range(d):
                z[(kk * p + pp) * d + a, 0] = M[a, pp]
    return z


# =========================================================================== #
# Public entry point                                                           #
# =========================================================================== #
def sparse_meshless_methods_basis(
    X: np.ndarray,
    T: np.ndarray,
    ym: np.ndarray,
    *,
    n_nodes: int = 16,
    frame_order: int = 1,
    support_scale: float = 1.25,
    lloyd_iters: int = 8,
    seed_index: Optional[int] = None,
    seed_nodes: Optional[Sequence[int]] = None,
    edge_reduce: str = "mean",
    return_distances: bool = False,
):
    """Build a sparse meshless model (Faure et al. 2011) on a heterogeneous mesh.

    Given a mesh ``(X, T)`` and a per-element Young's modulus ``ym``, this places
    ``n_nodes`` control frames using the compliance-distance metric, computes the
    material-aware Voronoi skinning weights, and assembles the sparse linear-blend
    skinning subspace ``B``.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Vertex positions.
    T : np.ndarray (t, 3)
        Triangle connectivity.
    ym : np.ndarray (t,) or (t, 1)
        Per-element Young's modulus (heterogeneous material map).
    n_nodes : int, optional
        Number of control nodes / frames (the model's sparsity). Soft regions
        automatically receive proportionally more of them.
    frame_order : {0, 1}, optional
        ``1`` = affine frames (default, the paper's recommended compromise),
        ``0`` = point/translation frames.
    support_scale : float, optional
        Widens or tightens the shape-function supports (see
        :func:`voronoi_shape_functions`).
    lloyd_iters : int, optional
        Lloyd relaxation passes during node distribution.
    seed_index : int, optional
        Deterministic first farthest-point seed.
    seed_nodes : sequence of int, optional
        Vertex indices to pre-place and hold fixed (e.g. one per rigid bone).
    edge_reduce : {"mean", "min", "max"}, optional
        Edge compliance aggregation (see :func:`compliance_graph`).
    return_distances : bool, optional
        If ``True``, also return the ``(k, n)`` node-to-vertex compliance
        distance field as a fifth output (handy for debugging / visualisation).

    Returns
    -------
    W : scipy.sparse.csr_matrix (n, k)
        Material-aware skinning weights (partition of unity, compact support).
    B : scipy.sparse.csr_matrix (n*dim, r)
        Sparse linear-blend-skinning subspace, ``x = B z + x0``.
    labels : np.ndarray (t,)
        Voronoi region (control-frame index) each triangle cell belongs to.
    nodes : np.ndarray (k,)
        Control-frame vertex indices.
    D_nodes : np.ndarray (k, n)
        Compliance distance from each control frame to every vertex. Returned
        **only** when ``return_distances=True``.
    """
    X = np.asarray(X, dtype=float)
    T = np.asarray(T)

    G = compliance_graph(X, T, ym, reduce=edge_reduce)
    nodes = sample_nodes(
        X, G, n_nodes, seed_index=seed_index, seed_nodes=seed_nodes,
        lloyd_iters=lloyd_iters,
    )
    D_nodes = compliance_distances(G, nodes)        # (k, n)

    _, labels = voronoi_labels(D_nodes, T)
    W = voronoi_shape_functions(D_nodes, nodes, support_scale=support_scale)
    B = sparse_lbs_jacobian(X, W, order=frame_order)

    if return_distances:
        return W, B, labels, nodes, D_nodes
    return W, B, labels, nodes
