"""Material-aware Voronoi skinning weights (Faure et al. 2011, Sec. 4.3).

The original paper smooths kernels by recursively subdividing Voronoi
iso-surfaces on a voxel grid, converging to shape functions that are ``1`` at
a node, ``0.5`` on its Voronoi frontier and ``0`` at the neighbouring nodes,
linear in compliance distance in between. Here that limit is written in closed
form as a clamped-linear tent whose radius is the distance to the nearest
other node, widened only where that would leave a vertex uncovered, which
keeps the weights continuous, compactly supported and exactly interpolating
without any post-processing.
"""

from __future__ import annotations

import numpy as np
import scipy as sp
import scipy.sparse


def voronoi_shape_functions(
    D_nodes: np.ndarray,
    nodes: np.ndarray,
    support_scale: float = 1.0,
) -> sp.sparse.csr_matrix:
    """Material-aware skinning weights ``W`` (the paper's Voronoi kernels).

    Every node ``a`` gets a clamped-linear tent in the compliance metric,

        ``w_a(v) = max(0, 1 - d_a(v) / R_a(v))``,
        ``R_a(v) = support_scale * max(r_a, d_(1)(v) + d_(2)(v))``,

    where ``r_a`` is the compliance distance from node ``a`` to its nearest
    other node and ``d_(1)(v) <= d_(2)(v)`` are the two smallest node distances
    of vertex ``v``. The tents are then normalised to a partition of unity.

    Wherever the per-node radius ``r_a`` covers a vertex this is the paper's
    ideal shape function: ``1`` at the node, ``0.5`` on the Voronoi frontier,
    ``0`` at the neighbouring node and *linear* in compliance distance in
    between, so across a soft joint between two stiff parts the two frames are
    blended linearly over the whole joint. Where a vertex lies beyond every
    per-node radius (thick soft regions far from any node) the radius widens
    to the sum of the vertex's two nearest node distances, so the nearest node
    always carries at least half the weight, no vertex is left uncovered and
    every support ends continuously. At a node vertex every other tent is
    zero, so with ``support_scale = 1`` the weights interpolate the nodes
    exactly. Crossing soft material is expensive in the metric, so a node's
    support does not leak through a joint into the next bone, which gives the
    sharp material interfaces the method is designed for.

    Parameters
    ----------
    D_nodes : np.ndarray (k, n)
        Compliance distances from each node to every vertex, as returned by
        :func:`simkit.compliance_distances`.
    nodes : np.ndarray (k,)
        Node vertex indices. Their rows of ``W`` are the identity.
    support_scale : float, optional
        Multiplies every tent radius. ``1.0`` (default) gives the compact,
        exactly-interpolating kernels above; ``> 1`` widens every support for
        smoother overlap (the analogue of using more Voronoi sub-divisions in
        the paper) at the price of exact interpolation, which is then
        re-imposed at the node vertices.

    Returns
    -------
    W : scipy.sparse.csr_matrix (n, k)
        Non-negative skinning weights whose rows sum to 1, with
        ``W[nodes] == I``.
    """
    D_nodes = np.asarray(D_nodes, dtype=float)
    k, n = D_nodes.shape
    nodes = np.asarray(nodes, dtype=int)
    D = np.where(np.isfinite(D_nodes), D_nodes, np.inf)

    # per-node radius: compliance distance to the nearest other node
    Dnn = D[:, nodes].copy()  # (k, k)
    np.fill_diagonal(Dnn, np.inf)
    r_node = Dnn.min(axis=1)  # (k,)
    # per-vertex floor: sum of the vertex's two nearest node distances
    r_vertex = np.partition(D, 1, axis=0)[:2].sum(axis=0) if k >= 2 else np.full(n, np.inf)
    R = support_scale * np.maximum(r_node[:, None], r_vertex[None, :])  # (k, n)
    R = np.where(np.isfinite(R) & (R > 0), R, 1.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        raw = np.maximum(0.0, 1.0 - D / R)  # (k, n)
    raw[~np.isfinite(D)] = 0.0
    if k == 1:
        raw[:] = 1.0
    W = raw.T  # (n, k)

    # Vertices unreachable from every node (disconnected mesh): keep the
    # partition of unity with a uniform row rather than dividing by zero.
    row_sum = W.sum(axis=1)
    empty = row_sum <= 1e-12
    if np.any(empty):
        W[empty, :] = 1.0 / k
        row_sum = W.sum(axis=1)
    W = W / row_sum[:, None]

    # Exact interpolation at the node vertices (a no-op for support_scale = 1).
    W[nodes, :] = 0.0
    W[nodes, np.arange(k)] = 1.0

    W[W < 1e-9] = 0.0
    return sp.sparse.csr_matrix(W)
