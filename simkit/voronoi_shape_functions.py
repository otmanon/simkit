"""Material-aware Voronoi skinning weights (Faure et al. 2011, Sec. 4.3).

The original paper smooths kernels by recursively subdividing Voronoi
iso-surfaces on a voxel grid. Here that is replaced by the closed-form
clamped-linear tent it converges to (value ``1`` at the node, ``0.5`` at the
Voronoi frontier, ``0`` at the neighbouring node), evaluated in the compliance
metric so material interfaces stay sharp.
"""

from __future__ import annotations

import numpy as np
import scipy as sp
import scipy.sparse


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
    ``0.5`` at the Voronoi frontier (half-way to the nearest neighbour) and
    ``0`` at that neighbour -- the paper's ideal, linear-in-compliance-distance
    shape function. The tents are then normalised to a partition of unity.
    Because the metric makes crossing soft material expensive, a node's
    support does not leak across a stiff bone into the flesh beyond it, giving
    the sharp interfaces the method is designed for.

    Interpolation (``w_a = 1`` at node ``a``, ``0`` at every other node) is
    enforced exactly at the node vertices so the weights are suitable for
    Dirichlet handles. Vertices beyond every node's support are assigned
    rigidly to their nearest node so the partition of unity always holds.

    Parameters
    ----------
    D_nodes : np.ndarray (k, n)
        Compliance distances from each node to every vertex, as returned by
        :func:`simkit.compliance_distances`.
    nodes : np.ndarray (k,)
        Node vertex indices (``D_nodes[:, nodes]`` are the node-to-node
        distances).
    support_scale : float, optional
        Multiplies each node's support radius. ``1.0`` gives compact,
        exactly-interpolating tents; ``>1`` widens supports for smoother
        overlap (the analogue of using more Voronoi sub-divisions in the
        paper).

    Returns
    -------
    W : scipy.sparse.csr_matrix (n, k)
        Non-negative skinning weights whose rows sum to 1, with
        ``W[nodes] == I``.
    """
    D_nodes = np.asarray(D_nodes, dtype=float)
    k, n = D_nodes.shape
    nodes = np.asarray(nodes, dtype=int)

    # nearest-other-node compliance distance -> per-node support radius
    Dnn = D_nodes[:, nodes].copy()  # (k, k)
    np.fill_diagonal(Dnn, np.inf)
    R = support_scale * np.min(Dnn, axis=1)  # (k,)
    R = np.where(np.isfinite(R) & (R > 0), R, 1.0)

    raw = np.maximum(0.0, 1.0 - D_nodes / R[:, None])  # (k, n)
    raw[~np.isfinite(D_nodes)] = 0.0
    W = raw.T  # (n, k)

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
