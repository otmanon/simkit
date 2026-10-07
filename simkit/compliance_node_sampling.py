"""Control-node distribution in the compliance metric (Faure et al. 2011, Sec. 4.4)."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import scipy as sp
import scipy.sparse

from .compliance_distances import compliance_distances


def _farthest_point_in_compliance_metric(
    G: sp.sparse.spmatrix, nodes: list[int], n_nodes: int
) -> list[int]:
    """Grow ``nodes`` to ``n_nodes`` entries by farthest-point sampling on ``G``."""
    n = G.shape[0]
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
    return nodes


def _lloyd_relaxation(
    X: np.ndarray,
    G: sp.sparse.spmatrix,
    nodes: np.ndarray,
    fixed: set[int],
    iters: int,
) -> np.ndarray:
    """Re-centre each free node at the vertex nearest its region's centroid."""
    for _ in range(max(0, iters)):
        D = compliance_distances(G, nodes)  # (k, n)
        labels = np.argmin(D, axis=0)
        moved = False
        for a in range(len(nodes)):
            if int(nodes[a]) in fixed:
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


def compliance_node_sampling(
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
    (each node re-centred within its Voronoi region). Pre-seeded nodes -- e.g.
    one per rigid part, per the paper's suggestion to avoid linear-blend-
    skinning artifacts -- are kept fixed during relaxation.

    This is the compliance-metric analogue of
    :func:`simkit.farthest_point_sampling`, which samples in Euclidean distance.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Vertex positions (used only for the Euclidean re-centring in Lloyd).
    G : scipy.sparse matrix (n, n)
        Compliance-edge graph from :func:`simkit.compliance_graph`.
    n_nodes : int
        Total number of control nodes to place (including any seeds).
    seed_index : int, optional
        First farthest-point seed. Defaults to the vertex with the smallest
        first coordinate (deterministic). Ignored when ``seed_nodes`` is given.
    seed_nodes : sequence of int, optional
        Vertex indices to pre-place and hold fixed during Lloyd relaxation.
    lloyd_iters : int, optional
        Number of Lloyd relaxation passes. ``0`` disables relaxation.

    Returns
    -------
    nodes : np.ndarray (n_nodes,)
        Vertex indices of the control nodes. Seeds come first, in the order
        given. Fewer than ``n_nodes`` are returned only if the mesh is
        disconnected and no reachable vertex remains.
    """
    X = np.asarray(X, dtype=float)
    fixed = list(dict.fromkeys(int(s) for s in (seed_nodes or [])))
    nodes = list(fixed)

    if not nodes:
        s0 = int(np.argmin(X[:, 0])) if seed_index is None else int(seed_index)
        nodes.append(s0)

    nodes = _farthest_point_in_compliance_metric(G, nodes, n_nodes)
    nodes = np.array(nodes, dtype=int)
    return _lloyd_relaxation(X, G, nodes, set(fixed), lloyd_iters)
