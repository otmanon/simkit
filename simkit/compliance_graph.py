"""Compliance-weighted mesh-edge graph (Faure et al. 2011).

The *compliance distance* of "Sparse Meshless Models of Complex Deformable
Solids" (Faure, Gilles, Bousquet, Pai, SIGGRAPH 2011) is a shortest-path metric
in which every step costs ``compliance x length`` with ``compliance = 1 / E``.
Stiff regions shrink to near-points in this metric and soft regions stretch
out, which is what lets a sparse set of control frames resolve a heterogeneous
stiffness map. This module builds the weighted graph that metric lives on; see
:func:`simkit.compliance_distances` for the shortest paths and
:func:`simkit.sparse_meshless_methods_basis` for the full pipeline.
"""

from __future__ import annotations

import numpy as np
import scipy as sp
import scipy.sparse


def _element_compliance(ym: np.ndarray) -> np.ndarray:
    """Per-element compliance ``c = 1 / E`` from a Young's modulus array."""
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
    ``||X_i - X_j|| * c_ij`` where ``c_ij`` is the compliance ``1 / E`` of the
    material the edge passes through, aggregated from the edge's incident
    triangles. Crossing stiff material is therefore *cheap* and crossing soft
    material is *expensive*, which is exactly the metric the paper's compliance
    distance induces.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Vertex positions.
    T : np.ndarray (t, 3)
        Triangle connectivity.
    ym : np.ndarray (t,) or (t, 1)
        Per-element Young's modulus (heterogeneous material map).
    reduce : {"mean", "min", "max"}, optional
        How to combine the compliances of the triangles incident to a shared
        edge. ``"mean"`` (default) averages them (a symmetric, interface-sharp
        choice); ``"min"`` follows the paper's "stiffest path" reading
        literally; ``"max"`` is the most conservative (softest incident
        material wins).

    Returns
    -------
    G : scipy.sparse.csr_matrix (n, n)
        Symmetric non-negative edge-weight matrix suitable for
        :func:`scipy.sparse.csgraph.dijkstra` and
        :func:`simkit.compliance_distances`.

    Examples
    --------
    >>> import numpy as np, simkit
    >>> X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    >>> T = np.array([[0, 1, 2]])
    >>> G = simkit.compliance_graph(X, T, ym=np.array([2.0]))
    >>> G[0, 1]          # length 1 * compliance 0.5
    0.5
    """
    X = np.asarray(X, dtype=float)
    T = np.asarray(T)
    n = X.shape[0]
    c = _element_compliance(ym)

    # All three edges of every triangle, canonicalised to (min, max).
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
        raise ValueError(f"unknown reduce={reduce!r}; expected 'mean', 'min' or 'max'")

    lengths = np.linalg.norm(X[Eu[:, 0]] - X[Eu[:, 1]], axis=1)
    w = lengths * c_edge

    G = sp.sparse.coo_matrix((w, (Eu[:, 0], Eu[:, 1])), shape=(n, n))
    G = G + G.T  # symmetric / undirected
    return G.tocsr()
