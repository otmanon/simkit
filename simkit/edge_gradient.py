"""Sparse gradient operator on mesh edges (inverse length weights)."""

from __future__ import annotations

import numpy as np
import scipy as sp

from .edge_lengths import edge_lengths


def edge_gradient(X: np.ndarray, E: np.ndarray) -> sp.sparse.csc_matrix:
    """Sparse edge gradient ``G`` with rows ``(x_i - x_j) / l_e``.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Vertex positions.
    E : np.ndarray (m, 2)
        Edge vertex indices.

    Returns
    -------
    G : scipy.sparse.csc_matrix (m, n)
        Edge gradient operator mapping vertex values to per-edge differences
        scaled by inverse edge length.
    """
    l = edge_lengths(X, E)
    li = 1 / l
    I = np.repeat(np.arange((E.shape[0]))[:, None], 2, axis=1)
    J = E
    V = np.hstack([-li[:, None], li[:, None]])
    G = sp.sparse.csc_matrix(
        (V.flatten(), (I.flatten(), J.flatten())),
        shape=(E.shape[0], X.shape[0]),
    )
    return G
