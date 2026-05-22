"""Edge-based graph Laplacian weighted by edge lengths."""

from __future__ import annotations

import numpy as np
import scipy as sp

from .edge_gradient import edge_gradient
from .edge_lengths import edge_lengths


def edge_laplacian(X: np.ndarray, E: np.ndarray) -> sp.sparse.csc_matrix:
    """Laplacian ``G^T diag(l) G`` from edge lengths and the edge gradient.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Vertex positions.
    E : np.ndarray (m, 2)
        Edge vertex indices.

    Returns
    -------
    L : scipy.sparse.csc_matrix (n, n)
        Edge-weighted Laplacian on vertices.
    """
    l = edge_lengths(X, E)
    A = sp.sparse.diags(l)
    G = edge_gradient(X, E)
    L = G.T @ A @ G
    return L
