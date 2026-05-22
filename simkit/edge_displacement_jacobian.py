"""Sparse Jacobian of edge displacement w.r.t. vertex positions."""

from __future__ import annotations

import numpy as np
import scipy as sp


def edge_displacement_jacobian(X: np.ndarray, E: np.ndarray) -> sp.sparse.csc_matrix:
    """Jacobian of edge displacement w.r.t. vertex positions.

    Each row maps the two endpoint vertex indices to ``+1`` and ``-1``,
    giving the displacement ``x_i - x_j`` for edge ``(i, j)``.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Vertex positions (only ``n`` is used for matrix size).
    E : np.ndarray (m, 2)
        Edge vertex indices.

    Returns
    -------
    J : scipy.sparse.csc_matrix (m, n)
        Sparse Jacobian of per-edge displacement w.r.t. vertex indices.
    """
    ones = np.ones((E.shape[0], 1))
    vals = np.concatenate([ones, -ones], axis=1)

    ii = np.repeat(np.arange(E.shape[0])[:, None], 2, axis=1)
    jj = E

    J = sp.sparse.csc_matrix(
        (vals.flatten(), (ii.flatten(), jj.flatten())),
        (E.shape[0], X.shape[0]),
    )

    return J
