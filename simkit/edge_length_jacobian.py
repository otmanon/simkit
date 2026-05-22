"""Sparse Jacobian of edge lengths w.r.t. flattened vertex positions."""

from __future__ import annotations

import numpy as np
import scipy as sp


def edge_length_jacobian(X: np.ndarray, E: np.ndarray) -> sp.sparse.csc_matrix:
    """Jacobian of edge length w.r.t. flattened vertex coordinates.

    Row ``e`` gives ``d l_e / d x`` for edge ``(i, j)`` with
    ``l_e = ||x_i - x_j||``.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Vertex positions.
    E : np.ndarray (m, 2)
        Edge vertex indices.

    Returns
    -------
    dl_dx : scipy.sparse.csc_matrix (m, n*dim)
        Sparse Jacobian of per-edge lengths w.r.t. ``x`` stacked in C order.
    """
    dim = X.shape[1]
    p = X[E[:, 0], :] - X[E[:, 1], :]
    p_norm = np.linalg.norm(p, axis=1)[:, None]
    d = p / p_norm

    dldx = np.concatenate([d, -d], axis=1)

    ii = np.repeat(np.arange(E.shape[0])[:, None], dim * 2, axis=1)
    jj = np.repeat(E[:, :] * dim, dim, axis=1) + np.tile(np.arange(dim), 2)
    vals = dldx

    dl_dx = sp.sparse.csc_matrix(
        (vals.flatten(), (ii.flatten(), jj.flatten())),
        (E.shape[0], X.shape[0] * dim),
    )

    return dl_dx
