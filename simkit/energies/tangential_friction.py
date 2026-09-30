"""Lagged viscous friction: a penalty on tangential motion away from anchors.

For points in contact at the start of a (time or load) step, record anchors
``p0_j`` and contact normals ``n_j``. During the step, sliding along the contact
surface away from the anchor costs::

    E = 1/2 * sum_j w_j * | (I - n_j n_j^T) (p_j - p0_j) |^2,   p = (S x).reshape(m, dim)

while motion along the normal is free. With the anchors, normals and active set
frozen for the step the energy is exactly quadratic in ``x``, so its gradient and
Hessian are exact. Only the global explicit tier (``*_x``) is provided; ``S``
holds the rows of the active points only.
"""

from __future__ import annotations

import numpy as np
import scipy as sp


def _projectors(n: np.ndarray, w: np.ndarray) -> sp.sparse.csr_matrix:
    n = np.atleast_2d(n)
    dim = n.shape[1]
    T = np.eye(dim)[None] - n[:, :, None] * n[:, None, :]
    return sp.sparse.block_diag(list(np.asarray(w).ravel()[:, None, None] * T), format="csr")


def tangential_friction_energy_x(x: np.ndarray, S: sp.sparse.spmatrix, p0: np.ndarray, n: np.ndarray,
                                 w: np.ndarray) -> float:
    """Friction energy.

    Parameters
    ----------
    x : np.ndarray (N,) or (N, 1)
        Degrees of freedom.
    S : scipy.sparse matrix (m*dim, N)
        Maps ``x`` to the stacked positions of the anchored points.
    p0 : np.ndarray (m, dim)
        Anchor positions.
    n : np.ndarray (m, dim)
        Unit contact normals at the anchors.
    w : np.ndarray (m,)
        Per-point friction weights (e.g. ``k_f`` times lumped area).

    Returns
    -------
    energy : float
    """
    r = S @ np.asarray(x).reshape(-1) - np.asarray(p0).reshape(-1)
    return float(0.5 * r @ (_projectors(n, w) @ r))


def tangential_friction_gradient_x(x: np.ndarray, S: sp.sparse.spmatrix, p0: np.ndarray, n: np.ndarray,
                                   w: np.ndarray) -> np.ndarray:
    """Friction gradient; shape ``(N, 1)``."""
    r = S @ np.asarray(x).reshape(-1) - np.asarray(p0).reshape(-1)
    return S.T @ (_projectors(n, w) @ r).reshape(-1, 1)


def tangential_friction_hessian_x(x: np.ndarray, S: sp.sparse.spmatrix, p0: np.ndarray, n: np.ndarray,
                                  w: np.ndarray) -> sp.sparse.csr_matrix:
    """Friction Hessian ``S^T blockdiag(w (I - n n^T)) S`` (constant, PSD)."""
    return (S.T @ _projectors(n, w) @ S).tocsr()
