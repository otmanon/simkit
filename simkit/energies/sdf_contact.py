"""Cubic penalty contact against a rigid object given by its signed distance.

Points ``P = (S x).reshape(m, dim)`` (for example, a mesh's surface vertices, or a
fine surface reached through a subspace map) are pushed out of an object whose
signed distance ``phi`` is known analytically::

    E = k / 3 * sum_j a_j * max(0, -phi(p_j))^3

``a_j`` are per-point weights (e.g. lumped surface areas). The penalty is C^2, so
Newton sees a continuous Hessian as a point enters contact. Only the global
explicit tier (``*_x``, prebuilt ``S``) is provided; the Hessian is the PSD
Gauss-Newton part ``2 k a d n n^T`` (the SDF curvature term is dropped).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import scipy as sp


def _depth(x, S, sdf, dim):
    P = (S @ np.asarray(x).reshape(-1)).reshape(-1, dim)
    phi, n = sdf(P)
    return np.maximum(-phi, 0.0), n


def sdf_contact_energy_x(x: np.ndarray, S: sp.sparse.spmatrix, sdf: Callable, k: float, a: np.ndarray,
                         dim: int = 3) -> float:
    """Contact energy.

    Parameters
    ----------
    x : np.ndarray (N,) or (N, 1)
        Degrees of freedom.
    S : scipy.sparse matrix (m*dim, N)
        Maps ``x`` to the stacked contact points.
    sdf : callable
        ``sdf(P) -> (phi, grad)`` with ``P`` of shape ``(m, dim)``.
    k : float
        Penalty stiffness.
    a : np.ndarray (m,)
        Per-point weights.
    dim : int, optional
        Spatial dimension.

    Returns
    -------
    energy : float
    """
    d, _ = _depth(x, S, sdf, dim)
    return float(k / 3.0 * (np.asarray(a).ravel() * d ** 3).sum())


def sdf_contact_gradient_x(x: np.ndarray, S: sp.sparse.spmatrix, sdf: Callable, k: float, a: np.ndarray,
                           dim: int = 3) -> np.ndarray:
    """Contact gradient, ``S^T [-k a d^2 grad(phi)]``; shape ``(N, 1)``."""
    d, n = _depth(x, S, sdf, dim)
    g = (-k * np.asarray(a).ravel() * d ** 2)[:, None] * n
    return S.T @ g.reshape(-1, 1)


def sdf_contact_hessian_x(x: np.ndarray, S: sp.sparse.spmatrix, sdf: Callable, k: float, a: np.ndarray,
                          dim: int = 3) -> sp.sparse.csr_matrix:
    """Gauss-Newton contact Hessian, ``S^T blockdiag(2 k a d n n^T) S``; PSD."""
    d, n = _depth(x, S, sdf, dim)
    act = np.nonzero(d > 0)[0]
    N = S.shape[1]
    if len(act) == 0:
        return sp.sparse.csr_matrix((N, N))
    rows = (dim * act[:, None] + np.arange(dim)).ravel()
    Sa = sp.sparse.csr_matrix(S)[rows]
    w = 2.0 * k * np.asarray(a).ravel()[act] * d[act]
    blocks = w[:, None, None] * n[act][:, :, None] * n[act][:, None, :]
    return (Sa.T @ sp.sparse.block_diag(list(blocks), format="csr") @ Sa).tocsr()
