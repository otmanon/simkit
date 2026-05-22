"""Quadratic Dirichlet penalty for pinning mesh vertices.

Builds the matrices for a pinning energy ``E = 1/2 || S x - y ||_Gamma^2`` that
holds selected vertex indices fixed at target positions.
"""

from __future__ import annotations

import numpy as np
import scipy as sp


def dirichlet_penalty(
    bI: np.ndarray,
    y: np.ndarray,
    nv: int,
    gamma: float | np.ndarray,
    only_b: bool = False,
    SGamma: sp.sparse.csc_matrix | None = None,
    return_SGamma: bool = False,
) -> (
    tuple[sp.sparse.csc_matrix, np.ndarray]
    | tuple[np.ndarray]
    | tuple[sp.sparse.csc_matrix, np.ndarray, sp.sparse.csc_matrix]
    | tuple[np.ndarray, sp.sparse.csc_matrix]
):
    """Quadratic pinning penalty ``1/2 || S x - y ||_Gamma^2``.

    Expands to ``1/2 x^T S^T Gamma S x - x^T S^T Gamma S y`` (plus a constant).

    Parameters
    ----------
    bI : np.ndarray (cn, 1) or (cn,)
        Indices of vertices to pin.
    y : np.ndarray (cn, d)
        Target positions of pinned vertices.
    nv : int
        Number of vertices in the mesh.
    gamma : float or np.ndarray (cn,)
        Pinning stiffness per constrained vertex (scalar broadcasts).
    only_b : bool, optional
        If ``True``, return only the linear term ``b`` (not the quadratic ``Q``).
    SGamma : scipy.sparse.csc_matrix, optional
        Precomputed operator ``S @ Gamma``; built from ``bI``, ``y``, and ``gamma``
        when omitted.
    return_SGamma : bool, optional
        If ``True``, append ``SGamma`` to the returned tuple.

    Returns
    -------
    Q : scipy.sparse.csc_matrix (n*d, n*d), optional
        Quadratic term matrix ``SGamma @ S.T`` (omitted when ``only_b=True``).
    b : np.ndarray (n*d, 1)
        Linear term ``-SGamma @ y`` (flattened).
    SGamma : scipy.sparse.csc_matrix, optional
        Selection-and-weight operator (only when ``return_SGamma=True``).

    Raises
    ------
    AssertionError
        If ``y`` is not two-dimensional.
    """
    assert y.ndim == 2
    if SGamma is None:
        nc = bI.shape[0]
        S = sp.sparse.csc_matrix((np.ones(nc), (bI, np.arange(nc))), (nv, nc))
        d = y.shape[1]

        S = sp.sparse.kron(S, sp.sparse.eye(d))
        cn = bI.shape[0]

        if isinstance(gamma, float) or isinstance(gamma, int):
            gamma = np.ones(cn) * gamma

        Gamma = sp.sparse.diags(gamma).tocsc()
        Gamma = sp.sparse.kron(Gamma, sp.sparse.identity(d))

        SGamma = S @ Gamma

    b = -SGamma @ y.reshape(-1, 1)

    if only_b:
        out: tuple = (b,)
    else:
        Q = SGamma @ S.T
        out = (Q, b)

    if return_SGamma:
        out = out + (SGamma,)
    return out
