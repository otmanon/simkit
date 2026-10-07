"""Sparse linear-blend-skinning subspace from compact-support weights.

Sparse counterpart of :func:`simkit.lbs_jacobian`: same DOF ordering for
affine frames, but assembled directly from the non-zeros of ``W`` so that the
block structure a reduced simulation exploits is preserved (Faure et al. 2011,
Sec. 3).
"""

from __future__ import annotations

import numpy as np
import scipy as sp
import scipy.sparse


def _point_frame_jacobian(
    i: np.ndarray, kk: np.ndarray, w: np.ndarray, n: int, k: int, dim: int
) -> sp.sparse.csr_matrix:
    """``B = kron(W, I_dim)`` from the COO triplets of ``W``: ``u_v = sum_k W[v,k] t_k``."""
    rows = (i[:, None] * dim + np.arange(dim)[None, :]).ravel()
    cols = (kk[:, None] * dim + np.arange(dim)[None, :]).ravel()
    data = np.repeat(w, dim)
    return sp.sparse.coo_matrix((data, (rows, cols)), shape=(n * dim, k * dim)).tocsr()


def _affine_frame_jacobian(
    X: np.ndarray, i: np.ndarray, kk: np.ndarray, w: np.ndarray, k: int
) -> sp.sparse.csr_matrix:
    """Affine-frame LBS Jacobian from the COO triplets of ``W``.

    Per node the polynomial basis is ``[x, y, (z,) 1]`` and the DOF index is
    ``(node * (dim + 1) + poly) * dim + component``, matching
    :func:`simkit.lbs_jacobian`.
    """
    n, dim = X.shape
    p = dim + 1
    V1 = np.hstack([X, np.ones((n, 1))])  # (n, dim+1)
    b = np.arange(dim)
    nnz = w.shape[0]

    val = w[:, None] * V1[i]  # (nnz, dim+1): W * [x, y, 1]
    # rows: vertex i, spatial component b (independent of the poly index)
    rows = np.broadcast_to((i[:, None] * dim + b[None, :])[:, None, :], (nnz, p, dim))
    # cols: DOF ((node * p + poly) * dim + b)
    poly = np.arange(p)
    cols = (kk[:, None] * p + poly[None, :])[:, :, None] * dim + b[None, None, :]
    data = np.broadcast_to(val[:, :, None], (nnz, p, dim))

    B = sp.sparse.coo_matrix(
        (data.ravel(), (rows.ravel(), cols.ravel())), shape=(n * dim, k * p * dim)
    )
    return B.tocsr()


def sparse_lbs_jacobian(
    X: np.ndarray, W: sp.sparse.spmatrix | np.ndarray, order: int = 1
) -> sp.sparse.csr_matrix:
    """Sparse LBS subspace ``B`` such that ``x = B z`` on positions.

    Assembles the linear-blend-skinning Jacobian of the (compact-support)
    weights ``W`` into a **sparse** basis, preserving the block structure that
    makes a reduced simulation cheap. For affine frames the layout matches the
    dense :func:`simkit.lbs_jacobian` exactly (same DOF ordering), so it is a
    drop-in replacement that additionally keeps the matrix sparse.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Rest vertex positions.
    W : scipy.sparse matrix or np.ndarray (n, k)
        Skinning weights over ``k`` nodes, e.g. from
        :func:`simkit.voronoi_shape_functions`.
    order : {0, 1}, optional
        Frame type. ``1`` (default) = affine frames: each node carries a
        ``dim x (dim + 1)`` transform, ``dim * (dim + 1)`` DOFs, and can
        stretch, shear, bend and rotate its neighbourhood. ``0`` =
        point/translation frames: ``dim`` DOFs per node.

    Returns
    -------
    B : scipy.sparse.csr_matrix (n*dim, r)
        Reduced subspace. ``r = k * dim * (dim + 1)`` for affine frames,
        ``r = k * dim`` for point frames. The map is on *positions* for affine
        frames (see :func:`simkit.lbs_affine_coordinates` for the rest-state
        ``z``) and on *displacements* for point frames.

    Raises
    ------
    ValueError
        If ``order`` is not 0 or 1.
    """
    X = np.asarray(X, dtype=float)
    n, dim = X.shape
    Wc = sp.sparse.csr_matrix(W).tocoo()
    i, kk, w = Wc.row, Wc.col, Wc.data
    k = Wc.shape[1]

    if order == 0:
        return _point_frame_jacobian(i, kk, w, n, k, dim)
    if order == 1:
        return _affine_frame_jacobian(X, i, kk, w, k)
    raise ValueError("only order 0 (point) and 1 (affine) frames are supported")
