"""Jacobian of linear blend skinning (LBS) w.r.t. the affine frame DOFs.

For rest positions ``V`` and weights ``W``, the deformed position of vertex
``i`` is ``sum_k W[i,k] * T_k(V_i)`` with affine transforms ``T_k``. This
module builds the matrix ``d x / d T`` in stacked form, either dense (the
default) or sparse (``sparse=True``), with identical DOF ordering so the two
are interchangeable. The sparse form is what compact-support weights such as
:func:`simkit.voronoi_shape_functions` call for: it preserves the block
structure a reduced simulation exploits (Faure et al. 2011, Sec. 3).
"""

from __future__ import annotations

import numpy as np
import scipy as sp
import scipy.sparse


def _dense_lbs_jacobian(V: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Dense assembly via Kronecker products."""
    n = V.shape[0]
    d = V.shape[1]
    k = W.shape[1]

    one_d1 = np.ones((d + 1, 1))
    one_k = np.ones((k, 1))

    # append 1s to V to make V1 , homogeneous
    V1 = np.hstack((V, np.ones((n, 1))))

    Wexp = np.kron(W, one_d1.T)
    V1exp = np.kron(one_k.T, V1)
    J = Wexp * V1exp
    Jexp = np.kron(J, np.identity(d))

    return Jexp


def _sparse_lbs_jacobian(V: np.ndarray, W: sp.sparse.spmatrix) -> sp.sparse.csr_matrix:
    """Sparse assembly from the non-zeros of ``W`` only.

    Per node the polynomial basis is ``[x, y, (z,) 1]`` and the DOF index is
    ``(node * (dim + 1) + poly) * dim + component``, matching the dense layout.
    """
    n, dim = V.shape
    Wc = sp.sparse.csr_matrix(W).tocoo()
    i, kk, w = Wc.row, Wc.col, Wc.data
    k = Wc.shape[1]
    p = dim + 1
    V1 = np.hstack([V, np.ones((n, 1))])  # (n, dim+1)
    b = np.arange(dim)
    nnz = w.shape[0]

    val = w[:, None] * V1[i]  # (nnz, dim+1): W * [x, y, 1]
    # rows: vertex i, spatial component b (independent of the poly index)
    rows = np.broadcast_to((i[:, None] * dim + b[None, :])[:, None, :], (nnz, p, dim))
    # cols: DOF ((node * p + poly) * dim + b)
    poly = np.arange(p)
    cols = (kk[:, None] * p + poly[None, :])[:, :, None] * dim + b[None, None, :]
    data = np.broadcast_to(val[:, :, None], (nnz, p, dim))

    J = sp.sparse.coo_matrix(
        (data.ravel(), (rows.ravel(), cols.ravel())), shape=(n * dim, k * p * dim)
    )
    return J.tocsr()


def lbs_jacobian(
    V: np.ndarray, W: np.ndarray | sp.sparse.spmatrix, sparse: bool = False
) -> np.ndarray | sp.sparse.csr_matrix:
    """Stacked Jacobian ``d x / d T`` for LBS with homogeneous transforms.

    Each of the ``k`` frames carries a ``d x (d + 1)`` affine transform
    ``[A_k | t_k]``, so there are ``k * d * (d + 1)`` DOFs. The DOF index is
    ``(k * (d + 1) + poly) * d + component`` with polynomial basis
    ``[x, y, (z,) 1]`` per frame. Because the weights sum to one, giving every
    frame the same affine map ``[A | t]`` reproduces ``x -> A x + t`` on the
    whole mesh; in this layout those coordinates are
    ``z = np.tile(np.c_[A, t].T.ravel(), k)``, and ``A = I, t = 0`` is the
    rest state.

    Parameters
    ----------
    V : np.ndarray (n, d)
        Rest vertex positions.
    W : np.ndarray or scipy.sparse matrix (n, k)
        Per-vertex skinning weights over ``k`` frames.
    sparse : bool, optional
        If ``True`` (default ``False``) assemble only the non-zeros of ``W`` and
        return a :class:`scipy.sparse.csr_matrix`. The sparse and dense results
        are numerically identical; use the sparse form with compact-support
        weights (e.g. :func:`simkit.voronoi_shape_functions`) so the block
        structure of a sparse meshless model is kept.

    Returns
    -------
    J : np.ndarray or scipy.sparse.csr_matrix (n*d, k*(d+1)*d)
        Jacobian of stacked deformed coordinates w.r.t. all frame DOFs.
        Dense ``np.ndarray`` by default, ``csr_matrix`` when ``sparse=True``.
    """
    V = np.asarray(V, dtype=float)
    if sparse:
        return _sparse_lbs_jacobian(V, W)
    if sp.sparse.issparse(W):
        W = W.toarray()
    return _dense_lbs_jacobian(V, np.asarray(W, dtype=float))
