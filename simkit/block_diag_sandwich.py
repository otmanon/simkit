"""Assemble ``J^T blockdiag(He) J``, the shared step of every Hessian tier."""

from __future__ import annotations

from typing import Union

import numpy as np
import scipy as sp

from ._backend import _cpp, use_cpp

Matrix = Union[np.ndarray, sp.sparse.spmatrix]


def block_diag_sandwich(J: Matrix, He: np.ndarray) -> Matrix:
    """Return ``J.T @ block_diag(He) @ J`` without materializing the block diagonal.

    This is the assembly step of all ``*_hessian_x`` / ``*_hessian_u`` /
    ``*_hessian_z`` functions: the per-element Hessian blocks ``He`` are
    sandwiched by the operator ``J`` (a deformation Jacobian, an edge
    jacobian, a reduced ``J @ B``, ...). The result is sparse when ``J`` is
    sparse and dense when ``J`` is dense, exactly like the explicit product.

    Parameters
    ----------
    J : scipy.sparse matrix or np.ndarray (t*b, m)
        Operator with ``b`` rows per element.
    He : np.ndarray (t, b, b)
        Per-element (volume-weighted, optionally PSD-projected) blocks.

    Returns
    -------
    Q : scipy.sparse.csc_matrix (m, m) or np.ndarray (m, m)
        Assembled matrix.
    """
    He = np.asarray(He, dtype=np.float64)
    if He.ndim == 2:
        He = He[None, :, :]
    t, b = He.shape[0], He.shape[1]
    if J.shape[0] != t * b:
        raise ValueError(
            "block_diag_sandwich: J has %d rows but He has %d blocks of size %d" % (J.shape[0], t, b)
        )
    if use_cpp():
        if sp.sparse.issparse(J):
            Jc = J.tocsr()
            indptr, indices, data = _cpp.block_diag_sandwich_csr(Jc.indptr, Jc.indices, Jc.data, J.shape[1], He)
            return sp.sparse.csr_matrix((data, indices, indptr), shape=(J.shape[1], J.shape[1])).tocsc()
        Q = _cpp.block_diag_sandwich_dense(np.ascontiguousarray(J, dtype=np.float64), He)
        return np.asarray(Q)
    H = sp.sparse.block_diag(He)
    return J.T @ H @ J
