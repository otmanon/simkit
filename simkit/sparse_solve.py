"""Direct solve of a sparse linear system, with Cholesky fast paths."""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import scipy as sp
import scipy.sparse.linalg

from ._backend import _cpp, use_cpp


def sparse_solve(A: sp.sparse.spmatrix, b: np.ndarray, symmetric: bool = True,
                 return_method: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, str]]:
    """Solve ``A x = b`` for a sparse ``A``.

    With the C++ backend and ``symmetric=True`` the system goes through
    Eigen's simplicial Cholesky factorizations: ``SimplicialLLT`` first, and
    ``SimplicialLDLT`` if the matrix is not positive definite. Only the lower
    triangle of ``A`` is read, and the symbolic analysis is reused between
    calls with the same sparsity pattern, which is what a Newton loop on a
    fixed mesh produces. If both factorizations fail, or on the NumPy
    backend, ``scipy.sparse.linalg.spsolve`` is used.

    Parameters
    ----------
    A : scipy.sparse matrix (n, n)
        System matrix.
    b : np.ndarray (n,) or (n, k)
        Right-hand side(s).
    symmetric : bool, optional
        Whether ``A`` is symmetric so the Cholesky paths may be used. Default
        ``True``.
    return_method : bool, optional
        Also return which solver produced ``x``: ``'llt'``, ``'ldlt'`` or
        ``'spsolve'``. Default ``False``.

    Returns
    -------
    x : np.ndarray
        Solution with the same shape as ``b``.
    method : str
        Only when ``return_method`` is ``True``.
    """
    b = np.asarray(b, dtype=np.float64)
    if use_cpp() and symmetric:
        # A symmetric matrix in CSR form is its own transpose in CSC form, so a
        # CSR input needs no conversion at all.
        Ac = A if sp.sparse.isspmatrix_csr(A) or sp.sparse.isspmatrix_csc(A) else A.tocsc()
        out = _cpp.sparse_solve_cholesky(Ac.indptr, Ac.indices, Ac.data, A.shape[0], b.reshape(b.shape[0], -1))
        if out is not None:
            x, method = out
            x = np.asarray(x).reshape(b.shape)
            return (x, method) if return_method else x
    x = np.asarray(sp.sparse.linalg.spsolve(A.tocsc(), b)).reshape(b.shape)
    return (x, "spsolve") if return_method else x
