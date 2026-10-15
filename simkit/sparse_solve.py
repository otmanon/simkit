"""Direct solve of a sparse linear system, with a Cholesky fast path."""

from __future__ import annotations

import numpy as np
import scipy as sp
import scipy.sparse.linalg

from ._backend import _cpp, use_cpp


def sparse_solve(A: sp.sparse.spmatrix, b: np.ndarray, symmetric: bool = True) -> np.ndarray:
    """Solve ``A x = b`` for a sparse ``A``.

    With the C++ backend and ``symmetric=True`` the system is factored with
    Eigen's ``SimplicialLDLT`` (only the lower triangle of ``A`` is read), which
    is several times faster than SuperLU on the symmetric positive definite
    systems a Newton step produces. If that factorization fails, or on the
    NumPy backend, ``scipy.sparse.linalg.spsolve`` is used.

    Parameters
    ----------
    A : scipy.sparse matrix (n, n)
        System matrix.
    b : np.ndarray (n,) or (n, k)
        Right-hand side(s).
    symmetric : bool, optional
        Whether ``A`` is symmetric so the Cholesky path may be used. Default
        ``True``.

    Returns
    -------
    x : np.ndarray
        Solution with the same shape as ``b``.
    """
    b = np.asarray(b, dtype=np.float64)
    if use_cpp() and symmetric:
        Ac = A.tocsc()
        x = _cpp.sparse_solve_ldlt(Ac.indptr, Ac.indices, Ac.data, A.shape[0], b.reshape(b.shape[0], -1))
        if x is not None:
            return np.asarray(x).reshape(b.shape)
    x = sp.sparse.linalg.spsolve(A.tocsc(), b)
    return np.asarray(x).reshape(b.shape)
