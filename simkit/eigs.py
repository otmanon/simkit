"""Generalized sparse eigenvalue solver with UMFPACK linear operator.

Computes a few generalized eigenpairs of an indefinite sparse matrix ``A``
with optional mass matrix ``M``, using UMFPACK for the required linear solves
instead of SciPy's default SuperLU backend.
"""

from typing import Optional, Tuple

import cvxopt
import cvxopt.umfpack
import numpy as np
import scipy as sp
from scipy.sparse import hstack, vstack
from scipy.sparse.linalg import LinearOperator


class umfpack_LU_LinearOperator(LinearOperator):
    """Sparse linear operator backed by a UMFPACK LU factorization.

    Overrides SciPy's default LU factorization (SuperLU) with UMFPACK for
    repeated solves ``A^{-1} v`` inside :func:`scipy.sparse.linalg.eigs`.

    Parameters
    ----------
    A : scipy.sparse matrix (n, n)
        Sparse matrix to factor.
    """

    def __init__(self, A: sp.sparse.spmatrix) -> None:
        self.A = A
        self.shape = A.shape
        self.dtype = A.dtype
        self.A = A

        [I, J] = A.nonzero()
        v = A[I, J]
        Ac = cvxopt.spmatrix(v, I, J, A.shape)
        F = cvxopt.umfpack.symbolic(Ac)
        self.numeric = cvxopt.umfpack.numeric(Ac, F)
        self.A = Ac
        # bc = cvxopt.matrix(b)
        # cvxopt.umfpack.linsolve(Ac, bc)
        # cnp = np.array(bc)
        super(umfpack_LU_LinearOperator, self).__init__(A.dtype, A.shape)

    def _matvec(self, v: np.ndarray) -> np.ndarray:
        """Apply the UMFPACK LU solve ``A^{-1} v``.

        Parameters
        ----------
        v : np.ndarray (n,)
            Right-hand side vector.

        Returns
        -------
        x : np.ndarray (n,)
            Solution of ``A x = v``.
        """
        b = cvxopt.matrix(v)
        x = b
        cvxopt.umfpack.solve(self.A, self.numeric, b)
        return x


def eigs(
    A: sp.sparse.spmatrix,
    k: int = 5,
    M: Optional[sp.sparse.spmatrix] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generalized eigenvalues and eigenvectors of sparse ``A`` with mass ``M``.

    Solves ``A x = lambda M x`` for the ``k`` eigenpairs closest to zero
    (``sigma=0``, largest magnitude). When ``k`` equals the full dimension,
    falls back to dense :func:`scipy.linalg.eigh`.

    Parameters
    ----------
    A : scipy.sparse matrix (n, n)
        Indefinite sparse matrix (e.g. stiffness).
    k : int, optional
        Number of eigenpairs to compute. Default 5.
    M : scipy.sparse matrix (n, n), optional
        Mass matrix. Defaults to the identity.

    Returns
    -------
    D : np.ndarray (k,)
        Generalized eigenvalues.
    B : np.ndarray (n, k)
        Generalized eigenvectors as columns.
    """
    if k == 0:
        return np.zeros((0)), np.zeros((A.shape[0], 0))
    if M is None:
        M = sp.sparse.identity(A.shape[0])

    OpInv = umfpack_LU_LinearOperator(A)
    n0 = np.ones(A.shape[0])

    if k < A.shape[0] - 1:
        [D, B] = sp.sparse.linalg.eigsh(
            A, M=M, k=k, sigma=0, which='LM', OPinv=OpInv, v0=n0
        )
    else:
        [D, B] = sp.linalg.eigh(A.toarray(), b=M.toarray())
        B = B[:, :k]
        D = D[:k]

    return D, B
