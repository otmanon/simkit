"""Mass-weighted orthonormalization of a column basis."""

from typing import Optional, Union

import numpy as np
import scipy as sp


def orthonormalize(
    B: Union[np.ndarray, sp.sparse.spmatrix],
    M: Optional[sp.sparse.spmatrix] = None,
    threshold: float = 1e-16,
) -> np.ndarray:
    """Orthonormalize columns of ``B`` with respect to mass matrix ``M``.

    Applies ``M^{1/2}``, QR factorization, then ``M^{-1/2}``, and drops
    columns whose QR diagonal sums fall below ``threshold``.

    Parameters
    ----------
    B : np.ndarray or scipy.sparse matrix (n, r)
        Basis columns to orthonormalize.
    M : scipy.sparse matrix (n, n), optional
        Symmetric positive diagonal mass matrix. Defaults to the identity.
    threshold : float, optional
        Drop columns with ``sum(|R_ii|) <= threshold``. Default 1e-16.

    Returns
    -------
    B_ortho : np.ndarray (n, r')
        Mass-orthonormal basis with rank-deficient columns removed.
    """
    if M is None:
        M = sp.sparse.identity(B.shape[0])
    # M = sp.sparse.identity(B.shape[0])

    msqrt = np.sqrt(M.diagonal())
    msqrti = 1 / msqrt
    Msqrt = sp.sparse.diags(msqrt, 0)
    Msqrti = sp.sparse.diags(msqrti, 0)
    Bm = Msqrt @ B

    [Q, R] = np.linalg.qr(Bm)

    sing = np.abs(R).sum(axis=0)
    nonsing = np.abs(R).sum(axis=1) > threshold  # check which rows are singular
    B3 = Msqrti @ Q[:, nonsing]

    # if M is None:
    #     M = sp.sparse.identity(B.shape[0])
    # # M = sp.sparse.identity(B.shape[0])

    # msqrt = np.sqrt(M.diagonal())
    # msqrti = 1 / msqrt
    # Msqrt = sp.sparse.diags(msqrt, 0)
    # Msqrti = sp.sparse.diags(msqrti, 0)
    # Bm = Msqrt @ B

    # [Q, R] = np.linalg.qr(Bm, mode='reduced')

    # [U, s, V] = np.linalg.svd(Bm, full_matrices=False)

    # sI = np.where(s > 1e-14)[0]
    # S = np.diag(s)
    # B2 = U @ S[:, sI] @ V[sI, :][:, sI]

    # B3 = Msqrti @ B2

    return B3
