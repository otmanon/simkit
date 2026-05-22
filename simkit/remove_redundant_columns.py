"""Remove mass-weighted redundant columns from a basis matrix."""

from typing import Optional

import numpy as np
import scipy as sp


def remove_redundant_columns(
    B: sp.sparse.spmatrix,
    M: Optional[sp.sparse.spmatrix] = None,
    threshold: float = 1e-16,
) -> np.ndarray:
    """Drop numerically dependent columns of ``B`` using a mass-weighted SVD.

    Scales columns by ``sqrt(diag(M))``, performs a thin SVD, and keeps only
    singular values above ``1e-14`` before unscaling.

    Parameters
    ----------
    B : scipy.sparse matrix (n, r)
        Basis or design matrix whose columns may be redundant.
    M : scipy.sparse matrix (n, n), optional
        Mass matrix for weighting. Defaults to the identity.
    threshold : float, optional
        Unused legacy parameter (retained for API compatibility).

    Returns
    -------
    B_reduced : np.ndarray (n, r')
        Full-space basis with linearly dependent columns removed.
    """

    if M is None:
        M = sp.sparse.identity(B.shape[0])

    msqrt = np.sqrt(M.diagonal())
    msqrti = 1 / msqrt
    Msqrt = sp.sparse.diags(msqrt, 0)
    Msqrti = sp.sparse.diags(msqrti, 0)
    Bm = Msqrt @ B

    [Q, R] = np.linalg.qr(Bm, mode='reduced')

    [U, s, V] = np.linalg.svd(Bm, full_matrices=False)

    sI = np.where(s > 1e-14)[0]
    S = np.diag(s)
    B2 = U @ S[:, sI] @ V[sI, :][:, sI]

    B3 = Msqrti @ B2

    return B3
