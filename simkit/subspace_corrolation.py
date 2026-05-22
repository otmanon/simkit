"""Normalized correlation matrix projected into a reduced subspace."""

from typing import Optional, Union

import numpy as np
import scipy as sp

# def subspace_corrolation(C : np.ndarray | sp.sparse.csc_matrix,  sI : np.ndarray, tI : np.ndarray, B : np.ndarray | sp.sparse.csc_matrix  =None, M : sp.sparse.csc_matrix = None):

#     var_sI = C[sI, sI]
#     var_tI = C[tI, tI]

#     if B is None:
#         H = C
#     else:
#         if M is None:
#             M = sp.sparse.identity(B.shape[0])

#         BMB = B.T @ M @ B
#         P = B @ np.linalg.inv(BMB) @ (B.T @ M)
#         H = P @ C @ P.T

#     cov_sI_tI = H[sI, tI]

#     rho = cov_sI_tI / np.sqrt(var_sI * var_tI)



#     return rho


def subspace_corrolation(
    C: Union[np.ndarray, sp.sparse.csc_matrix],
    B: Optional[Union[np.ndarray, sp.sparse.csc_matrix]] = None,
    M: Optional[sp.sparse.csc_matrix] = None,
) -> sp.sparse.csc_matrix:
    """Pearson-style correlation matrix, optionally projected onto a subspace.

    When ``B`` is given, ``C`` is projected with the mass-weighted orthogonal
    projector ``P = B (B^T M B)^{-1} B^T M`` before normalization.

    Parameters
    ----------
    C : np.ndarray or scipy.sparse.csc_matrix (n, n)
        Covariance or Hessian matrix.
    B : np.ndarray or scipy.sparse.csc_matrix (n, r), optional
        Reduced basis. If None, ``C`` is used directly.
    M : scipy.sparse.csc_matrix (n, n), optional
        Mass matrix for the projection. Identity if None.

    Returns
    -------
    rho : scipy.sparse.csc_matrix (n, n)
        Entry-wise normalized correlation ``H_ij / sqrt(H_ii H_jj)``.
    """
    if B is None:
        H = C
    else:
        if M is None:
            M = sp.sparse.identity(B.shape[0])

        BMB = B.T @ M @ B
        P = B @ np.linalg.inv(BMB) @ (B.T @ M)
        H = P @ C @ P.T

    A = H.diagonal()[:, None]
    AA = A @ A.T
    inv_sqAA = 1 / np.sqrt(AA)

    rho = H * inv_sqAA
    return rho
