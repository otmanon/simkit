"""Normalized correlation matrix projected into a reduced subspace."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import scipy as sp


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

    if sp.sparse.issparse(H):
        diag = H.diagonal()
    else:
        diag = np.diag(H)

    inv_scale = 1.0 / np.sqrt(np.maximum(diag[:, None] * diag[None, :], 1e-30))

    if sp.sparse.issparse(H):
        rho = H.multiply(inv_scale)
        return rho.tocsc()

    return sp.sparse.csc_matrix(H * inv_scale)
