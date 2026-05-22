"""Rotation-corrected SVD (Rusinkiewicz-style ``svd_rv``)."""

from typing import Tuple

import numpy as np


def svd_rv(F: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD with rotation fix so ``det(U V^T) >= 0``.

    For each square matrix ``F``, returns ``U``, diagonal ``S``, and ``V`` such
    that ``F = U S V^T`` and ``U V^T`` is a proper rotation when possible.
    Follows the construction in Kim et al., *Dynamic Deformables*.

    Parameters
    ----------
    F : np.ndarray (n, d, d)
        Batch of square deformation gradients.

    Returns
    -------
    U : np.ndarray (n, d, d)
        Left singular vectors (sign-corrected).
    S : np.ndarray (n, d, d)
        Diagonal matrices of singular values.
    V : np.ndarray (n, d, d)
        Right singular vectors (sign-corrected).
    """
    dim = F.shape[-1]
    F = F.reshape(-1, dim, dim)

    d = F.shape[1]
    [U, S, VT] = np.linalg.svd(F)

    V = VT.transpose([0, 2, 1])

    I = np.tile(np.identity(d), (F.shape[0], 1, 1))

    S = I * S[:, None, :]

    L = I
    L[:, d - 1, d - 1] = np.linalg.det(U @ V.transpose(0, 2, 1))

    detU = np.linalg.det(U)
    detV = np.linalg.det(V)
    uI = np.logical_and(detU < 0, detV > 0)[:, None, None]
    vI = np.logical_and(detV < 0, detU > 0)[:, None, None]

    Ut = uI * U @ L + np.logical_not(uI) * U
    Vt = vI * V @ L + np.logical_not(vI) * V
    St = S @ L

    return Ut, St, Vt
