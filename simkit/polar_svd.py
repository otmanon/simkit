"""Polar decomposition and rotation-corrected SVD (``svd_rv``)."""

from typing import Tuple

import numpy as np


def svd_rv(F: np.ndarray, flip: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD with optional rotation fix so ``det(U V^T) >= 0``.

    When ``flip`` is True, adjusts ``U`` and ``V`` so the composed rotation
    has non-negative determinant (Rusinkiewicz-style ``svd_rv``).

    Parameters
    ----------
    F : np.ndarray (t, d, d) or (d, d)
        Batch of square matrices (or a single matrix, promoted to batch size 1).
    flip : bool, optional
        Apply the rotation fix. Default True.

    Returns
    -------
    U : np.ndarray (t, d, d)
        Left singular vectors (possibly sign-corrected).
    S : np.ndarray (t, d, d)
        Diagonal matrices of singular values.
    V : np.ndarray (t, d, d)
        Right singular vectors (possibly sign-corrected).
    """
    if len(F.shape) == 2:
        F = F[None, :, :]

    d = F.shape[1]
    [U, S, VT] = np.linalg.svd(F)

    # check if S
    V = VT.transpose([0, 2, 1])

    I = np.tile(np.identity(d), (F.shape[0], 1, 1))
    S = I * S[:, None, :]

    L = I

    if flip:
        L[:, d - 1, d - 1] = np.linalg.det(U @ V)

    detU = np.linalg.det(U)
    detV = np.linalg.det(V)
    uI = np.logical_and(detU < 0, detV > 0)[:, None, None]
    vI = np.logical_and(detV < 0, detU > 0)[:, None, None]

    Ut = uI * U @ L + np.logical_not(uI) * U
    Vt = vI * V @ L + np.logical_not(vI) * V
    St = S @ L

    return Ut, St, Vt


def polar_svd(F: np.ndarray, flip: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Polar decomposition ``F ≈ R S`` via SVD.

    Parameters
    ----------
    F : np.ndarray (t, d, d) or (d, d)
        Batch of square matrices (or a single matrix).
    flip : bool, optional
        If True, use :func:`svd_rv` for a proper rotation factor; otherwise
        plain ``numpy.linalg.svd``. Default True.

    Returns
    -------
    R : np.ndarray (t, d, d)
        Rotation (or closest rotation when ``flip`` is True).
    SS : np.ndarray (t, d, d)
        Symmetric stretch factor ``V S V^T``.
    """
    if flip:
        [U, S, V] = svd_rv(F, flip=flip)
        VT = V.transpose([0, 2, 1])
    else:
        [U, s, VT] = np.linalg.svd(F)
        V = VT.transpose([0, 2, 1])
        I = np.tile(np.identity(d), (F.shape[0], 1, 1))
        S = I * S[:, None, :]

    R = U @ VT
    SS = V @ S @ VT

    return R, SS
