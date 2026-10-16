"""Polar decomposition and rotation-corrected SVD (``svd_rv``)."""

from typing import Tuple

import numpy as np
from ._backend import _cpp, use_cpp
from .svd_rv import svd_rv as _svd_rv


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
    if flip:
        # Same decomposition as simkit.svd_rv (NumPy or C++ backend).
        return _svd_rv(F)

    d = F.shape[1]
    [U, s, VT] = np.linalg.svd(F)
    V = VT.transpose([0, 2, 1])
    S = np.identity(d)[None, :, :] * s[:, None, :]
    return U, S, V


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
    F = np.asarray(F)
    if len(F.shape) == 2:
        F = F[None, :, :]
    if flip and use_cpp():
        return _cpp.polar_svd(F)
    [U, S, V] = svd_rv(F, flip=flip)
    VT = V.transpose([0, 2, 1])

    R = U @ VT
    SS = V @ S @ VT

    return R, SS
