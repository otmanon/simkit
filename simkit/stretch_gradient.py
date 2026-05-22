"""Gradient of the polar stretch factor w.r.t. deformation gradient and DOFs."""

from typing import Optional

import numpy as np
import scipy as sp

from .polar_svd import polar_svd
from .rotation_gradient import rotation_gradient_F


def stretch_gradient(F: np.ndarray) -> np.ndarray:
    """Gradient of stretch w.r.t. ``F`` (alias for :func:`stretch_gradient_dF`).

    Parameters
    ----------
    F : np.ndarray (t, d, d)
        Batch of deformation gradients.

    Returns
    -------
    dSdF : np.ndarray (t, d, d, d, d)
        Derivative of the symmetric stretch factor w.r.t. each ``F``.
    """
    return stretch_gradient_dF(F)


def stretch_gradient_dF(F: np.ndarray) -> np.ndarray:
    """Gradient of the polar stretch factor w.r.t. ``F``.

    Uses the rotation gradient and the identity term from ``R @ I`` in the
    polar decomposition ``F = R S``.

    Parameters
    ----------
    F : np.ndarray (t, d, d)
        Batch of deformation gradients.

    Returns
    -------
    dSdF : np.ndarray (t, d, d, d, d)
        Derivative of the symmetric stretch factor w.r.t. each ``F``.
    """
    [R, S] = polar_svd(F)
    dim = F.shape[-1]
    dRdF = rotation_gradient_F(F).reshape(-1, dim, dim, dim, dim)
    dRdFF = np.einsum('...mnki, ...kj->...mnij', dRdF, F)
    I = np.zeros((dim, dim, dim, dim))
    for i in range(dim):
        for j in range(dim):
            I[i, j, i, j] = 1
    RI = np.einsum('...ki, ...mnkj->...mnij', R, I)
    dSdF = dRdFF + RI
    return dSdF


def stretch_gradient_dx(
    X: np.ndarray,
    J: sp.sparse.spmatrix,
    Ci: Optional[sp.sparse.spmatrix] = None,
    dim: Optional[int] = None,
    Jq: Optional[np.ndarray] = None,
) -> sp.sparse.spmatrix:
    """Gradient of stretch w.r.t. vertex positions ``X``.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Vertex positions (reshaped internally to a column vector).
    J : scipy.sparse matrix (t * d * d, n * dim)
        Deformation Jacobian map.
    Ci : scipy.sparse matrix, optional
        Symmetric-stretch reduction map applied on the right.
    dim : int, optional
        Spatial dimension. Inferred from ``X.shape[1]`` if None.
    Jq : np.ndarray, optional
        Constant offset added to ``J @ x`` before reshaping to ``F``.

    Returns
    -------
    dsdx : scipy.sparse matrix
        Stretch gradient w.r.t. the stacked vertex coordinates.
    """
    if dim is None:
        dim = X.shape[1]
    x = X.reshape(-1, 1)

    if Jq is None:
        F = (J @ x).reshape(-1, dim, dim)
    else:
        F = (J @ x + Jq).reshape(-1, dim, dim)

    dSdF = stretch_gradient(F).reshape(-1, dim * dim, dim * dim)
    dSdFb = sp.sparse.block_diag(dSdF)
    dsdx = J.T @ dSdFb

    if Ci is not None:
        dsdx = dsdx @ Ci.T
    return dsdx


def stretch_gradient_dz(
    z: np.ndarray,
    GJB: sp.sparse.spmatrix,
    dim: int,
    Ci: Optional[sp.sparse.spmatrix] = None,
    GJq: Optional[np.ndarray] = None,
) -> sp.sparse.spmatrix:
    """Gradient of stretch w.r.t. reduced coordinates ``z``.

    Parameters
    ----------
    z : np.ndarray (r, 1)
        Reduced coordinates.
    GJB : scipy.sparse matrix
        Grouped deformation Jacobian map in the subspace.
    dim : int
        Spatial dimension.
    Ci : scipy.sparse matrix, optional
        Symmetric-stretch reduction map.
    GJq : np.ndarray, optional
        Constant offset in the grouped Jacobian map.

    Returns
    -------
    dsdz : scipy.sparse matrix
        Stretch gradient w.r.t. ``z``.
    """
    dsdx = stretch_gradient_dx(z, GJB, Ci, dim, Jq=GJq)

    return dsdx
