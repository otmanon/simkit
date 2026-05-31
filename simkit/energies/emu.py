"""EMU directional-stiffness elastic energy.

Follows the standardized layout (see :mod:`simkit.energies.arap`). EMU is a 2D
energy with a per-element direction ``d`` and directional stiffness ``a``:
``psi = 0.5 * a * (d^T F^T F d)``. Only the ``F`` representation exists.

Element tier (``*_element_F``)
    Per-element density and derivatives. Material params ``d``, ``a`` only, no ``vol``.

Global explicit tier (``*_x``)
    Takes a prebuilt deformation Jacobian ``J`` and weights ``vol``.

Self-contained tier (no suffix)
    Builds ``J`` and ``vol`` from rest geometry ``(X, T)``.

Notes
-----
The Hessian is ``kron(I, d d^T) * a``, which is positive semi-definite for
``a >= 0``; the ``psd`` flag is accepted for interface consistency but is a
no-op.
"""

from typing import Optional

import numpy as np
import scipy as sp

from ..deformation_jacobian import deformation_jacobian
from ..volume import volume


# --------------------------------------------------------------------------- #
# Element tier: deformation gradient (F) representation                       #
# --------------------------------------------------------------------------- #
def emu_energy_element_F(F: np.ndarray, d: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Per-element EMU energy density.

    Parameters
    ----------
    F : np.ndarray (t, dim, dim)
        Per-element deformation gradients.
    d : np.ndarray (t, dim)
        Per-element direction vectors.
    a : np.ndarray (t, 1)
        Per-element directional stiffness.

    Returns
    -------
    psi : np.ndarray (t, 1)
        Per-element energy densities. No quadrature weighting applied.
    """
    a = np.asarray(a).reshape(-1, 1, 1)
    FF = F.transpose(0, 2, 1) @ F
    de = d[:, :, None]
    dFFd = de.transpose(0, 2, 1) @ FF @ de
    psi = 0.5 * (a * dFFd)[:, :, 0]
    return psi.reshape(-1, 1)


def emu_gradient_element_F(F: np.ndarray, d: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Per-element first Piola-Kirchhoff stress (gradient w.r.t. ``F``).

    Parameters
    ----------
    F : np.ndarray (t, dim, dim)
        Per-element deformation gradients.
    d : np.ndarray (t, dim)
        Per-element direction vectors.
    a : np.ndarray (t, 1)
        Per-element directional stiffness.

    Returns
    -------
    P : np.ndarray (t, dim, dim)
        Per-element stress blocks. No quadrature weighting applied.
    """
    a = np.asarray(a).reshape(-1, 1, 1)
    de = d[:, :, None]
    ddT = de @ de.transpose(0, 2, 1)
    P = F @ ddT * a
    return P


def emu_hessian_element_F(F: np.ndarray, d: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Per-element Hessian of the density w.r.t. ``F`` (vectorized blocks).

    Parameters
    ----------
    F : np.ndarray (t, dim, dim)
        Per-element deformation gradients. Only the shape is used; the EMU
        Hessian is constant in ``F``.
    d : np.ndarray (t, dim)
        Per-element direction vectors.
    a : np.ndarray (t, 1)
        Per-element directional stiffness.

    Returns
    -------
    H : np.ndarray (t, dim*dim, dim*dim)
        Per-element Hessian blocks. Constant in ``F`` and positive
        semi-definite. No quadrature weighting applied.
    """
    dim = F.shape[-1]
    a = np.asarray(a).reshape(-1, 1, 1)
    de = d[:, :, None]
    ddT = de @ de.transpose(0, 2, 1)
    Q = np.kron(np.identity(dim), ddT)
    return Q * a


# --------------------------------------------------------------------------- #
# Global explicit tier: position (x) variable                                 #
# --------------------------------------------------------------------------- #
def emu_energy_x(X: np.ndarray, J: sp.sparse.spmatrix, d: np.ndarray, a: np.ndarray, vol: np.ndarray) -> float:
    """Assembled EMU energy at positions ``X``.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Current vertex positions.
    J : scipy.sparse matrix (t*dim*dim, n*dim)
        Deformation Jacobian.
    d : np.ndarray (t, dim)
        Per-element direction vectors.
    a : np.ndarray (t, 1)
        Per-element directional stiffness.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.

    Returns
    -------
    E : float
        Total EMU energy.
    """
    dim = X.shape[1]
    F = (J @ X.reshape(-1, 1)).reshape(-1, dim, dim)
    psi = emu_energy_element_F(F, d, a)
    return float((np.asarray(vol).reshape(-1, 1) * psi).sum())


def emu_gradient_x(X: np.ndarray, J: sp.sparse.spmatrix, d: np.ndarray, a: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Assembled EMU gradient w.r.t. positions ``X``.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Current vertex positions.
    J : scipy.sparse matrix (t*dim*dim, n*dim)
        Deformation Jacobian.
    d : np.ndarray (t, dim)
        Per-element direction vectors.
    a : np.ndarray (t, 1)
        Per-element directional stiffness.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.

    Returns
    -------
    g : np.ndarray (n*dim, 1)
        Assembled energy gradient.
    """
    dim = X.shape[1]
    F = (J @ X.reshape(-1, 1)).reshape(-1, dim, dim)
    P = emu_gradient_element_F(F, d, a)
    P = P * np.asarray(vol).reshape(-1, 1, 1)
    return J.transpose() @ P.reshape(-1, 1)


def emu_hessian_x(X: np.ndarray, J: sp.sparse.spmatrix, d: np.ndarray, a: np.ndarray, vol: np.ndarray, psd: bool = True) -> sp.sparse.spmatrix:
    """Assembled EMU Hessian w.r.t. positions ``X``.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Current vertex positions. Only the shape is used; the Hessian is
        constant in ``X``.
    J : scipy.sparse matrix (t*dim*dim, n*dim)
        Deformation Jacobian.
    d : np.ndarray (t, dim)
        Per-element direction vectors.
    a : np.ndarray (t, 1)
        Per-element directional stiffness.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.
    psd : bool, optional
        Accepted for interface consistency; has no effect (Hessian is already
        PSD).

    Returns
    -------
    Q : scipy.sparse.csc_matrix (n*dim, n*dim)
        Assembled energy Hessian.
    """
    dim = X.shape[1]
    F = (J @ X.reshape(-1, 1)).reshape(-1, dim, dim)
    He = emu_hessian_element_F(F, d, a)
    He = He * np.asarray(vol).reshape(-1, 1, 1)
    H = sp.sparse.block_diag(He)
    return J.transpose() @ H @ J


# --------------------------------------------------------------------------- #
# Global explicit tier: displacement (u) variable                             #
# --------------------------------------------------------------------------- #
def emu_energy_u(u: np.ndarray, J: sp.sparse.spmatrix, Jx_bar: np.ndarray, d: np.ndarray, a: np.ndarray, vol: np.ndarray) -> float:
    """Assembled EMU energy at displacement ``u`` from a reference ``x_bar``.

    Equivalent to :func:`emu_energy_x` evaluated at ``x_bar + u`` but avoids
    recomputing ``J @ x_bar`` on every call. The reference ``x_bar`` is
    arbitrary (not required to be the rest pose).

    Parameters
    ----------
    u : np.ndarray (n, dim)
        Displacement from the reference configuration.
    J : scipy.sparse matrix (t*dim*dim, n*dim)
        Deformation Jacobian.
    Jx_bar : np.ndarray (t*dim*dim, 1)
        Precomputed ``J @ x_bar.reshape(-1, 1)``.
    d : np.ndarray (t, dim)
        Per-element direction vectors.
    a : np.ndarray (t, 1)
        Per-element directional stiffness.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.

    Returns
    -------
    E : float
        Total EMU energy.
    """
    dim = u.shape[1]
    F = (J @ u.reshape(-1, 1) + Jx_bar).reshape(-1, dim, dim)
    psi = emu_energy_element_F(F, d, a)
    return float((np.asarray(vol).reshape(-1, 1) * psi).sum())


def emu_gradient_u(u: np.ndarray, J: sp.sparse.spmatrix, Jx_bar: np.ndarray, d: np.ndarray, a: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Assembled EMU gradient w.r.t. displacement ``u``.

    Parameters
    ----------
    u : np.ndarray (n, dim)
        Displacement from the reference configuration.
    J : scipy.sparse matrix (t*dim*dim, n*dim)
        Deformation Jacobian.
    Jx_bar : np.ndarray (t*dim*dim, 1)
        Precomputed ``J @ x_bar.reshape(-1, 1)``.
    d : np.ndarray (t, dim)
        Per-element direction vectors.
    a : np.ndarray (t, 1)
        Per-element directional stiffness.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.

    Returns
    -------
    g : np.ndarray (n*dim, 1)
        Assembled energy gradient.
    """
    dim = u.shape[1]
    F = (J @ u.reshape(-1, 1) + Jx_bar).reshape(-1, dim, dim)
    P = emu_gradient_element_F(F, d, a)
    P = P * np.asarray(vol).reshape(-1, 1, 1)
    return J.transpose() @ P.reshape(-1, 1)


def emu_hessian_u(u: np.ndarray, J: sp.sparse.spmatrix, Jx_bar: np.ndarray, d: np.ndarray, a: np.ndarray, vol: np.ndarray, psd: bool = True) -> sp.sparse.spmatrix:
    """Assembled EMU Hessian w.r.t. displacement ``u``.

    Parameters
    ----------
    u : np.ndarray (n, dim)
        Displacement from the reference configuration. Only the shape is used;
        the Hessian is constant in ``u``.
    J : scipy.sparse matrix (t*dim*dim, n*dim)
        Deformation Jacobian.
    Jx_bar : np.ndarray (t*dim*dim, 1)
        Precomputed ``J @ x_bar.reshape(-1, 1)``.
    d : np.ndarray (t, dim)
        Per-element direction vectors.
    a : np.ndarray (t, 1)
        Per-element directional stiffness.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.
    psd : bool, optional
        Accepted for interface consistency; has no effect (Hessian is PSD).

    Returns
    -------
    Q : scipy.sparse.csc_matrix (n*dim, n*dim)
        Assembled energy Hessian.
    """
    dim = u.shape[1]
    F = (J @ u.reshape(-1, 1) + Jx_bar).reshape(-1, dim, dim)
    He = emu_hessian_element_F(F, d, a)
    He = He * np.asarray(vol).reshape(-1, 1, 1)
    H = sp.sparse.block_diag(He)
    return J.transpose() @ H @ J


def emu_force_matrix(X: np.ndarray, d: np.ndarray, vol: np.ndarray, J: sp.sparse.spmatrix) -> sp.sparse.spmatrix:
    """Per-element EMU force-assembly matrix (utility, unit stiffness).

    Builds the linear map from unit directional stiffness to assembled nodal
    forces. Used for actuation/control set-ups rather than energy evaluation.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Current vertex positions.
    d : np.ndarray (t, dim)
        Per-element direction vectors.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.
    J : scipy.sparse matrix (t*dim*dim, n*dim)
        Deformation Jacobian.

    Returns
    -------
    K : scipy.sparse matrix (t, n*dim)
        Per-element force-assembly matrix.
    """
    dim = X.shape[1]
    F = (J @ X.reshape(-1, 1)).reshape(-1, dim, dim)
    a = np.ones((d.shape[0], 1))
    P = emu_gradient_element_F(F, d, a) * np.asarray(vol).reshape(-1, 1, 1)
    P_mat = sp.sparse.diags(P.reshape(-1)) @ J
    N = sp.sparse.kron(sp.sparse.eye(d.shape[0]), np.ones((1, dim * dim)))
    return N @ P_mat


# --------------------------------------------------------------------------- #
# Self-contained tier: builds J and vol from rest geometry                    #
# --------------------------------------------------------------------------- #
def emu_energy(X: np.ndarray, T: np.ndarray, d: np.ndarray, a: np.ndarray, U: Optional[np.ndarray] = None) -> float:
    """EMU energy, building the operator and weights from rest geometry.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Rest vertex positions. Used to build ``J`` and ``vol``.
    T : np.ndarray (t, dim+1)
        Element connectivity.
    d : np.ndarray (t, dim)
        Per-element direction vectors.
    a : np.ndarray (t, 1)
        Per-element directional stiffness.
    U : np.ndarray (n, dim), optional
        Current vertex positions. Defaults to ``X``.

    Returns
    -------
    E : float
        Total EMU energy.
    """
    if U is None:
        U = X
    J = deformation_jacobian(X, T)
    vol = volume(X, T)
    return emu_energy_x(U, J, d, a, vol)


def emu_gradient(X: np.ndarray, T: np.ndarray, d: np.ndarray, a: np.ndarray, U: Optional[np.ndarray] = None) -> np.ndarray:
    """EMU gradient, building the operator and weights from rest geometry.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Rest vertex positions.
    T : np.ndarray (t, dim+1)
        Element connectivity.
    d : np.ndarray (t, dim)
        Per-element direction vectors.
    a : np.ndarray (t, 1)
        Per-element directional stiffness.
    U : np.ndarray (n, dim), optional
        Current vertex positions. Defaults to ``X``.

    Returns
    -------
    g : np.ndarray (n*dim, 1)
        Assembled energy gradient.
    """
    if U is None:
        U = X
    J = deformation_jacobian(X, T)
    vol = volume(X, T)
    return emu_gradient_x(U, J, d, a, vol)


def emu_hessian(X: np.ndarray, T: np.ndarray, d: np.ndarray, a: np.ndarray, U: Optional[np.ndarray] = None, psd: bool = True) -> sp.sparse.spmatrix:
    """EMU Hessian, building the operator and weights from rest geometry.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Rest vertex positions.
    T : np.ndarray (t, dim+1)
        Element connectivity.
    d : np.ndarray (t, dim)
        Per-element direction vectors.
    a : np.ndarray (t, 1)
        Per-element directional stiffness.
    U : np.ndarray (n, dim), optional
        Current vertex positions. Defaults to ``X``.
    psd : bool, optional
        Accepted for interface consistency; has no effect.

    Returns
    -------
    Q : scipy.sparse.csc_matrix (n*dim, n*dim)
        Assembled energy Hessian.
    """
    if U is None:
        U = X
    J = deformation_jacobian(X, T)
    vol = volume(X, T)
    return emu_hessian_x(U, J, d, a, vol, psd=psd)