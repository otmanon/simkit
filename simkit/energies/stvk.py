"""Saint Venant-Kirchhoff (StVK) elastic energy.

Implements the classical Saint Venant-Kirchhoff hyperelastic energy density
in the form taught in the FEM-deformables course notes by Sifakis and
Barbic (http://barbic.usc.edu/femdefo/):

    psi(F) = mu * tr(E**2) + (lam / 2) * tr(E)**2
    E      = (1/2) * (F^T F - I)              (Green strain tensor)

The first Piola-Kirchhoff stress and Hessian have clean closed forms

    P                = F * S
    S                = 2*mu*E + lam*tr(E)*I    (second Piola-Kirchhoff stress)
    d2 psi / dF dF   = 2*mu * delta_ik * E_lj
                     + mu * F_il * F_kj
                     + mu * (F F^T)_ik * delta_jl
                     + lam * F_ij * F_kl
                     + lam * tr(E) * delta_ik * delta_jl

Note that StVK is *not* inversion robust: because the energy depends on F
only through ``F^T F``, it cannot distinguish ``F`` from ``-F`` and admits
spurious low-energy inverted configurations.

Follows the standardized three-tier layout (see :mod:`simkit.energies.arap`
for the reference). This energy has only the deformation gradient (``F``)
representation, so there is no ``_S`` tier:

Element tier (``*_element_F``)
    Per-element density and derivative blocks. Material parameters ``mu`` and
    ``lam`` only: no quadrature weight ``vol``, no summation, no operator.

Global explicit tier (``*_x``)
    Takes a prebuilt deformation Jacobian ``J`` and weights ``vol``, calls the
    element tier, weights, and assembles.

Self-contained tier (no suffix)
    Builds ``J`` and ``vol`` from rest geometry ``(X, T)``.

Notes
-----
Closed-form gradient and Hessian derived from the energy structure and
verified symbolically by ``scripts/derive_stvk_and_neo_hookean.py``.
The element-tier Hessian blocks are returned in row-major ``F`` layout
(``H[t, i*dim + j, k*dim + l]``), which matches the layout produced by
``(J @ x).reshape(-1, dim, dim)``.
"""

from typing import Optional

import numpy as np
import scipy as sp

from ..deformation_jacobian import deformation_jacobian
from ..volume import volume
from ..psd_project import psd_project


# --------------------------------------------------------------------------- #
# Element tier: deformation gradient (F) representation                       #
# --------------------------------------------------------------------------- #
def stvk_energy_element_F(F: np.ndarray, mu: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Per-element Saint Venant-Kirchhoff energy density.

    Parameters
    ----------
    F : np.ndarray (t, dim, dim)
        Per-element deformation gradients.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.

    Returns
    -------
    psi : np.ndarray (t, 1)
        Per-element energy densities. No quadrature weighting applied.
        Zero at the rest state ``F = I``.
    """
    dim = F.shape[-1]
    F = F.reshape(-1, dim, dim)
    mu = np.asarray(mu).reshape(-1, 1)
    lam = np.asarray(lam).reshape(-1, 1)

    if dim not in (2, 3):
        raise ValueError("StVK supports dim=2 or dim=3")

    C = F.swapaxes(-1, -2) @ F                                    # (t, d, d)
    E = 0.5 * (C - np.eye(dim))                                   # (t, d, d)
    tr_E = np.trace(E, axis1=-2, axis2=-1).reshape(-1, 1)         # (t, 1)
    tr_E2 = (E ** 2).sum(axis=(-1, -2)).reshape(-1, 1)            # (t, 1), == E:E

    psi = mu * tr_E2 + (lam / 2.0) * tr_E ** 2
    return psi


def stvk_gradient_element_F(F: np.ndarray, mu: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Per-element first Piola-Kirchhoff stress (gradient w.r.t. ``F``).

    Uses the closed form ``P = F * (2 mu E + lam tr(E) I)``.

    Parameters
    ----------
    F : np.ndarray (t, dim, dim)
        Per-element deformation gradients.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.

    Returns
    -------
    P : np.ndarray (t, dim, dim)
        Per-element PK1 stress blocks. No quadrature weighting applied.
    """
    dim = F.shape[-1]
    if dim not in (2, 3):
        raise ValueError("StVK supports dim=2 or dim=3")

    mu_b = np.asarray(mu).reshape(-1, 1, 1)
    lam_b = np.asarray(lam).reshape(-1, 1, 1)

    C = F.swapaxes(-1, -2) @ F
    E = 0.5 * (C - np.eye(dim))
    tr_E = np.trace(E, axis1=-2, axis2=-1).reshape(-1, 1, 1)
    S = 2.0 * mu_b * E + lam_b * tr_E * np.eye(dim)
    P = F @ S
    return P


def stvk_hessian_element_F(F: np.ndarray, mu: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Per-element Hessian of the density w.r.t. ``F`` (vectorized blocks).

    Implements the structural formula

        H[ij, kl] = 2*mu * delta_ik * E_lj
                  + mu * F_il * F_kj
                  + mu * (F F^T)_ik * delta_jl
                  + lam * F_ij * F_kl
                  + lam * tr(E) * delta_ik * delta_jl

    Parameters
    ----------
    F : np.ndarray (t, dim, dim)
        Per-element deformation gradients.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.

    Returns
    -------
    H : np.ndarray (t, dim*dim, dim*dim)
        Per-element Hessian blocks in row-major ``F`` layout
        (``H[t, i*dim + j, k*dim + l]``). No quadrature weighting applied.
        Not PSD-projected; projection happens in the global tier.
    """
    dim = F.shape[-1]
    if dim not in (2, 3):
        raise ValueError("StVK supports dim=2 or dim=3")

    t = F.shape[0]
    mu_b = np.asarray(mu).reshape(-1, 1, 1, 1, 1)
    lam_b = np.asarray(lam).reshape(-1, 1, 1, 1, 1)

    C = F.swapaxes(-1, -2) @ F
    E = 0.5 * (C - np.eye(dim))
    FFT = F @ F.swapaxes(-1, -2)
    tr_E = np.trace(E, axis1=-2, axis2=-1).reshape(-1, 1, 1, 1, 1)
    Id = np.eye(dim)

    H5 = (
        2.0 * mu_b * np.einsum("ik,tlj->tijkl", Id, E)
        + mu_b * np.einsum("til,tkj->tijkl", F, F)
        + mu_b * np.einsum("tik,jl->tijkl", FFT, Id)
        + lam_b * np.einsum("tij,tkl->tijkl", F, F)
        + lam_b * tr_E * np.einsum("ik,jl->ijkl", Id, Id)[None]
    )
    H = H5.reshape(t, dim * dim, dim * dim)
    return H


# --------------------------------------------------------------------------- #
# Global explicit tier: position (x) variable                                 #
# --------------------------------------------------------------------------- #
def stvk_energy_x(X: np.ndarray, J: sp.sparse.spmatrix, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray) -> float:
    """Assembled StVK energy at positions ``X``.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Current vertex positions.
    J : scipy.sparse matrix (t*dim*dim, n*dim)
        Deformation Jacobian.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.

    Returns
    -------
    E : float
        Total StVK energy.
    """
    dim = X.shape[1]
    F = (J @ X.reshape(-1, 1)).reshape(-1, dim, dim)
    psi = stvk_energy_element_F(F, mu, lam)
    E = float((np.asarray(vol).reshape(-1, 1) * psi).sum())
    return E


def stvk_gradient_x(X: np.ndarray, J: sp.sparse.spmatrix, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Assembled StVK gradient w.r.t. positions ``X``.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Current vertex positions.
    J : scipy.sparse matrix (t*dim*dim, n*dim)
        Deformation Jacobian.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.

    Returns
    -------
    g : np.ndarray (n*dim, 1)
        Assembled energy gradient.
    """
    dim = X.shape[1]
    F = (J @ X.reshape(-1, 1)).reshape(-1, dim, dim)
    P = stvk_gradient_element_F(F, mu, lam)
    P = P * np.asarray(vol).reshape(-1, 1, 1)
    g = J.transpose() @ P.reshape(-1, 1)
    return g


def stvk_hessian_x(X: np.ndarray, J: sp.sparse.spmatrix, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray, psd: bool = True) -> sp.sparse.spmatrix:
    """Assembled StVK Hessian w.r.t. positions ``X``.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Current vertex positions.
    J : scipy.sparse matrix (t*dim*dim, n*dim)
        Deformation Jacobian.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.
    psd : bool, optional
        If ``True`` (default), project each per-element block to the nearest
        positive semi-definite matrix before assembly.

    Returns
    -------
    Q : scipy.sparse.csc_matrix (n*dim, n*dim)
        Assembled energy Hessian.
    """
    dim = X.shape[1]
    F = (J @ X.reshape(-1, 1)).reshape(-1, dim, dim)
    He = stvk_hessian_element_F(F, mu, lam)
    He = He * np.asarray(vol).reshape(-1, 1, 1)
    if psd:
        He = psd_project(He)
    H = sp.sparse.block_diag(He)
    Q = J.transpose() @ H @ J
    return Q


# --------------------------------------------------------------------------- #
# Global explicit tier: displacement (u) variable                             #
# --------------------------------------------------------------------------- #
def stvk_energy_u(u: np.ndarray, J: sp.sparse.spmatrix, Jx_bar: np.ndarray, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray) -> float:
    """Assembled StVK energy at displacement ``u`` from a reference ``x_bar``.

    Equivalent to :func:`stvk_energy_x` evaluated at ``x_bar + u`` but
    avoids recomputing ``J @ x_bar`` on every call.

    Parameters
    ----------
    u : np.ndarray (n, dim)
        Displacement from the reference configuration.
    J : scipy.sparse matrix (t*dim*dim, n*dim)
        Deformation Jacobian.
    Jx_bar : np.ndarray (t*dim*dim, 1)
        Precomputed ``J @ x_bar.reshape(-1, 1)``.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.

    Returns
    -------
    E : float
        Total StVK energy.
    """
    dim = u.shape[1]
    F = (J @ u.reshape(-1, 1) + Jx_bar).reshape(-1, dim, dim)
    psi = stvk_energy_element_F(F, mu, lam)
    E = float((np.asarray(vol).reshape(-1, 1) * psi).sum())
    return E


def stvk_gradient_u(u: np.ndarray, J: sp.sparse.spmatrix, Jx_bar: np.ndarray, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Assembled StVK gradient w.r.t. displacement ``u``."""
    dim = u.shape[1]
    F = (J @ u.reshape(-1, 1) + Jx_bar).reshape(-1, dim, dim)
    P = stvk_gradient_element_F(F, mu, lam)
    P = P * np.asarray(vol).reshape(-1, 1, 1)
    g = J.transpose() @ P.reshape(-1, 1)
    return g


def stvk_hessian_u(u: np.ndarray, J: sp.sparse.spmatrix, Jx_bar: np.ndarray, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray, psd: bool = True) -> sp.sparse.spmatrix:
    """Assembled StVK Hessian w.r.t. displacement ``u``."""
    dim = u.shape[1]
    F = (J @ u.reshape(-1, 1) + Jx_bar).reshape(-1, dim, dim)
    He = stvk_hessian_element_F(F, mu, lam)
    He = He * np.asarray(vol).reshape(-1, 1, 1)
    if psd:
        He = psd_project(He)
    H = sp.sparse.block_diag(He)
    Q = J.transpose() @ H @ J
    return Q


# --------------------------------------------------------------------------- #
# Self-contained tier: builds J and vol from rest geometry                    #
# --------------------------------------------------------------------------- #
def stvk_energy(X: np.ndarray, T: np.ndarray, mu: np.ndarray, lam: np.ndarray, U: Optional[np.ndarray] = None) -> float:
    """StVK energy, building the operator and weights from rest geometry."""
    if U is None:
        U = X
    J = deformation_jacobian(X, T)
    vol = volume(X, T)
    return stvk_energy_x(U, J, mu, lam, vol)


def stvk_gradient(X: np.ndarray, T: np.ndarray, mu: np.ndarray, lam: np.ndarray, U: Optional[np.ndarray] = None) -> np.ndarray:
    """StVK gradient, building the operator and weights from rest geometry."""
    if U is None:
        U = X
    J = deformation_jacobian(X, T)
    vol = volume(X, T)
    return stvk_gradient_x(U, J, mu, lam, vol)


def stvk_hessian(X: np.ndarray, T: np.ndarray, mu: np.ndarray, lam: np.ndarray, U: Optional[np.ndarray] = None, psd: bool = True) -> sp.sparse.spmatrix:
    """StVK Hessian, building the operator and weights from rest geometry."""
    if U is None:
        U = X
    J = deformation_jacobian(X, T)
    vol = volume(X, T)
    return stvk_hessian_x(U, J, mu, lam, vol, psd=psd)
