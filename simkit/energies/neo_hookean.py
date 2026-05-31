"""Classical Neo-Hookean elastic energy.

Implements the classical Neo-Hookean hyperelastic energy density in the
form taught in the FEM-deformables course notes by Sifakis and Barbic
(http://barbic.usc.edu/femdefo/):

    psi(F) = (mu / 2) * (I_C - dim)
           - mu * log(J)
           + (lam / 2) * log(J)**2

    I_C    = ||F||_F**2 = sum_{i,j} F_{ij}**2
    J      = det(F)

The closed-form first Piola-Kirchhoff stress and Hessian are

    P                = mu * F + (lam * log(J) - mu) * F^{-T}
    d2 psi / dF dF   = mu * delta_ik * delta_jl
                     + lam * F^{-T}_{ij} * F^{-T}_{kl}
                     + (mu - lam * log(J)) * F^{-T}_{il} * F^{-T}_{kj}

Note that the log term makes the energy diverge as ``J -> 0+`` and become
undefined (NaN) for ``J <= 0``. This is the classical "inversion is
unphysical" property -- and is exactly what the stable Neo-Hookean
formulations in :mod:`simkit.energies.macklin_mueller_neo_hookean` and
:mod:`simkit.energies.stable_neo_hookean` were designed to fix.

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
verified symbolically by ``scripts/derive_stvk_and_neo_hookean.py``. The
element-tier Hessian blocks are returned in row-major ``F`` layout
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
def neo_hookean_energy_element_F(F: np.ndarray, mu: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Per-element classical Neo-Hookean energy density.

    Parameters
    ----------
    F : np.ndarray (t, dim, dim)
        Per-element deformation gradients. Must have ``det(F) > 0`` for the
        energy to be defined.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.

    Returns
    -------
    psi : np.ndarray (t, 1)
        Per-element energy densities. No quadrature weighting applied.
        Zero at the rest state ``F = I``; diverges as ``det(F) -> 0+``;
        NaN where ``det(F) <= 0``.
    """
    dim = F.shape[-1]
    F = F.reshape(-1, dim, dim)
    mu = np.asarray(mu).reshape(-1, 1)
    lam = np.asarray(lam).reshape(-1, 1)

    if dim not in (2, 3):
        raise ValueError("Neo-Hookean supports dim=2 or dim=3")

    I_C = (F ** 2).sum(axis=(-1, -2)).reshape(-1, 1)
    J = np.linalg.det(F).reshape(-1, 1)
    log_J = np.log(J)

    psi = (mu / 2.0) * (I_C - dim) - mu * log_J + (lam / 2.0) * log_J ** 2
    return psi


def neo_hookean_gradient_element_F(F: np.ndarray, mu: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Per-element first Piola-Kirchhoff stress (gradient w.r.t. ``F``).

    Uses the closed form ``P = mu * F + (lam * log(J) - mu) * F^{-T}``.

    Parameters
    ----------
    F : np.ndarray (t, dim, dim)
        Per-element deformation gradients (must be invertible).
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
        raise ValueError("Neo-Hookean supports dim=2 or dim=3")

    mu_b = np.asarray(mu).reshape(-1, 1, 1)
    lam_b = np.asarray(lam).reshape(-1, 1, 1)

    J = np.linalg.det(F).reshape(-1, 1, 1)
    log_J = np.log(J)
    F_invT = np.linalg.inv(F).swapaxes(-1, -2)

    P = mu_b * F + (lam_b * log_J - mu_b) * F_invT
    return P


def neo_hookean_hessian_element_F(F: np.ndarray, mu: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Per-element Hessian of the density w.r.t. ``F`` (vectorized blocks).

    Implements the structural formula

        H[ij, kl] = mu * delta_ik * delta_jl
                  + lam * F^{-T}_{ij} * F^{-T}_{kl}
                  + (mu - lam * log(J)) * F^{-T}_{il} * F^{-T}_{kj}

    Parameters
    ----------
    F : np.ndarray (t, dim, dim)
        Per-element deformation gradients (must be invertible).
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
        raise ValueError("Neo-Hookean supports dim=2 or dim=3")

    t = F.shape[0]
    mu_b = np.asarray(mu).reshape(-1, 1, 1, 1, 1)
    lam_b = np.asarray(lam).reshape(-1, 1, 1, 1, 1)

    J = np.linalg.det(F).reshape(-1, 1, 1, 1, 1)
    log_J = np.log(J)
    F_invT = np.linalg.inv(F).swapaxes(-1, -2)
    Id = np.eye(dim)

    H5 = (
        mu_b * np.einsum("ik,jl->ijkl", Id, Id)[None]
        + lam_b * np.einsum("tij,tkl->tijkl", F_invT, F_invT)
        + (mu_b - lam_b * log_J) * np.einsum("til,tkj->tijkl", F_invT, F_invT)
    )
    H = H5.reshape(t, dim * dim, dim * dim)
    return H


# --------------------------------------------------------------------------- #
# Global explicit tier: position (x) variable                                 #
# --------------------------------------------------------------------------- #
def neo_hookean_energy_x(X: np.ndarray, J: sp.sparse.spmatrix, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray) -> float:
    """Assembled classical Neo-Hookean energy at positions ``X``.

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
        Total Neo-Hookean energy.
    """
    dim = X.shape[1]
    F = (J @ X.reshape(-1, 1)).reshape(-1, dim, dim)
    psi = neo_hookean_energy_element_F(F, mu, lam)
    E = float((np.asarray(vol).reshape(-1, 1) * psi).sum())
    return E


def neo_hookean_gradient_x(X: np.ndarray, J: sp.sparse.spmatrix, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Assembled classical Neo-Hookean gradient w.r.t. positions ``X``."""
    dim = X.shape[1]
    F = (J @ X.reshape(-1, 1)).reshape(-1, dim, dim)
    P = neo_hookean_gradient_element_F(F, mu, lam)
    P = P * np.asarray(vol).reshape(-1, 1, 1)
    g = J.transpose() @ P.reshape(-1, 1)
    return g


def neo_hookean_hessian_x(X: np.ndarray, J: sp.sparse.spmatrix, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray, psd: bool = True) -> sp.sparse.spmatrix:
    """Assembled classical Neo-Hookean Hessian w.r.t. positions ``X``."""
    dim = X.shape[1]
    F = (J @ X.reshape(-1, 1)).reshape(-1, dim, dim)
    He = neo_hookean_hessian_element_F(F, mu, lam)
    He = He * np.asarray(vol).reshape(-1, 1, 1)
    if psd:
        He = psd_project(He)
    H = sp.sparse.block_diag(He)
    Q = J.transpose() @ H @ J
    return Q


# --------------------------------------------------------------------------- #
# Global explicit tier: displacement (u) variable                             #
# --------------------------------------------------------------------------- #
def neo_hookean_energy_u(u: np.ndarray, J: sp.sparse.spmatrix, Jx_bar: np.ndarray, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray) -> float:
    """Assembled classical Neo-Hookean energy at displacement ``u`` from a reference ``x_bar``."""
    dim = u.shape[1]
    F = (J @ u.reshape(-1, 1) + Jx_bar).reshape(-1, dim, dim)
    psi = neo_hookean_energy_element_F(F, mu, lam)
    E = float((np.asarray(vol).reshape(-1, 1) * psi).sum())
    return E


def neo_hookean_gradient_u(u: np.ndarray, J: sp.sparse.spmatrix, Jx_bar: np.ndarray, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Assembled classical Neo-Hookean gradient w.r.t. displacement ``u``."""
    dim = u.shape[1]
    F = (J @ u.reshape(-1, 1) + Jx_bar).reshape(-1, dim, dim)
    P = neo_hookean_gradient_element_F(F, mu, lam)
    P = P * np.asarray(vol).reshape(-1, 1, 1)
    g = J.transpose() @ P.reshape(-1, 1)
    return g


def neo_hookean_hessian_u(u: np.ndarray, J: sp.sparse.spmatrix, Jx_bar: np.ndarray, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray, psd: bool = True) -> sp.sparse.spmatrix:
    """Assembled classical Neo-Hookean Hessian w.r.t. displacement ``u``."""
    dim = u.shape[1]
    F = (J @ u.reshape(-1, 1) + Jx_bar).reshape(-1, dim, dim)
    He = neo_hookean_hessian_element_F(F, mu, lam)
    He = He * np.asarray(vol).reshape(-1, 1, 1)
    if psd:
        He = psd_project(He)
    H = sp.sparse.block_diag(He)
    Q = J.transpose() @ H @ J
    return Q


# --------------------------------------------------------------------------- #
# Self-contained tier: builds J and vol from rest geometry                    #
# --------------------------------------------------------------------------- #
def neo_hookean_energy(X: np.ndarray, T: np.ndarray, mu: np.ndarray, lam: np.ndarray, U: Optional[np.ndarray] = None) -> float:
    """Classical Neo-Hookean energy, building the operator and weights from rest geometry."""
    if U is None:
        U = X
    J = deformation_jacobian(X, T)
    vol = volume(X, T)
    return neo_hookean_energy_x(U, J, mu, lam, vol)


def neo_hookean_gradient(X: np.ndarray, T: np.ndarray, mu: np.ndarray, lam: np.ndarray, U: Optional[np.ndarray] = None) -> np.ndarray:
    """Classical Neo-Hookean gradient, building the operator and weights from rest geometry."""
    if U is None:
        U = X
    J = deformation_jacobian(X, T)
    vol = volume(X, T)
    return neo_hookean_gradient_x(U, J, mu, lam, vol)


def neo_hookean_hessian(X: np.ndarray, T: np.ndarray, mu: np.ndarray, lam: np.ndarray, U: Optional[np.ndarray] = None, psd: bool = True) -> sp.sparse.spmatrix:
    """Classical Neo-Hookean Hessian, building the operator and weights from rest geometry."""
    if U is None:
        U = X
    J = deformation_jacobian(X, T)
    vol = volume(X, T)
    return neo_hookean_hessian_x(U, J, mu, lam, vol, psd=psd)
