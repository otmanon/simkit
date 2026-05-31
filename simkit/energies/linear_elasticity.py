"""Linear elasticity energy.

Follows the standardized three-tier layout (see :mod:`simkit.energies.arap`
for the reference). Linear elasticity has only the deformation gradient
(``F``) representation, so there is no ``_S`` tier:

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
Linearized (small-strain) elasticity from the Sifakis course notes
(https://www.cs.toronto.edu/~jacobson/seminar/sifakis-course-notes-2012.pdf).
The density is
``psi = mu * ||eps||_F^2 + (lam/2) * tr(eps)^2`` with the small-strain tensor
``eps = (F + F^T)/2 - I``. Being quadratic in ``F``, the Hessian is constant
and positive semi-definite, so no PSD projection is needed; the ``psd`` flag is
accepted for interface consistency but has no effect.
"""

from typing import Optional

import numpy as np
import scipy as sp

from ..deformation_jacobian import deformation_jacobian
from ..volume import volume
from ..vectorized_trace import vectorized_trace
from ..vectorized_transpose import vectorized_transpose


# --------------------------------------------------------------------------- #
# Element tier: deformation gradient (F) representation                       #
# --------------------------------------------------------------------------- #
def linear_elasticity_energy_element_F(F: np.ndarray, mu: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Per-element linear elasticity energy density.

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
    """
    assert F.ndim == 3
    assert F.shape[1] == F.shape[2]
    mu = np.asarray(mu).reshape(-1, 1)
    lam = np.asarray(lam).reshape(-1, 1)

    eps = (F + F.transpose(0, 2, 1)) / 2 - np.identity(F.shape[1])[None, :, :]
    density = (
        mu.flatten() * np.sum(eps ** 2, axis=(1, 2))
        + (lam.flatten() / 2.0) * np.trace(eps, axis1=1, axis2=2) ** 2
    )
    return density.reshape(-1, 1)


def linear_elasticity_gradient_element_F(F: np.ndarray, mu: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Per-element first Piola-Kirchhoff stress (gradient w.r.t. ``F``).

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
    assert F.ndim == 3
    assert F.shape[1] == F.shape[2]
    mu = np.asarray(mu).reshape(-1, 1, 1)
    lam = np.asarray(lam).reshape(-1, 1, 1)

    I = np.identity(F.shape[1])[None, :, :]
    P = (
        mu * (F + F.transpose(0, 2, 1) - 2 * I)
        + lam * np.trace(F - I, axis1=1, axis2=2)[:, None, None] * I
    )
    return P


def linear_elasticity_hessian_element_F(F: np.ndarray, mu: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Per-element Hessian of the density w.r.t. ``F`` (vectorized blocks).

    Parameters
    ----------
    F : np.ndarray (t, dim, dim)
        Per-element deformation gradients. Only the shape is used; the linear
        elasticity Hessian is constant in ``F``.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.

    Returns
    -------
    H : np.ndarray (t, dim*dim, dim*dim)
        Per-element Hessian blocks in vectorized ``F`` layout. Constant in
        ``F`` and positive semi-definite. No quadrature weighting applied.
    """
    assert F.ndim == 3
    assert F.shape[1] == F.shape[2]
    mu = np.asarray(mu).reshape(-1, 1, 1)
    lam = np.asarray(lam).reshape(-1, 1, 1)

    dim = F.shape[1]
    I = np.identity(dim * dim)
    T = np.asarray(vectorized_transpose(1, dim).toarray())
    Tr = np.asarray(vectorized_trace(1, dim).toarray())

    H1 = np.repeat((I + T)[None, :, :], F.shape[0], axis=0)
    H2 = np.repeat((Tr.T @ Tr)[None, :, :], F.shape[0], axis=0)

    H = mu * H1 + lam * H2
    return H


# --------------------------------------------------------------------------- #
# Global explicit tier: position (x) variable                                 #
# --------------------------------------------------------------------------- #
def linear_elasticity_energy_x(X: np.ndarray, J: sp.sparse.spmatrix, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray) -> float:
    """Assembled linear elasticity energy at positions ``X``.

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
        Total linear elasticity energy.
    """
    dim = X.shape[1]
    F = (J @ X.reshape(-1, 1)).reshape(-1, dim, dim)
    psi = linear_elasticity_energy_element_F(F, mu, lam)
    E = float((np.asarray(vol).reshape(-1, 1) * psi).sum())
    return E


def linear_elasticity_gradient_x(X: np.ndarray, J: sp.sparse.spmatrix, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Assembled linear elasticity gradient w.r.t. positions ``X``.

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
    P = linear_elasticity_gradient_element_F(F, mu, lam)
    P = P * np.asarray(vol).reshape(-1, 1, 1)
    g = J.transpose() @ P.reshape(-1, 1)
    return g


def linear_elasticity_hessian_x(X: np.ndarray, J: sp.sparse.spmatrix, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray, psd: bool = True) -> sp.sparse.spmatrix:
    """Assembled linear elasticity Hessian w.r.t. positions ``X``.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Current vertex positions. Only the shape is used; the Hessian is
        constant in ``X``.
    J : scipy.sparse matrix (t*dim*dim, n*dim)
        Deformation Jacobian.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.
    psd : bool, optional
        Accepted for interface consistency. The linear elasticity Hessian is
        already positive semi-definite, so this flag has no effect.

    Returns
    -------
    Q : scipy.sparse.csc_matrix (n*dim, n*dim)
        Assembled energy Hessian.
    """
    dim = X.shape[1]
    F = (J @ X.reshape(-1, 1)).reshape(-1, dim, dim)
    He = linear_elasticity_hessian_element_F(F, mu, lam)
    He = He * np.asarray(vol).reshape(-1, 1, 1)
    H = sp.sparse.block_diag(He)
    Q = J.transpose() @ H @ J
    return Q


# --------------------------------------------------------------------------- #
# Global explicit tier: displacement (u) variable                             #
# --------------------------------------------------------------------------- #
def linear_elasticity_energy_u(u: np.ndarray, J: sp.sparse.spmatrix, Jx_bar: np.ndarray, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray) -> float:
    """Assembled linear elasticity energy at displacement ``u`` from a reference ``x_bar``.

    Equivalent to :func:`linear_elasticity_energy_x` evaluated at ``x_bar + u``
    but avoids recomputing ``J @ x_bar`` on every call. The reference ``x_bar``
    is arbitrary (not required to be the rest pose).

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
        Total linear elasticity energy.
    """
    dim = u.shape[1]
    F = (J @ u.reshape(-1, 1) + Jx_bar).reshape(-1, dim, dim)
    psi = linear_elasticity_energy_element_F(F, mu, lam)
    E = float((np.asarray(vol).reshape(-1, 1) * psi).sum())
    return E


def linear_elasticity_gradient_u(u: np.ndarray, J: sp.sparse.spmatrix, Jx_bar: np.ndarray, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Assembled linear elasticity gradient w.r.t. displacement ``u``.

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
    g : np.ndarray (n*dim, 1)
        Assembled energy gradient.
    """
    dim = u.shape[1]
    F = (J @ u.reshape(-1, 1) + Jx_bar).reshape(-1, dim, dim)
    P = linear_elasticity_gradient_element_F(F, mu, lam)
    P = P * np.asarray(vol).reshape(-1, 1, 1)
    g = J.transpose() @ P.reshape(-1, 1)
    return g


def linear_elasticity_hessian_u(u: np.ndarray, J: sp.sparse.spmatrix, Jx_bar: np.ndarray, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray, psd: bool = True) -> sp.sparse.spmatrix:
    """Assembled linear elasticity Hessian w.r.t. displacement ``u``.

    Parameters
    ----------
    u : np.ndarray (n, dim)
        Displacement from the reference configuration. Only the shape is used;
        the Hessian is constant in ``u``.
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
    psd : bool, optional
        Accepted for interface consistency; has no effect (Hessian is PSD).

    Returns
    -------
    Q : scipy.sparse.csc_matrix (n*dim, n*dim)
        Assembled energy Hessian.
    """
    dim = u.shape[1]
    F = (J @ u.reshape(-1, 1) + Jx_bar).reshape(-1, dim, dim)
    He = linear_elasticity_hessian_element_F(F, mu, lam)
    He = He * np.asarray(vol).reshape(-1, 1, 1)
    H = sp.sparse.block_diag(He)
    Q = J.transpose() @ H @ J
    return Q


# --------------------------------------------------------------------------- #
# Self-contained tier: builds J and vol from rest geometry                    #
# --------------------------------------------------------------------------- #
def linear_elasticity_energy(X: np.ndarray, T: np.ndarray, mu: np.ndarray, lam: np.ndarray, U: Optional[np.ndarray] = None) -> float:
    """Linear elasticity energy, building the operator and weights from rest geometry.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Rest vertex positions. Used to build ``J`` and ``vol``.
    T : np.ndarray (t, dim+1)
        Element connectivity.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.
    U : np.ndarray (n, dim), optional
        Current vertex positions. Defaults to ``X``.

    Returns
    -------
    E : float
        Total linear elasticity energy.
    """
    if U is None:
        U = X
    J = deformation_jacobian(X, T)
    vol = volume(X, T)
    return linear_elasticity_energy_x(U, J, mu, lam, vol)


def linear_elasticity_gradient(X: np.ndarray, T: np.ndarray, mu: np.ndarray, lam: np.ndarray, U: Optional[np.ndarray] = None) -> np.ndarray:
    """Linear elasticity gradient, building the operator and weights from rest geometry.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Rest vertex positions.
    T : np.ndarray (t, dim+1)
        Element connectivity.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.
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
    return linear_elasticity_gradient_x(U, J, mu, lam, vol)


def linear_elasticity_hessian(X: np.ndarray, T: np.ndarray, mu: np.ndarray, lam: np.ndarray, U: Optional[np.ndarray] = None, psd: bool = True) -> sp.sparse.spmatrix:
    """Linear elasticity Hessian, building the operator and weights from rest geometry.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Rest vertex positions.
    T : np.ndarray (t, dim+1)
        Element connectivity.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.
    U : np.ndarray (n, dim), optional
        Current vertex positions. Defaults to ``X``.
    psd : bool, optional
        Accepted for interface consistency; has no effect (Hessian is already
        PSD).

    Returns
    -------
    Q : scipy.sparse.csc_matrix (n*dim, n*dim)
        Assembled energy Hessian.
    """
    if U is None:
        U = X
    J = deformation_jacobian(X, T)
    vol = volume(X, T)
    return linear_elasticity_hessian_x(U, J, mu, lam, vol, psd=psd)