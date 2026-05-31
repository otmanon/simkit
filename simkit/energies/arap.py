"""As-Rigid-As-Possible (ARAP) elastic energy.

Reference implementation for the standardized energy layout used throughout
``simkit.energies``. Every energy exposes three tiers, where the suffix tells
you how much you must supply yourself:

Element tier (``*_element_F`` / ``*_element_S``)
    Per-element densities and derivative blocks. Material parameters only:
    no quadrature weight ``vol``, no summation, no global operator. Inputs and
    outputs are stacked per-element arrays. This is the only tier that holds
    the material formula.

Global explicit tier (``*_x`` / ``*_S``)
    Takes a prebuilt operator (``J`` for ``x``; none for ``S``, which is its
    own variable) and the quadrature weights ``vol``. Calls the element tier,
    weights by ``vol``, and assembles. This is what a simulation loop calls
    every step, since ``J`` and ``vol`` are built once and reused.

Self-contained tier (no suffix)
    Builds ``J`` and ``vol`` from the rest geometry ``(X, T)`` and forwards to
    the explicit tier. The a-la-carte one-liner for demos and tests.

Notes
-----
ARAP density: :math:`\\psi(F) = \\tfrac{1}{2}\\,\\mu\\,\\lVert F - R\\rVert_F^2`,
where :math:`R` is the rotation from the polar decomposition of :math:`F`.
"""

from typing import Optional, Tuple

import numpy as np
import scipy as sp

from ..polar_svd import polar_svd
from ..rotation_gradient import rotation_gradient_F
from ..deformation_jacobian import deformation_jacobian
from ..volume import volume
from ..psd_project import psd_project


def _voigt_arap(k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Voigt weights and rest-state vector for the compact ``S`` representation.

    Parameters
    ----------
    k : int
        Length of the compact symmetric stretch vector. ``3`` for 2D,
        ``6`` for 3D.

    Returns
    -------
    w : np.ndarray (1, k)
        Off-diagonal doubling weights for the Frobenius inner product.
    i : np.ndarray (1, k)
        Rest-state stretch vector (identity in compact form).
    """
    if k == 3:
        w = np.array([1.0, 1.0, 2.0])[None, :]
        i = np.array([1.0, 1.0, 0.0])[None, :]
    elif k == 6:
        w = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])[None, :]
        i = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])[None, :]
    else:
        raise ValueError("Compact S must have k=3 (2D) or k=6 (3D)")
    return w, i


# --------------------------------------------------------------------------- #
# Element tier: deformation gradient (F) representation                       #
# --------------------------------------------------------------------------- #
def arap_energy_element_F(F: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Per-element ARAP energy density in terms of the deformation gradient.

    Parameters
    ----------
    F : np.ndarray (t, dim, dim)
        Per-element deformation gradients.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.

    Returns
    -------
    psi : np.ndarray (t, 1)
        Per-element energy densities. No quadrature weighting applied.
    """
    dim = F.shape[-1]
    F = F.reshape(-1, dim, dim)
    mu = np.asarray(mu).reshape(-1, 1)
    [R, S] = polar_svd(F)
    psi = 0.5 * mu * np.sum((F - R) ** 2, axis=(1, 2))[:, None]
    return psi


def arap_gradient_element_F(F: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Per-element first Piola-Kirchhoff stress (gradient w.r.t. ``F``).

    Parameters
    ----------
    F : np.ndarray (t, dim, dim)
        Per-element deformation gradients.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.

    Returns
    -------
    P : np.ndarray (t, dim, dim)
        Per-element PK1 stress blocks. No quadrature weighting applied.
    """
    dim = F.shape[-1]
    F = F.reshape(-1, dim, dim)
    mu = np.asarray(mu).reshape(-1, 1, 1)
    [R, S] = polar_svd(F)
    P = mu * (F - R)
    return P


def arap_hessian_element_F(F: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Per-element Hessian of the density w.r.t. ``F`` (vectorized blocks).

    Parameters
    ----------
    F : np.ndarray (t, dim, dim)
        Per-element deformation gradients.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.

    Returns
    -------
    H : np.ndarray (t, dim*dim, dim*dim)
        Per-element Hessian blocks in vectorized ``F`` layout. No quadrature
        weighting applied.

    Notes
    -----
    Uses ``H = mu * ( I -  dR/dF)``, carried over verbatim from the original
    implementation.
    """
    dim = F.shape[-1]
    F = F.reshape(-1, dim, dim)
    n = F.shape[0]
    mu = np.asarray(mu).reshape(-1, 1, 1)
    I = np.tile(np.identity(dim * dim), (n, 1, 1))
    H =  I - rotation_gradient_F(F)
    H = mu * H
    return H


# --------------------------------------------------------------------------- #
# Element tier: stretch (S) representation                                    #
# --------------------------------------------------------------------------- #
def arap_energy_element_S(S: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Per-element ARAP energy density in terms of the stretch ``S``.

    Parameters
    ----------
    S : np.ndarray (t, dim, dim) or (t, k)
        Per-element symmetric stretch, either as full matrices ``(t, dim, dim)``
        or as compact Voigt vectors ``(t, k)`` with ``k = 3`` (2D) or ``6`` (3D).
    mu : np.ndarray (t, 1)
        Per-element shear modulus.

    Returns
    -------
    psi : np.ndarray (t, 1)
        Per-element energy densities. No quadrature weighting applied.
    """
    assert S.ndim == 2 or S.ndim == 3
    mu = np.asarray(mu).reshape(-1, 1)
    if S.ndim == 3:
        dim = S.shape[-1]
        psi = S - np.eye(dim)[None, :, :]
        density = 0.5 * mu * np.sum(psi ** 2, axis=(1, 2))[:, None]
    else:
        w, i = _voigt_arap(S.shape[-1])
        psi = S - i
        density = 0.5 * mu * np.sum(psi ** 2 * w, axis=1)[:, None]
    return density


def arap_gradient_element_S(S: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Per-element gradient of the density w.r.t. the stretch ``S``.

    Parameters
    ----------
    S : np.ndarray (t, dim, dim) or (t, k)
        Per-element symmetric stretch (full or compact Voigt form).
    mu : np.ndarray (t, 1)
        Per-element shear modulus.

    Returns
    -------
    P : np.ndarray (t, dim, dim) or (t, k)
        Per-element gradient blocks, matching the input representation. No
        quadrature weighting applied.
    """
    assert S.ndim == 2 or S.ndim == 3
    if S.ndim == 3:
        dim = S.shape[-1]
        mu = np.asarray(mu).reshape(-1, 1, 1)
        P = mu * (S - np.eye(dim)[None, :, :])
    else:
        mu = np.asarray(mu).reshape(-1, 1)
        w, i = _voigt_arap(S.shape[-1])
        P = mu * ((S - i) * w)
    return P


def arap_hessian_element_S(S: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Per-element Hessian of the density w.r.t. the stretch ``S``.

    Parameters
    ----------
    S : np.ndarray (t, dim, dim) or (t, k)
        Per-element symmetric stretch (full or compact Voigt form).
    mu : np.ndarray (t, 1)
        Per-element shear modulus.

    Returns
    -------
    H : np.ndarray (t, b, b)
        Per-element Hessian blocks, where ``b = dim*dim`` for the full
        representation and ``b = k`` for the compact one. No quadrature
        weighting applied.
    """
    assert S.ndim == 2 or S.ndim == 3
    mu = np.asarray(mu).reshape(-1, 1, 1)
    if S.ndim == 3:
        t = S.shape[0]
        dim = S.shape[-1]
        H = mu * np.tile(np.identity(dim * dim), (t, 1, 1))
    else:
        k = S.shape[-1]
        w, _ = _voigt_arap(k)
        diag = w[0] * np.eye(k)[None, :, :]
        H = mu * diag
    return H


# --------------------------------------------------------------------------- #
# Global explicit tier: position (x) variable                                 #
# --------------------------------------------------------------------------- #
def arap_energy_x(X: np.ndarray, J: sp.sparse.spmatrix, mu: np.ndarray, vol: np.ndarray) -> float:
    """Assembled ARAP energy at positions ``X`` given a prebuilt operator.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Current vertex positions.
    J : scipy.sparse matrix (t*dim*dim, n*dim)
        Deformation Jacobian mapping flattened positions to flattened ``F``.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights (rest volumes).

    Returns
    -------
    E : float
        Total ARAP energy.
    """
    dim = X.shape[1]
    F = (J @ X.reshape(-1, 1)).reshape(-1, dim, dim)
    psi = arap_energy_element_F(F, mu)
    E = float((np.asarray(vol).reshape(-1, 1) * psi).sum())
    return E


def arap_gradient_x(X: np.ndarray, J: sp.sparse.spmatrix, mu: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Assembled ARAP gradient w.r.t. positions ``X``.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Current vertex positions.
    J : scipy.sparse matrix (t*dim*dim, n*dim)
        Deformation Jacobian.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.

    Returns
    -------
    g : np.ndarray (n*dim, 1)
        Assembled energy gradient.
    """
    dim = X.shape[1]
    F = (J @ X.reshape(-1, 1)).reshape(-1, dim, dim)
    P = arap_gradient_element_F(F, mu)
    P = P * np.asarray(vol).reshape(-1, 1, 1)
    g = J.transpose() @ P.reshape(-1, 1)
    return g


def arap_hessian_x(X: np.ndarray, J: sp.sparse.spmatrix, mu: np.ndarray, vol: np.ndarray, psd: bool = True) -> sp.sparse.spmatrix:
    """Assembled ARAP Hessian w.r.t. positions ``X``.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Current vertex positions.
    J : scipy.sparse matrix (t*dim*dim, n*dim)
        Deformation Jacobian.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
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
    He = arap_hessian_element_F(F, mu)
    He = He * np.asarray(vol).reshape(-1, 1, 1)
    if psd:
        He = psd_project(He)
    H = sp.sparse.block_diag(He)
    Q = J.transpose() @ H @ J
    return Q


# --------------------------------------------------------------------------- #
# Global explicit tier: displacement (u) variable                             #
# --------------------------------------------------------------------------- #
def arap_energy_u(u: np.ndarray, J: sp.sparse.spmatrix, Jx_bar: np.ndarray, mu: np.ndarray, vol: np.ndarray) -> float:
    """Assembled ARAP energy at displacement ``u`` from a reference ``x_bar``.

    Equivalent to :func:`arap_energy_x` evaluated at ``x_bar + u`` but avoids
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
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.

    Returns
    -------
    E : float
        Total ARAP energy.
    """
    dim = u.shape[1]
    F = (J @ u.reshape(-1, 1) + Jx_bar).reshape(-1, dim, dim)
    psi = arap_energy_element_F(F, mu)
    E = float((np.asarray(vol).reshape(-1, 1) * psi).sum())
    return E


def arap_gradient_u(u: np.ndarray, J: sp.sparse.spmatrix, Jx_bar: np.ndarray, mu: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Assembled ARAP gradient w.r.t. displacement ``u``.

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
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.

    Returns
    -------
    g : np.ndarray (n*dim, 1)
        Assembled energy gradient.
    """
    dim = u.shape[1]
    F = (J @ u.reshape(-1, 1) + Jx_bar).reshape(-1, dim, dim)
    P = arap_gradient_element_F(F, mu)
    P = P * np.asarray(vol).reshape(-1, 1, 1)
    g = J.transpose() @ P.reshape(-1, 1)
    return g


def arap_hessian_u(u: np.ndarray, J: sp.sparse.spmatrix, Jx_bar: np.ndarray, mu: np.ndarray, vol: np.ndarray, psd: bool = True) -> sp.sparse.spmatrix:
    """Assembled ARAP Hessian w.r.t. displacement ``u``.

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
    dim = u.shape[1]
    F = (J @ u.reshape(-1, 1) + Jx_bar).reshape(-1, dim, dim)
    He = arap_hessian_element_F(F, mu)
    He = He * np.asarray(vol).reshape(-1, 1, 1)
    if psd:
        He = psd_project(He)
    H = sp.sparse.block_diag(He)
    Q = J.transpose() @ H @ J
    return Q


# --------------------------------------------------------------------------- #
# Global explicit tier: stretch (S) variable                                  #
# --------------------------------------------------------------------------- #
def arap_energy_S(S: np.ndarray, mu: np.ndarray, vol: np.ndarray) -> float:
    """Assembled ARAP energy in the stretch variable ``S``.

    Parameters
    ----------
    S : np.ndarray (t, dim, dim) or (t, k)
        Per-element stretch (full or compact Voigt form).
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.

    Returns
    -------
    E : float
        Total ARAP energy.
    """
    psi = arap_energy_element_S(S, mu)
    E = float((np.asarray(vol).reshape(-1, 1) * psi).sum())
    return E


def arap_gradient_S(S: np.ndarray, mu: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Assembled ARAP gradient in the stretch variable ``S``.

    Parameters
    ----------
    S : np.ndarray (t, dim, dim) or (t, k)
        Per-element stretch (full or compact Voigt form).
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.

    Returns
    -------
    g : np.ndarray (t*b, 1)
        Flattened, volume-weighted gradient, where ``b = dim*dim`` (full) or
        ``b = k`` (compact).
    """
    P = arap_gradient_element_S(S, mu)
    if S.ndim == 3:
        P = P * np.asarray(vol).reshape(-1, 1, 1)
    else:
        P = P * np.asarray(vol).reshape(-1, 1)
    return P.reshape(-1, 1)


def arap_hessian_S(S: np.ndarray, mu: np.ndarray, vol: np.ndarray) -> sp.sparse.spmatrix:
    """Assembled ARAP Hessian in the stretch variable ``S``.

    Parameters
    ----------
    S : np.ndarray (t, dim, dim) or (t, k)
        Per-element stretch (full or compact Voigt form).
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.

    Returns
    -------
    H : scipy.sparse matrix (t*b, t*b)
        Block-diagonal, volume-weighted Hessian. ``S`` is its own variable, so
        no deformation operator is applied.
    """
    He = arap_hessian_element_S(S, mu)
    He = He * np.asarray(vol).reshape(-1, 1, 1)
    H = sp.sparse.block_diag(He)
    return H


# --------------------------------------------------------------------------- #
# Self-contained tier: builds J and vol from rest geometry                    #
# --------------------------------------------------------------------------- #
def arap_energy(X: np.ndarray, T: np.ndarray, mu: np.ndarray, U: Optional[np.ndarray] = None) -> float:
    """ARAP energy, building the operator and weights from rest geometry.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Rest vertex positions. Used to build ``J`` and ``vol``.
    T : np.ndarray (t, dim+1)
        Element connectivity (simplices).
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    U : np.ndarray (n, dim), optional
        Current (deformed) vertex positions at which to evaluate the energy.
        Defaults to the rest positions ``X`` (energy zero).

    Returns
    -------
    E : float
        Total ARAP energy.
    """
    if U is None:
        U = X
    J = deformation_jacobian(X, T)
    vol = volume(X, T)
    return arap_energy_x(U, J, mu, vol)


def arap_gradient(X: np.ndarray, T: np.ndarray, mu: np.ndarray, U: Optional[np.ndarray] = None) -> np.ndarray:
    """ARAP gradient, building the operator and weights from rest geometry.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Rest vertex positions.
    T : np.ndarray (t, dim+1)
        Element connectivity.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
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
    return arap_gradient_x(U, J, mu, vol)


def arap_hessian(X: np.ndarray, T: np.ndarray, mu: np.ndarray, U: Optional[np.ndarray] = None, psd: bool = True) -> sp.sparse.spmatrix:
    """ARAP Hessian, building the operator and weights from rest geometry.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Rest vertex positions.
    T : np.ndarray (t, dim+1)
        Element connectivity.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    U : np.ndarray (n, dim), optional
        Current vertex positions. Defaults to ``X``.
    psd : bool, optional
        Project per-element blocks to PSD before assembly. Default ``True``.

    Returns
    -------
    Q : scipy.sparse.csc_matrix (n*dim, n*dim)
        Assembled energy Hessian.
    """
    if U is None:
        U = X
    J = deformation_jacobian(X, T)
    vol = volume(X, T)
    return arap_hessian_x(U, J, mu, vol, psd=psd)