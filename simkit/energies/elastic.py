"""Elastic material energy, gradient, and Hessian (combined).

Material dispatch over the per-material modules (``arap``, ``fcr``,
``linear_elasticity``, ``macklin_mueller_neo_hookean``), selected by the
``material`` string (one of ``'linear-elasticity'``, ``'arap'``, ``'fcr'``,
``'macklin-mueller-neo-hookean'``).

This file combines what were previously ``elastic_energy.py``,
``elastic_gradient.py`` and ``elastic_hessian.py`` into one module. The three
families share the standardized tiers:

Element tier (``*_element_F``)
    Per-element density / stress / Hessian via the chosen material. No ``vol``
    weighting.

Global explicit tiers (``*_x``, ``*_z``)
    Apply ``vol`` and the operator (``J`` or the reduced ``JB``) during assembly.

Self-contained tier (no suffix)
    Build ``J`` and ``vol`` from rest geometry ``(X, T)``.

ARAP takes only ``mu`` (no ``lam``); the dispatchers handle that signature
difference internally so callers pass ``lam`` uniformly. The reduced-coordinate
precompute classes ``ElasticEnergyZPrecomp`` and ``ElasticEnergyZFilteredPrecomp``
live here too.
"""

from typing import Optional

import numpy as np
import scipy as sp

from .arap import (
    arap_energy_element_F,
    arap_energy_element_S,
    arap_energy_u,
    arap_gradient_element_F,
    arap_gradient_element_S,
    arap_gradient_u,
    arap_hessian_element_F,
    arap_hessian_element_S,
    arap_hessian_u,
)
from .fcr import (
    fcr_energy_element_F,
    fcr_energy_u,
    fcr_gradient_element_F,
    fcr_gradient_u,
    fcr_hessian_element_F,
    fcr_hessian_u,
)
from .linear_elasticity import (
    linear_elasticity_energy_element_F,
    linear_elasticity_energy_u,
    linear_elasticity_gradient_element_F,
    linear_elasticity_gradient_u,
    linear_elasticity_hessian_element_F,
    linear_elasticity_hessian_u,
)
from .macklin_mueller_neo_hookean import (
    macklin_mueller_neo_hookean_energy_element_F,
    macklin_mueller_neo_hookean_energy_u,
    macklin_mueller_neo_hookean_gradient_element_F,
    macklin_mueller_neo_hookean_gradient_u,
    macklin_mueller_neo_hookean_hessian_element_F,
    macklin_mueller_neo_hookean_hessian_u,
)
from ..deformation_jacobian import deformation_jacobian
from ..volume import volume
from ..psd_project import psd_project

_MATERIALS = ('linear-elasticity', 'arap', 'fcr', 'macklin-mueller-neo-hookean')



# =========================================================================== #
# ENERGY
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Element tier                                                                #
# --------------------------------------------------------------------------- #
def elastic_energy_element_F(F: np.ndarray, mu: np.ndarray, lam: np.ndarray, material: str) -> np.ndarray:
    """Per-element elastic energy density for the chosen material.

    Parameters
    ----------
    F : np.ndarray (t, dim, dim)
        Per-element deformation gradients.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter (ignored for ``'arap'``).
    material : str
        One of ``'linear-elasticity'``, ``'arap'``, ``'fcr'``, ``'macklin-mueller-neo-hookean'``.

    Returns
    -------
    psi : np.ndarray (t, 1)
        Per-element energy densities. No quadrature weighting applied.
    """
    if material == 'linear-elasticity':
        return linear_elasticity_energy_element_F(F, mu, lam)
    elif material == 'arap':
        return arap_energy_element_F(F, mu)
    elif material == 'fcr':
        return fcr_energy_element_F(F, mu, lam)
    elif material == 'macklin-mueller-neo-hookean':
        return macklin_mueller_neo_hookean_energy_element_F(F, mu, lam)
    raise ValueError("Unknown material type: " + str(material))


# --------------------------------------------------------------------------- #
# Global explicit tier: position (x) variable                                 #
# --------------------------------------------------------------------------- #
def elastic_energy_x(X: np.ndarray, J: sp.sparse.spmatrix, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray, material: str) -> float:
    """Assembled elastic energy at positions ``X`` for the chosen material.

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
    material : str
        Material identifier.

    Returns
    -------
    e : float
        Total elastic energy.
    """
    dim = X.shape[1]
    F = (J @ X.reshape(-1, 1)).reshape(-1, dim, dim)
    psi = elastic_energy_element_F(F, mu, lam, material)
    return float((np.asarray(vol).reshape(-1, 1) * psi).sum())


# --------------------------------------------------------------------------- #
# Global explicit tier: displacement (u) variable                             #
# --------------------------------------------------------------------------- #
def elastic_energy_u(u: np.ndarray, J: sp.sparse.spmatrix, Jx_bar: np.ndarray, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray, material: str) -> float:
    """Assembled elastic energy at displacement ``u`` from a reference ``x_bar``.

    Dispatches to the per-material ``*_energy_u`` function. ``Jx_bar`` is the
    precomputed flattened deformation gradient at the (arbitrary) reference
    configuration, ``J @ x_bar.reshape(-1, 1)``.

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
        Per-element first Lame parameter (ignored for ``'arap'``).
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.
    material : str
        Material identifier.

    Returns
    -------
    e : float
        Total elastic energy.
    """
    if material == 'linear-elasticity':
        return linear_elasticity_energy_u(u, J, Jx_bar, mu, lam, vol)
    elif material == 'arap':
        return arap_energy_u(u, J, Jx_bar, mu, vol)
    elif material == 'fcr':
        return fcr_energy_u(u, J, Jx_bar, mu, lam, vol)
    elif material == 'macklin-mueller-neo-hookean':
        return macklin_mueller_neo_hookean_energy_u(u, J, Jx_bar, mu, lam, vol)
    raise ValueError("Unknown material type: " + str(material))


class ElasticEnergyZPrecomp:
    """Precompute for the reduced elastic energy (``x = B z``).

    Parameters
    ----------
    B : scipy.sparse matrix (n*dim, r)
        Reduced basis.
    x0 : np.ndarray (n*dim, 1) or None
        Constant offset; defaults to zero.
    G : scipy.sparse matrix
        Selection / projection operator applied to ``J``.
    J : scipy.sparse matrix
        Deformation Jacobian.
    dim : int
        Spatial dimension.

    Attributes
    ----------
    JB : scipy.sparse matrix
        Reduced deformation operator ``G J B``.
    Jx0 : np.ndarray
        Reduced constant offset ``G J x0``.
    dim : int
        Spatial dimension.
    """

    def __init__(self, B: sp.sparse.spmatrix, x0: Optional[np.ndarray], G: sp.sparse.spmatrix, J: sp.sparse.spmatrix, dim: int):
        self.JB = G @ J @ B
        self.dim = dim
        if x0 is None:
            x0 = np.zeros((B.shape[0], 1))
        self.Jx0 = G @ J @ x0


class ElasticEnergyZFilteredPrecomp(ElasticEnergyZPrecomp):
    """Reduced elastic precompute with an added quadratic (filter) term.

    Parameters
    ----------
    B : scipy.sparse matrix (n*dim, r)
        Reduced basis.
    x0 : np.ndarray (n*dim, 1) or None
        Constant offset.
    G : scipy.sparse matrix
        Selection / projection operator.
    J : scipy.sparse matrix
        Deformation Jacobian.
    dim : int
        Spatial dimension.
    mu : np.ndarray (t,)
        Per-element shear modulus.
    vol : np.ndarray (t,)
        Per-element quadrature weights.

    Attributes
    ----------
    BJAMuJB : scipy.sparse matrix (r, r)
        Reduced filter matrix.
    BJAMuJx0 : np.ndarray (r, 1)
        Reduced filter offset.
    """

    def __init__(self, B: sp.sparse.spmatrix, x0: Optional[np.ndarray], G: sp.sparse.spmatrix, J: sp.sparse.spmatrix, dim: int, mu: np.ndarray, vol: np.ndarray):
        super().__init__(B, x0, G, J, dim)
        if x0 is None:
            x0 = np.zeros((B.shape[0], 1))
        AMu = sp.sparse.diags(np.asarray(mu).flatten() * np.asarray(vol).flatten())
        AMue = sp.sparse.kron(AMu, sp.sparse.eye(dim * dim))
        JAMuJ = J.T @ AMue @ J
        BJAMuJ = B.T @ JAMuJ
        self.BJAMuJB = BJAMuJ @ B
        self.BJAMuJx0 = BJAMuJ @ x0


def elastic_energy_z(z: np.ndarray, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray, material: str, precomp: ElasticEnergyZPrecomp, F: Optional[np.ndarray] = None) -> float:
    """Assembled reduced elastic energy.

    Parameters
    ----------
    z : np.ndarray (r, 1)
        Reduced coordinates.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.
    material : str
        Material identifier.
    precomp : ElasticEnergyZPrecomp
        Reduced operator precompute.
    F : np.ndarray (t, dim, dim), optional
        Precomputed deformation gradients; rebuilt from ``z`` if omitted.

    Returns
    -------
    e : float
        Total reduced elastic energy.
    """
    if F is None:
        dim = precomp.dim
        F = (precomp.JB @ z + precomp.Jx0).reshape(-1, dim, dim)
    psi = elastic_energy_element_F(F, mu, lam, material)
    return float((np.asarray(vol).reshape(-1, 1) * psi).sum())


def elastic_energy_filtered_z(z: np.ndarray, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray, material: str, precomp: ElasticEnergyZFilteredPrecomp, F: Optional[np.ndarray] = None) -> float:
    """Assembled reduced elastic energy plus the quadratic filter term.

    Parameters
    ----------
    z : np.ndarray (r, 1)
        Reduced coordinates.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.
    material : str
        Material identifier.
    precomp : ElasticEnergyZFilteredPrecomp
        Filtered reduced precompute.
    F : np.ndarray (t, dim, dim), optional
        Precomputed deformation gradients; rebuilt from ``z`` if omitted.

    Returns
    -------
    e : float
        Total reduced elastic energy including the filter term.
    """
    e = elastic_energy_z(z, mu, lam, vol, material, precomp, F=F)
    e += float(0.5 * z.T @ precomp.BJAMuJB @ z + z.T @ precomp.BJAMuJx0)
    return e


def elastic_energy_S(S: np.ndarray, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray, material: str) -> float:
    """Assembled elastic energy in the stretch ``S`` variable (ARAP only).

    Parameters
    ----------
    S : np.ndarray
        Per-element stretch (full ``(t, dim, dim)`` or compact Voigt form).
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Unused; present for interface consistency.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.
    material : str
        Material identifier; only ``'arap'`` is supported.

    Returns
    -------
    e : float
        Total elastic energy.
    """
    if material == 'arap':
        psi = arap_energy_element_S(S, mu)
        return float((np.asarray(vol).reshape(-1, 1) * psi).sum())
    raise ValueError("Unknown or unsupported material type for S: " + str(material))


# --------------------------------------------------------------------------- #
# Self-contained tier: builds J and vol from rest geometry                    #
# --------------------------------------------------------------------------- #
def elastic_energy(X: np.ndarray, T: np.ndarray, mu: np.ndarray, lam: np.ndarray, material: str, U: Optional[np.ndarray] = None) -> float:
    """Elastic energy, building the operator and weights from rest geometry.

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
    material : str
        Material identifier.
    U : np.ndarray (n, dim), optional
        Current vertex positions. Defaults to ``X``.

    Returns
    -------
    e : float
        Total elastic energy.
    """
    if U is None:
        U = X
    J = deformation_jacobian(X, T)
    vol = volume(X, T)
    return elastic_energy_x(U, J, mu, lam, vol, material)

# =========================================================================== #
# GRADIENT
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Element tier                                                                #
# --------------------------------------------------------------------------- #
def elastic_gradient_element_F(F: np.ndarray, mu: np.ndarray, lam: np.ndarray, material: str) -> np.ndarray:
    """Per-element first Piola-Kirchhoff stress for the chosen material.

    Parameters
    ----------
    F : np.ndarray (t, dim, dim)
        Per-element deformation gradients.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter (ignored for ``'arap'``).
    material : str
        Material identifier.

    Returns
    -------
    P : np.ndarray (t, dim, dim)
        Per-element stress blocks. No quadrature weighting applied.
    """
    if material == 'linear-elasticity':
        return linear_elasticity_gradient_element_F(F, mu, lam)
    elif material == 'arap':
        return arap_gradient_element_F(F, mu)
    elif material == 'fcr':
        return fcr_gradient_element_F(F, mu, lam)
    elif material == 'macklin-mueller-neo-hookean':
        return macklin_mueller_neo_hookean_gradient_element_F(F, mu, lam)
    raise ValueError("Unknown material type: " + str(material))


# --------------------------------------------------------------------------- #
# Global explicit tier: position (x) variable                                 #
# --------------------------------------------------------------------------- #
def elastic_gradient_x(X: np.ndarray, J: sp.sparse.spmatrix, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray, material: str) -> np.ndarray:
    """Assembled elastic gradient at positions ``X``.

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
    material : str
        Material identifier.

    Returns
    -------
    g : np.ndarray (n*dim, 1)
        Assembled gradient.
    """
    dim = X.shape[1]
    F = (J @ X.reshape(-1, 1)).reshape(-1, dim, dim)
    P = elastic_gradient_element_F(F, mu, lam, material)
    P = P * np.asarray(vol).reshape(-1, 1, 1)
    return J.transpose() @ P.reshape(-1, 1)


# --------------------------------------------------------------------------- #
# Global explicit tier: displacement (u) variable                             #
# --------------------------------------------------------------------------- #
def elastic_gradient_u(u: np.ndarray, J: sp.sparse.spmatrix, Jx_bar: np.ndarray, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray, material: str) -> np.ndarray:
    """Assembled elastic gradient at displacement ``u`` from a reference ``x_bar``.

    Dispatches to the per-material ``*_gradient_u`` function.

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
        Per-element first Lame parameter (ignored for ``'arap'``).
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.
    material : str
        Material identifier.

    Returns
    -------
    g : np.ndarray (n*dim, 1)
        Assembled gradient.
    """
    if material == 'linear-elasticity':
        return linear_elasticity_gradient_u(u, J, Jx_bar, mu, lam, vol)
    elif material == 'arap':
        return arap_gradient_u(u, J, Jx_bar, mu, vol)
    elif material == 'fcr':
        return fcr_gradient_u(u, J, Jx_bar, mu, lam, vol)
    elif material == 'macklin-mueller-neo-hookean':
        return macklin_mueller_neo_hookean_gradient_u(u, J, Jx_bar, mu, lam, vol)
    raise ValueError("Unknown material type: " + str(material))


def elastic_gradient_z(z: np.ndarray, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray, material: str, precomp: ElasticEnergyZPrecomp, F: Optional[np.ndarray] = None) -> np.ndarray:
    """Assembled reduced elastic gradient.

    Parameters
    ----------
    z : np.ndarray (r, 1)
        Reduced coordinates.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.
    material : str
        Material identifier.
    precomp : ElasticEnergyZPrecomp
        Reduced operator precompute.
    F : np.ndarray (t, dim, dim), optional
        Precomputed deformation gradients; rebuilt from ``z`` if omitted.

    Returns
    -------
    g : np.ndarray (r, 1)
        Assembled reduced gradient.
    """
    if F is None:
        dim = precomp.dim
        F = (precomp.JB @ z + precomp.Jx0).reshape(-1, dim, dim)
    P = elastic_gradient_element_F(F, mu, lam, material)
    P = P * np.asarray(vol).reshape(-1, 1, 1)
    return precomp.JB.transpose() @ P.reshape(-1, 1)


def elastic_gradient_filtered_z(z: np.ndarray, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray, material: str, precomp: ElasticEnergyZFilteredPrecomp, F: Optional[np.ndarray] = None) -> np.ndarray:
    """Assembled reduced elastic gradient plus the quadratic filter term.

    Parameters
    ----------
    z : np.ndarray (r, 1)
        Reduced coordinates.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.
    material : str
        Material identifier.
    precomp : ElasticEnergyZFilteredPrecomp
        Filtered reduced precompute.
    F : np.ndarray (t, dim, dim), optional
        Precomputed deformation gradients; rebuilt from ``z`` if omitted.

    Returns
    -------
    g : np.ndarray (r, 1)
        Assembled reduced gradient including the filter term.
    """
    g = elastic_gradient_z(z, mu, lam, vol, material, precomp, F=F)
    precomp_g = precomp.BJAMuJx0 + precomp.BJAMuJB @ z
    return g + precomp_g


def elastic_gradient_S(S: np.ndarray, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray, material: str) -> np.ndarray:
    """Per-element elastic gradient in the stretch ``S`` variable (ARAP only).

    Parameters
    ----------
    S : np.ndarray
        Per-element stretch (full or compact Voigt form).
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Unused; present for interface consistency.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.
    material : str
        Material identifier; only ``'arap'`` is supported.

    Returns
    -------
    g : np.ndarray
        Per-element weighted gradient w.r.t. ``S``.
    """
    if material == 'arap':
        g = arap_gradient_element_S(S, mu)
        return g * np.asarray(vol).reshape((-1,) + (1,) * (g.ndim - 1))
    raise ValueError("Unknown or unsupported material type for S: " + str(material))


# --------------------------------------------------------------------------- #
# Self-contained tier: builds J and vol from rest geometry                    #
# --------------------------------------------------------------------------- #
def elastic_gradient(X: np.ndarray, T: np.ndarray, mu: np.ndarray, lam: np.ndarray, material: str, U: Optional[np.ndarray] = None) -> np.ndarray:
    """Elastic gradient, building the operator and weights from rest geometry.

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
    material : str
        Material identifier.
    U : np.ndarray (n, dim), optional
        Current vertex positions. Defaults to ``X``.

    Returns
    -------
    g : np.ndarray (n*dim, 1)
        Assembled gradient.
    """
    if U is None:
        U = X
    J = deformation_jacobian(X, T)
    vol = volume(X, T)
    return elastic_gradient_x(U, J, mu, lam, vol, material)

# =========================================================================== #
# HESSIAN
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Element tier                                                                #
# --------------------------------------------------------------------------- #
def elastic_hessian_element_F(F: np.ndarray, mu: np.ndarray, lam: np.ndarray, material: str, psd: bool = True) -> np.ndarray:
    """Per-element elastic Hessian blocks for the chosen material.

    Parameters
    ----------
    F : np.ndarray (t, dim, dim)
        Per-element deformation gradients.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter (ignored for ``'arap'``).
    material : str
        Material identifier.
    psd : bool, optional
        If ``True`` (default), project each per-element block to PSD.

    Returns
    -------
    Q : np.ndarray (t, dim*dim, dim*dim)
        Per-element Hessian blocks. No quadrature weighting applied.
    """
    if material == 'linear-elasticity':
        Q = linear_elasticity_hessian_element_F(F, mu, lam)
    elif material == 'arap':
        Q = arap_hessian_element_F(F, mu)
    elif material == 'fcr':
        Q = fcr_hessian_element_F(F, mu, lam)
    elif material == 'macklin-mueller-neo-hookean':
        Q = macklin_mueller_neo_hookean_hessian_element_F(F, mu, lam)
    else:
        raise ValueError("Unknown material type: " + str(material))
    if psd:
        Q = psd_project(Q)
    return Q


# --------------------------------------------------------------------------- #
# Global explicit tier: position (x) variable                                 #
# --------------------------------------------------------------------------- #
def elastic_hessian_x(X: np.ndarray, J: sp.sparse.spmatrix, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray, material: str, psd: bool = True) -> sp.sparse.spmatrix:
    """Assembled elastic Hessian at positions ``X``.

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
    material : str
        Material identifier.
    psd : bool, optional
        If ``True`` (default), project each per-element block to PSD before
        assembly.

    Returns
    -------
    Q : scipy.sparse.csc_matrix (n*dim, n*dim)
        Assembled Hessian.
    """
    dim = X.shape[1]
    F = (J @ X.reshape(-1, 1)).reshape(-1, dim, dim)
    He = elastic_hessian_element_F(F, mu, lam, material, psd=psd)
    He = He * np.asarray(vol).reshape(-1, 1, 1)
    H = sp.sparse.block_diag(He)
    return J.transpose() @ H @ J


# --------------------------------------------------------------------------- #
# Global explicit tier: displacement (u) variable                             #
# --------------------------------------------------------------------------- #
def elastic_hessian_u(u: np.ndarray, J: sp.sparse.spmatrix, Jx_bar: np.ndarray, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray, material: str, psd: bool = True) -> sp.sparse.spmatrix:
    """Assembled elastic Hessian at displacement ``u`` from a reference ``x_bar``.

    Dispatches to the per-material ``*_hessian_u`` function.

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
        Per-element first Lame parameter (ignored for ``'arap'``).
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.
    material : str
        Material identifier.
    psd : bool, optional
        If ``True`` (default), project each per-element block to PSD before
        assembly.

    Returns
    -------
    Q : scipy.sparse.csc_matrix (n*dim, n*dim)
        Assembled Hessian.
    """
    if material == 'linear-elasticity':
        return linear_elasticity_hessian_u(u, J, Jx_bar, mu, lam, vol, psd=psd)
    elif material == 'arap':
        return arap_hessian_u(u, J, Jx_bar, mu, vol, psd=psd)
    elif material == 'fcr':
        return fcr_hessian_u(u, J, Jx_bar, mu, lam, vol, psd=psd)
    elif material == 'macklin-mueller-neo-hookean':
        return macklin_mueller_neo_hookean_hessian_u(u, J, Jx_bar, mu, lam, vol, psd=psd)
    raise ValueError("Unknown material type: " + str(material))


def elastic_hessian_z(z: np.ndarray, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray, material: str, precomp: ElasticEnergyZPrecomp, F: Optional[np.ndarray] = None, psd: bool = True) -> sp.sparse.spmatrix:
    """Assembled reduced elastic Hessian.

    Parameters
    ----------
    z : np.ndarray (r, 1)
        Reduced coordinates.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.
    material : str
        Material identifier.
    precomp : ElasticEnergyZPrecomp
        Reduced operator precompute.
    F : np.ndarray (t, dim, dim), optional
        Precomputed deformation gradients; rebuilt from ``z`` if omitted.
    psd : bool, optional
        If ``True`` (default), project each per-element block to PSD.

    Returns
    -------
    H : np.ndarray or scipy.sparse matrix (r, r)
        Assembled reduced Hessian.
    """
    if F is None:
        dim = precomp.dim
        F = (precomp.JB @ z + precomp.Jx0).reshape(-1, dim, dim)
    He = elastic_hessian_element_F(F, mu, lam, material, psd=psd)
    He = He * np.asarray(vol).reshape(-1, 1, 1)
    H = sp.sparse.block_diag(He)
    return precomp.JB.transpose() @ H @ precomp.JB


def elastic_hessian_filtered_z(z: np.ndarray, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray, material: str, precomp: ElasticEnergyZFilteredPrecomp, F: Optional[np.ndarray] = None, psd: bool = True) -> sp.sparse.spmatrix:
    """Assembled reduced elastic Hessian plus the quadratic filter term.

    Parameters
    ----------
    z : np.ndarray (r, 1)
        Reduced coordinates.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.
    material : str
        Material identifier.
    precomp : ElasticEnergyZFilteredPrecomp
        Filtered reduced precompute.
    F : np.ndarray (t, dim, dim), optional
        Precomputed deformation gradients; rebuilt from ``z`` if omitted.
    psd : bool, optional
        If ``True`` (default), project each per-element block to PSD.

    Returns
    -------
    H : np.ndarray or scipy.sparse matrix (r, r)
        Assembled reduced Hessian including the filter term.
    """
    H = elastic_hessian_z(z, mu, lam, vol, material, precomp, F=F, psd=psd)
    return H + precomp.BJAMuJB


def elastic_hessian_S(S: np.ndarray, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray, material: str) -> np.ndarray:
    """Per-element elastic Hessian in the stretch ``S`` variable (ARAP only).

    Parameters
    ----------
    S : np.ndarray
        Per-element stretch (full or compact Voigt form).
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Unused; present for interface consistency.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.
    material : str
        Material identifier; only ``'arap'`` is supported.

    Returns
    -------
    H : np.ndarray
        Per-element weighted Hessian w.r.t. ``S``.
    """
    if material == 'arap':
        H = arap_hessian_element_S(S, mu)
        return H * np.asarray(vol).reshape((-1,) + (1,) * (H.ndim - 1))
    raise ValueError("Unknown or unsupported material type for S: " + str(material))


# --------------------------------------------------------------------------- #
# Self-contained tier: builds J and vol from rest geometry                    #
# --------------------------------------------------------------------------- #
def elastic_hessian(X: np.ndarray, T: np.ndarray, mu: np.ndarray, lam: np.ndarray, material: str, U: Optional[np.ndarray] = None, psd: bool = True) -> sp.sparse.spmatrix:
    """Elastic Hessian, building the operator and weights from rest geometry.

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
    material : str
        Material identifier.
    U : np.ndarray (n, dim), optional
        Current vertex positions. Defaults to ``X``.
    psd : bool, optional
        If ``True`` (default), project each per-element block to PSD.

    Returns
    -------
    Q : scipy.sparse.csc_matrix (n*dim, n*dim)
        Assembled Hessian.
    """
    if U is None:
        U = X
    J = deformation_jacobian(X, T)
    vol = volume(X, T)
    return elastic_hessian_x(U, J, mu, lam, vol, material, psd=psd)