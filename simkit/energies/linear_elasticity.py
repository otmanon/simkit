# import numpy as np
# import scipy as sp

# from ..deformation_jacobian import deformation_jacobian
# from ..volume import volume
# from ..vectorized_trace import vectorized_trace
# from ..vectorized_transpose import vectorized_transpose

# def linear_elasticity_energy_F(F, mu, lam, vol):
#     """
#     Linear Elasticity from the Sifakis Notes
#     https://www.cs.toronto.edu/~jacobson/seminar/sifakis-course-notes-2012.pdf
#     """
#     assert(F.ndim == 3)
#     assert(F.shape[1] == F.shape[2])
    
#     mu = np.array(mu).reshape(-1, 1)
#     lam = np.array(lam).reshape(-1, 1)
#     vol = np.array(vol).reshape(-1, 1)
#     eps = (F + F.transpose(0, 2, 1))/2 - np.identity(F.shape[1])[None, :, :]
#     density = mu.flatten()* np.sum(eps**2, axis=(1, 2)) + (lam.flatten()/2.0)* np.trace(eps, axis1=1, axis2=2)**2
#     E = (density * vol.flatten()).sum()
#     return E


# def linear_elasticity_energy_density_F(F, mu, lam, vol):
#     """
#     Linear Elasticity from the Sifakis Notes
#     https://www.cs.toronto.edu/~jacobson/seminar/sifakis-course-notes-2012.pdf
#     """
#     assert(F.ndim == 3)
#     assert(F.shape[1] == F.shape[2])
    
#     mu = np.array(mu).reshape(-1, 1)
#     lam = np.array(lam).reshape(-1, 1)
#     vol = np.array(vol).reshape(-1, 1)
#     eps = (F + F.transpose(0, 2, 1))/2 - np.identity(F.shape[1])[None, :, :]
#     density = mu.flatten()* np.sum(eps**2, axis=(1, 2)) + (lam.flatten()/2.0)* np.trace(eps, axis1=1, axis2=2)**2
#     psi = (density * vol.flatten())
#     return psi



# def linear_elasticity_energy_df(f, mu, lam, vol, dim):
#     """
#     Linear Elasticity from the Sifakis Notes
#     https://www.cs.toronto.edu/~jacobson/seminar/sifakis-course-notes-2012.pdf
#     """

#     nt = f.shape[0]//(dim*dim)

#     T = vectorized_transpose(nt, dim)
#     Tr = vectorized_trace(nt, dim)

#     i = np.repeat(np.identity(dim)[None, :, :], nt, axis=0).flatten()[:, None]
#     eps = (f + T @ f)/2 - i


#     density = mu * Tr @ (eps**2) + (lam/2.0)* (Tr @ eps)**2

#     E = (density.flatten() * vol.flatten()).sum()

#     return E





# def linear_elasticity_gradient_dF(F, mu, lam, vol):
#     """
#     Linear Elasticity from the Sifakis Notes
#     https://www.cs.toronto.edu/~jacobson/seminar/sifakis-course-notes-2012.pdf
#     """

#     mu = np.array(mu)
#     lam = np.array(lam)
#     vol = np.array(vol)
#     assert(F.ndim == 3)
#     assert(F.shape[1] == F.shape[2])

#     # dtracedeps = np.repeat(np.identity(F.shape[1])[None, :], F.shape[0], axis=0)
#     I = np.identity(F.shape[1])[None, :, :]
#     dPsidF = mu.reshape(-1, 1, 1) * (F + F.transpose(0, 2, 1) - 2*I ) + lam.reshape(-1, 1, 1) * np.trace(F - I, axis1=1, axis2=2)[:, None, None] * np.identity(F.shape[1])[None, :, :]
#     P = (dPsidF * vol.flatten()[:, None, None])

#     return P



# def linear_elasticity_gradient_dx(F, mu, lam, vol, J):
#     P = linear_elasticity_gradient_dF(F, mu, lam, vol)

#     g = J.T @ P.reshape(-1, 1)
#     return g




# def linear_elasticity_hessian(**kwargs):
#     if 'mu' in kwargs:
#         mu = kwargs['mu']
#     else :
#         mu = 1

#     if 'lam' in kwargs:
#         lam = kwargs['lam']
#     else :
#         lam = 1

#     if 'X'  in kwargs and 'T' in kwargs:
#         X = kwargs['X']
#         T = kwargs['T']

#         if 'U' in kwargs:
#             U = kwargs['U']
#         else:
#             U = kwargs['X'].copy()
        
#         if 'J' in kwargs:
#             J = kwargs['J']
#         else:        
#             J = deformation_jacobian(X, T)

#         if 'vol' in kwargs:
#             vol = kwargs['vol']
#         else:
#             vol = volume(X, T)

#         return linear_elasticity_hessian_d2x(U,mu, lam, vol, J=J)
#     else:
#         ValueError("X and T are required")


# def linear_elasticity_hessian_d2F(F, mu, lam, vol):
    
#     mu = np.array(mu)
#     lam = np.array(lam)
#     vol = np.array(vol)
#     assert(F.ndim == 3)
#     assert(F.shape[1] == F.shape[2])

#     dim = F.shape[1]

#     I = np.identity(dim*dim)
#     T = np.asarray(vectorized_transpose(1, F.shape[1]).toarray())
#     Tr = np.asarray(vectorized_trace(1, F.shape[1]).toarray())


#     H1 = np.repeat((I + T)[None, :, :], F.shape[0], axis=0)
#     H2 = np.repeat((Tr.T @ Tr)[None, :, :], F.shape[0], axis=0)

#     H = (mu.reshape(-1, 1, 1) * H1 + lam.reshape(-1, 1, 1) * H2)*vol.flatten()[:, None, None]

#     return H


# def linear_elasticity_hessian_d2x(U, mu,  lam, vol, J):

#     dim = U.shape[1]
#     F = (J @ U.reshape(-1, 1)).reshape(-1, dim, dim)
#     d2psidF2 = linear_elasticity_hessian_d2F(F, mu=mu, lam=lam, vol=vol)
#     H = sp.sparse.block_diag(d2psidF2)  # block diagonal hessian matrix
#     Q = J.transpose() @ H @ J
#     return Q


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