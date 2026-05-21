
# import numpy as np


# def kinetic_energy(x : np.ndarray, y : np.ndarray, M, h : float):
#     """
#     Computes the kinetic energy of the elastic system

#     Parameters
#     ----------
#     x : (n*d, 1) numpy array
#         positions of the elastic system

#     y : (n*d, 1) numpy array
#         inertial target positions (2 x_curr -  x_prev or equivalently x_curr + h * x_dot_curr)

#     Returns
#     -------
#     k : float
#         Kinetic energy of the elastic system
#     """
#     d = x - y
#     g = 0.5 * d.T @ M @ d * (1/ (h**2))
#     return g



# class KineticEnergyZPrecomp():
#     def __init__(self, B,  M):
#         self.BMB = B.T @ M @ B
    
        
# def kinetic_energy_z(z : np.ndarray, y : np.ndarray, h, precomp : KineticEnergyZPrecomp):
#     """
#     Computes the kinetic energy of the reduced elastic system where the reduced dofs z are the reduced positions x = Bz

#     Parameters
#     ----------
#     z : (r, 1) numpy array
#         reduced positions of the elastic system

#     y : (n*d, 1) numpy array
#         inertial target positions (2 x_curr -  x_prev or equivalently x_curr + h * x_dot_curr)

#     Returns
#     -------
#     k : float
#         Kinetic energy of the elastic system
#     """
    
#     d = z - y
#     E = 0.5 * d.T @ precomp.BMB @ d * (1/ (h**2))

#     return E



# def kinetic_gradient(x : np.ndarray, y : np.ndarray, M, h : float):
#     d = x - y
#     g = M @ d * (1/ (h**2))
#     return g




# def kinetic_gradient_z(z : np.ndarray, y : np.ndarray, h, precomp):
#     d = z - y
#     g = precomp.BMB @ d * (1/ (h**2))
#     return g



# def kinetic_hessian(M, h : float):
   
#     H = M *(1/ (h**2))
#     return H


# def kinetic_hessian_z(h, precomp : KineticEnergyZPrecomp):
#     H = precomp.BMB * (1/ (h**2))
#     return H


"""Kinetic (inertial) energy for implicit time integration.

A global quadratic energy ``0.5/h^2 * (x - y)^T M (x - y)`` measuring deviation
from the inertial target ``y``. It has no per-element / ``F`` decomposition, so
it is single-tier: full-space functions plus reduced (``_z``) variants backed by
a small precompute.
"""

import numpy as np
import scipy as sp


def kinetic_energy(x: np.ndarray, y: np.ndarray, M: sp.sparse.spmatrix, h: float) -> float:
    """Kinetic energy of the elastic system.

    Parameters
    ----------
    x : np.ndarray (n*d, 1)
        Current positions.
    y : np.ndarray (n*d, 1)
        Inertial target positions (``2 x_curr - x_prev``, i.e. ``x_curr + h *
        x_dot_curr``).
    M : scipy.sparse matrix (n*d, n*d)
        Mass matrix.
    h : float
        Timestep.

    Returns
    -------
    k : float
        Kinetic energy.
    """
    d = x - y
    return float(0.5 * (d.T @ M @ d) * (1 / (h ** 2)))


def kinetic_gradient(x: np.ndarray, y: np.ndarray, M: sp.sparse.spmatrix, h: float) -> np.ndarray:
    """Gradient of the kinetic energy w.r.t. ``x``.

    Parameters
    ----------
    x : np.ndarray (n*d, 1)
        Current positions.
    y : np.ndarray (n*d, 1)
        Inertial target positions.
    M : scipy.sparse matrix (n*d, n*d)
        Mass matrix.
    h : float
        Timestep.

    Returns
    -------
    g : np.ndarray (n*d, 1)
        Energy gradient.
    """
    d = x - y
    return M @ d * (1 / (h ** 2))


def kinetic_hessian(M: sp.sparse.spmatrix, h: float) -> sp.sparse.spmatrix:
    """Hessian of the kinetic energy w.r.t. ``x``.

    Parameters
    ----------
    M : scipy.sparse matrix (n*d, n*d)
        Mass matrix.
    h : float
        Timestep.

    Returns
    -------
    H : scipy.sparse matrix (n*d, n*d)
        Energy Hessian, ``M / h^2``. PSD by construction.
    """
    return M * (1 / (h ** 2))


class KineticEnergyZPrecomp:
    """Precompute for the reduced kinetic energy.

    Parameters
    ----------
    B : scipy.sparse matrix (n*d, r)
        Reduced basis.
    M : scipy.sparse matrix (n*d, n*d)
        Mass matrix.

    Attributes
    ----------
    BMB : np.ndarray or scipy.sparse matrix (r, r)
        Reduced mass matrix ``B^T M B``.
    """

    def __init__(self, B: sp.sparse.spmatrix, M: sp.sparse.spmatrix):
        self.BMB = B.T @ M @ B


def kinetic_energy_z(z: np.ndarray, y: np.ndarray, h: float, precomp: KineticEnergyZPrecomp) -> float:
    """Kinetic energy of the reduced system (``x = B z``).

    Parameters
    ----------
    z : np.ndarray (r, 1)
        Reduced positions.
    y : np.ndarray (r, 1)
        Reduced inertial target.
    h : float
        Timestep.
    precomp : KineticEnergyZPrecomp
        Precompute holding the reduced mass matrix.

    Returns
    -------
    k : float
        Reduced kinetic energy.
    """
    d = z - y
    return float(0.5 * (d.T @ precomp.BMB @ d) * (1 / (h ** 2)))


def kinetic_gradient_z(z: np.ndarray, y: np.ndarray, h: float, precomp: KineticEnergyZPrecomp) -> np.ndarray:
    """Gradient of the reduced kinetic energy w.r.t. ``z``.

    Parameters
    ----------
    z : np.ndarray (r, 1)
        Reduced positions.
    y : np.ndarray (r, 1)
        Reduced inertial target.
    h : float
        Timestep.
    precomp : KineticEnergyZPrecomp
        Precompute holding the reduced mass matrix.

    Returns
    -------
    g : np.ndarray (r, 1)
        Reduced energy gradient.
    """
    d = z - y
    return precomp.BMB @ d * (1 / (h ** 2))


def kinetic_hessian_z(h: float, precomp: KineticEnergyZPrecomp) -> sp.sparse.spmatrix:
    """Hessian of the reduced kinetic energy w.r.t. ``z``.

    Parameters
    ----------
    h : float
        Timestep.
    precomp : KineticEnergyZPrecomp
        Precompute holding the reduced mass matrix.

    Returns
    -------
    H : np.ndarray or scipy.sparse matrix (r, r)
        Reduced energy Hessian, ``BMB / h^2``.
    """
    return precomp.BMB * (1 / (h ** 2))