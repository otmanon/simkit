"""Kinetic (inertial) energy for implicit time integration.

A global quadratic energy ``0.5 * c / h^2 * (x - y)^T M (x - y)`` measuring
deviation from an inertial target ``y``, where ``c`` is an integrator-dependent
constant. It has no per-element / ``F`` decomposition, so it is single-tier:
full-space functions plus reduced (``_z``) variants backed by a small precompute.

Two integrators are provided:

Backward Euler (``*_be``)
    ``c = 1`` and the target ``y`` is supplied directly (``y = x_prev + h *
    v_prev``, equivalently ``2 x_curr - x_prev``).

BDF2 (``*_bdf2``)
    Constant-step BDF2 approximates ``x'`` with ``(3 x - 4 x_prev + x_prev2) /
    (2 h)``. The matching inertial energy uses ``c = 9 / 4`` and the target
    ``y = (4/3) x_prev - (1/3) x_prev2``, built internally from the two history
    states.
"""

import numpy as np
import scipy as sp

# Integrator constants (the leading factor on the 1/h^2 quadratic).
_BE_COEFF = 1.0
_BDF2_COEFF = 9.0 / 4.0


def _bdf2_target(x_prev: np.ndarray, x_prev2: np.ndarray) -> np.ndarray:
    """Constant-step BDF2 inertial target ``(4/3) x_prev - (1/3) x_prev2``."""
    return (4.0 / 3.0) * x_prev - (1.0 / 3.0) * x_prev2


# --------------------------------------------------------------------------- #
# Shared quadratic core                                                       #
# --------------------------------------------------------------------------- #
def _kinetic_energy(d: np.ndarray, M: sp.sparse.spmatrix, h: float, c: float) -> float:
    """``0.5 * c / h^2 * d^T M d`` as a Python float."""
    return float((0.5 * c * (d.T @ M @ d) * (1 / (h ** 2))).item())


def _kinetic_gradient(d: np.ndarray, M: sp.sparse.spmatrix, h: float, c: float) -> np.ndarray:
    """``c / h^2 * M d``."""
    return c * (M @ d) * (1 / (h ** 2))


def _kinetic_hessian(M: sp.sparse.spmatrix, h: float, c: float) -> sp.sparse.spmatrix:
    """``c / h^2 * M``."""
    return M * (c / (h ** 2))


# --------------------------------------------------------------------------- #
# Backward Euler                                                              #
# --------------------------------------------------------------------------- #
def kinetic_energy_be(x: np.ndarray, y: np.ndarray, M: sp.sparse.spmatrix, h: float) -> float:
    """Backward-Euler kinetic energy.

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
    return _kinetic_energy(x - y, M, h, _BE_COEFF)


def kinetic_gradient_be(x: np.ndarray, y: np.ndarray, M: sp.sparse.spmatrix, h: float) -> np.ndarray:
    """Gradient of the backward-Euler kinetic energy w.r.t. ``x``.

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
    return _kinetic_gradient(x - y, M, h, _BE_COEFF)


def kinetic_hessian_be(M: sp.sparse.spmatrix, h: float) -> sp.sparse.spmatrix:
    """Hessian of the backward-Euler kinetic energy w.r.t. ``x``.

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
    return _kinetic_hessian(M, h, _BE_COEFF)


# --------------------------------------------------------------------------- #
# BDF2                                                                        #
# --------------------------------------------------------------------------- #
def kinetic_energy_bdf2(x: np.ndarray, x_prev: np.ndarray, x_prev2: np.ndarray, M: sp.sparse.spmatrix, h: float) -> float:
    """Constant-step BDF2 kinetic energy.

    Builds the inertial target ``y = (4/3) x_prev - (1/3) x_prev2`` from the two
    history states and uses the BDF2 coefficient ``c = 9/4``.

    Parameters
    ----------
    x : np.ndarray (n*d, 1)
        Current positions.
    x_prev : np.ndarray (n*d, 1)
        Positions at the previous step.
    x_prev2 : np.ndarray (n*d, 1)
        Positions two steps back.
    M : scipy.sparse matrix (n*d, n*d)
        Mass matrix.
    h : float
        Timestep.

    Returns
    -------
    k : float
        Kinetic energy.
    """
    y = _bdf2_target(x_prev, x_prev2)
    return _kinetic_energy(x - y, M, h, _BDF2_COEFF)


def kinetic_gradient_bdf2(x: np.ndarray, x_prev: np.ndarray, x_prev2: np.ndarray, M: sp.sparse.spmatrix, h: float) -> np.ndarray:
    """Gradient of the BDF2 kinetic energy w.r.t. ``x``.

    Parameters
    ----------
    x : np.ndarray (n*d, 1)
        Current positions.
    x_prev : np.ndarray (n*d, 1)
        Positions at the previous step.
    x_prev2 : np.ndarray (n*d, 1)
        Positions two steps back.
    M : scipy.sparse matrix (n*d, n*d)
        Mass matrix.
    h : float
        Timestep.

    Returns
    -------
    g : np.ndarray (n*d, 1)
        Energy gradient.
    """
    y = _bdf2_target(x_prev, x_prev2)
    return _kinetic_gradient(x - y, M, h, _BDF2_COEFF)


def kinetic_hessian_bdf2(M: sp.sparse.spmatrix, h: float) -> sp.sparse.spmatrix:
    """Hessian of the BDF2 kinetic energy w.r.t. ``x``.

    Parameters
    ----------
    M : scipy.sparse matrix (n*d, n*d)
        Mass matrix.
    h : float
        Timestep.

    Returns
    -------
    H : scipy.sparse matrix (n*d, n*d)
        Energy Hessian, ``(9/4) M / h^2``. PSD by construction. Independent of
        ``x`` and of the history states.
    """
    return _kinetic_hessian(M, h, _BDF2_COEFF)


# --------------------------------------------------------------------------- #
# Reduced (z) variants                                                        #
# --------------------------------------------------------------------------- #
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


def kinetic_energy_be_z(z: np.ndarray, y: np.ndarray, h: float, precomp: KineticEnergyZPrecomp) -> float:
    """Backward-Euler kinetic energy of the reduced system (``x = B z``).

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
    return _kinetic_energy(z - y, precomp.BMB, h, _BE_COEFF)


def kinetic_gradient_be_z(z: np.ndarray, y: np.ndarray, h: float, precomp: KineticEnergyZPrecomp) -> np.ndarray:
    """Gradient of the reduced backward-Euler kinetic energy w.r.t. ``z``.

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
    return _kinetic_gradient(z - y, precomp.BMB, h, _BE_COEFF)


def kinetic_hessian_be_z(h: float, precomp: KineticEnergyZPrecomp) -> sp.sparse.spmatrix:
    """Hessian of the reduced backward-Euler kinetic energy w.r.t. ``z``.

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
    return _kinetic_hessian(precomp.BMB, h, _BE_COEFF)


def kinetic_energy_bdf2_z(z: np.ndarray, z_prev: np.ndarray, z_prev2: np.ndarray, h: float, precomp: KineticEnergyZPrecomp) -> float:
    """BDF2 kinetic energy of the reduced system (``x = B z``).

    Builds the reduced target ``(4/3) z_prev - (1/3) z_prev2`` and uses the BDF2
    coefficient ``c = 9/4``.

    Parameters
    ----------
    z : np.ndarray (r, 1)
        Reduced positions.
    z_prev : np.ndarray (r, 1)
        Reduced positions at the previous step.
    z_prev2 : np.ndarray (r, 1)
        Reduced positions two steps back.
    h : float
        Timestep.
    precomp : KineticEnergyZPrecomp
        Precompute holding the reduced mass matrix.

    Returns
    -------
    k : float
        Reduced kinetic energy.
    """
    y = _bdf2_target(z_prev, z_prev2)
    return _kinetic_energy(z - y, precomp.BMB, h, _BDF2_COEFF)


def kinetic_gradient_bdf2_z(z: np.ndarray, z_prev: np.ndarray, z_prev2: np.ndarray, h: float, precomp: KineticEnergyZPrecomp) -> np.ndarray:
    """Gradient of the reduced BDF2 kinetic energy w.r.t. ``z``.

    Parameters
    ----------
    z : np.ndarray (r, 1)
        Reduced positions.
    z_prev : np.ndarray (r, 1)
        Reduced positions at the previous step.
    z_prev2 : np.ndarray (r, 1)
        Reduced positions two steps back.
    h : float
        Timestep.
    precomp : KineticEnergyZPrecomp
        Precompute holding the reduced mass matrix.

    Returns
    -------
    g : np.ndarray (r, 1)
        Reduced energy gradient.
    """
    y = _bdf2_target(z_prev, z_prev2)
    return _kinetic_gradient(z - y, precomp.BMB, h, _BDF2_COEFF)


def kinetic_hessian_bdf2_z(h: float, precomp: KineticEnergyZPrecomp) -> sp.sparse.spmatrix:
    """Hessian of the reduced BDF2 kinetic energy w.r.t. ``z``.

    Parameters
    ----------
    h : float
        Timestep.
    precomp : KineticEnergyZPrecomp
        Precompute holding the reduced mass matrix.

    Returns
    -------
    H : np.ndarray or scipy.sparse matrix (r, r)
        Reduced energy Hessian, ``(9/4) BMB / h^2``.
    """
    return _kinetic_hessian(precomp.BMB, h, _BDF2_COEFF)