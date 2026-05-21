"""Kinetic (inertial) energy for implicit time integration.

A global quadratic energy ``0.5 * c / h^2 * (x - x_tilde)^T M (x - x_tilde)``
measuring deviation of a candidate position ``x`` from an inertial target
``x_tilde``, where ``c`` is an integrator-dependent constant. It has no
per-element / ``F`` decomposition, so it is single-tier: full-space functions
plus reduced (``_z``) variants backed by a small precompute.

Positions-only API
------------------
``x`` is the optimization variable: the *candidate* for the next-step position
the solver iterates on. The integrators take only known positions; the required
velocity history is reconstructed internally from those positions via
:func:`velocity_be` / :func:`velocity_bdf2`. Because the BDF2 target needs both
``v_curr`` and ``v_prev`` reconstructed by the second-order backward difference,
BDF2 consumes one more position level than backward Euler.

Backward Euler (``*_be``)
    ``c = 1``. ``v_curr = (x_curr - x_prev) / h`` and
    ``x_tilde = x_curr + h * v_curr`` (i.e. ``2 x_curr - x_prev``). History:
    ``x_curr``, ``x_prev``.

BDF2 (``*_bdf2``)
    ``c = (3/2)^2 = 9/4``. ``v_curr`` and ``v_prev`` come from the BDF2 backward
    difference, and
    ``x_tilde = (4/3) x_curr - (1/3) x_prev + (8h/9) v_curr - (2h/9) v_prev``.
    History: ``x_curr``, ``x_prev``, ``x_prev2``, ``x_prev3``.
"""

import numpy as np
import scipy as sp

# Integrator constants (the leading factor on the 1/h^2 quadratic).
_BE_COEFF = 1.0
_BDF2_COEFF = 9.0 / 4.0


# --------------------------------------------------------------------------- #
# Velocity reconstruction from positions                                      #
# --------------------------------------------------------------------------- #
def velocity_be(x_curr: np.ndarray, x_prev: np.ndarray, h: float) -> np.ndarray:
    """Backward-Euler velocity estimate ``(x_curr - x_prev) / h``.

    First-order backward difference. Exact for constant-velocity motion.

    Parameters
    ----------
    x_curr : np.ndarray (*, 1)
        Most recent known position.
    x_prev : np.ndarray (*, 1)
        Position one step earlier.
    h : float
        Timestep.

    Returns
    -------
    v : np.ndarray (*, 1)
        Velocity estimate at ``x_curr``.
    """
    return (x_curr - x_prev) / h


def velocity_bdf2(x_curr: np.ndarray, x_prev: np.ndarray, x_prev2: np.ndarray, h: float) -> np.ndarray:
    """BDF2 velocity estimate ``(3 x_curr - 4 x_prev + x_prev2) / (2 h)``.

    Second-order backward difference. Exact for quadratic motion.

    Parameters
    ----------
    x_curr : np.ndarray (*, 1)
        Most recent known position.
    x_prev : np.ndarray (*, 1)
        Position one step earlier.
    x_prev2 : np.ndarray (*, 1)
        Position two steps earlier.
    h : float
        Timestep.

    Returns
    -------
    v : np.ndarray (*, 1)
        Velocity estimate at ``x_curr``.
    """
    return (3.0 * x_curr - 4.0 * x_prev + x_prev2) / (2.0 * h)


def _be_target(x_curr: np.ndarray, x_prev: np.ndarray, h: float) -> np.ndarray:
    """Backward-Euler inertial target ``x_curr + h * v_curr``."""
    v_curr = velocity_be(x_curr, x_prev, h)
    return x_curr + h * v_curr


def _bdf2_target(x_curr: np.ndarray, x_prev: np.ndarray, x_prev2: np.ndarray, x_prev3: np.ndarray, h: float) -> np.ndarray:
    """Constant-step BDF2 inertial target built from reconstructed velocities.

    ``(4/3) x_curr - (1/3) x_prev + (8h/9) v_curr - (2h/9) v_prev`` with both
    velocities from :func:`velocity_bdf2`.
    """
    v_curr = velocity_bdf2(x_curr, x_prev, x_prev2, h)
    v_prev = velocity_bdf2(x_prev, x_prev2, x_prev3, h)
    return (4.0 / 3.0) * x_curr - (1.0 / 3.0) * x_prev + (8.0 * h / 9.0) * v_curr - (2.0 * h / 9.0) * v_prev


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
def kinetic_energy_be(x: np.ndarray, x_curr: np.ndarray, x_prev: np.ndarray, M: sp.sparse.spmatrix, h: float) -> float:
    """Backward-Euler kinetic energy of a candidate position ``x``.

    The velocity is reconstructed from ``x_curr`` and ``x_prev`` and the target
    is ``x_tilde = x_curr + h v_curr``; ``c = 1``.

    Parameters
    ----------
    x : np.ndarray (n*d, 1)
        Candidate next-step position (the optimization variable).
    x_curr : np.ndarray (n*d, 1)
        Most recent known position.
    x_prev : np.ndarray (n*d, 1)
        Position one step earlier.
    M : scipy.sparse matrix (n*d, n*d)
        Mass matrix.
    h : float
        Timestep.

    Returns
    -------
    k : float
        Kinetic energy.
    """
    x_tilde = _be_target(x_curr, x_prev, h)
    return _kinetic_energy(x - x_tilde, M, h, _BE_COEFF)


def kinetic_gradient_be(x: np.ndarray, x_curr: np.ndarray, x_prev: np.ndarray, M: sp.sparse.spmatrix, h: float) -> np.ndarray:
    """Gradient of the backward-Euler kinetic energy w.r.t. ``x``.

    Parameters
    ----------
    x : np.ndarray (n*d, 1)
        Candidate next-step position.
    x_curr : np.ndarray (n*d, 1)
        Most recent known position.
    x_prev : np.ndarray (n*d, 1)
        Position one step earlier.
    M : scipy.sparse matrix (n*d, n*d)
        Mass matrix.
    h : float
        Timestep.

    Returns
    -------
    g : np.ndarray (n*d, 1)
        Energy gradient.
    """
    x_tilde = _be_target(x_curr, x_prev, h)
    return _kinetic_gradient(x - x_tilde, M, h, _BE_COEFF)


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
        Energy Hessian, ``M / h^2``. PSD by construction. Independent of ``x``
        and of the history states.
    """
    return _kinetic_hessian(M, h, _BE_COEFF)


# --------------------------------------------------------------------------- #
# BDF2                                                                        #
# --------------------------------------------------------------------------- #
def kinetic_energy_bdf2(x: np.ndarray, x_curr: np.ndarray, x_prev: np.ndarray, x_prev2: np.ndarray, x_prev3: np.ndarray, M: sp.sparse.spmatrix, h: float) -> float:
    """Constant-step BDF2 kinetic energy of a candidate position ``x``.

    ``v_curr`` and ``v_prev`` are reconstructed by the BDF2 backward difference
    and the target is
    ``x_tilde = (4/3) x_curr - (1/3) x_prev + (8h/9) v_curr - (2h/9) v_prev``;
    ``c = 9/4``.

    Parameters
    ----------
    x : np.ndarray (n*d, 1)
        Candidate next-step position (the optimization variable).
    x_curr : np.ndarray (n*d, 1)
        Most recent known position.
    x_prev : np.ndarray (n*d, 1)
        Position one step earlier.
    x_prev2 : np.ndarray (n*d, 1)
        Position two steps earlier.
    x_prev3 : np.ndarray (n*d, 1)
        Position three steps earlier.
    M : scipy.sparse matrix (n*d, n*d)
        Mass matrix.
    h : float
        Timestep.

    Returns
    -------
    k : float
        Kinetic energy.
    """
    x_tilde = _bdf2_target(x_curr, x_prev, x_prev2, x_prev3, h)
    return _kinetic_energy(x - x_tilde, M, h, _BDF2_COEFF)


def kinetic_gradient_bdf2(x: np.ndarray, x_curr: np.ndarray, x_prev: np.ndarray, x_prev2: np.ndarray, x_prev3: np.ndarray, M: sp.sparse.spmatrix, h: float) -> np.ndarray:
    """Gradient of the BDF2 kinetic energy w.r.t. ``x``.

    Parameters
    ----------
    x : np.ndarray (n*d, 1)
        Candidate next-step position.
    x_curr : np.ndarray (n*d, 1)
        Most recent known position.
    x_prev : np.ndarray (n*d, 1)
        Position one step earlier.
    x_prev2 : np.ndarray (n*d, 1)
        Position two steps earlier.
    x_prev3 : np.ndarray (n*d, 1)
        Position three steps earlier.
    M : scipy.sparse matrix (n*d, n*d)
        Mass matrix.
    h : float
        Timestep.

    Returns
    -------
    g : np.ndarray (n*d, 1)
        Energy gradient.
    """
    x_tilde = _bdf2_target(x_curr, x_prev, x_prev2, x_prev3, h)
    return _kinetic_gradient(x - x_tilde, M, h, _BDF2_COEFF)


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


def kinetic_energy_be_z(z: np.ndarray, z_curr: np.ndarray, z_prev: np.ndarray, h: float, precomp: KineticEnergyZPrecomp) -> float:
    """Backward-Euler kinetic energy of the reduced system (``x = B z``).

    Parameters
    ----------
    z : np.ndarray (r, 1)
        Candidate reduced next-step position.
    z_curr : np.ndarray (r, 1)
        Most recent known reduced position.
    z_prev : np.ndarray (r, 1)
        Reduced position one step earlier.
    h : float
        Timestep.
    precomp : KineticEnergyZPrecomp
        Precompute holding the reduced mass matrix.

    Returns
    -------
    k : float
        Reduced kinetic energy.
    """
    z_tilde = _be_target(z_curr, z_prev, h)
    return _kinetic_energy(z - z_tilde, precomp.BMB, h, _BE_COEFF)


def kinetic_gradient_be_z(z: np.ndarray, z_curr: np.ndarray, z_prev: np.ndarray, h: float, precomp: KineticEnergyZPrecomp) -> np.ndarray:
    """Gradient of the reduced backward-Euler kinetic energy w.r.t. ``z``.

    Parameters
    ----------
    z : np.ndarray (r, 1)
        Candidate reduced next-step position.
    z_curr : np.ndarray (r, 1)
        Most recent known reduced position.
    z_prev : np.ndarray (r, 1)
        Reduced position one step earlier.
    h : float
        Timestep.
    precomp : KineticEnergyZPrecomp
        Precompute holding the reduced mass matrix.

    Returns
    -------
    g : np.ndarray (r, 1)
        Reduced energy gradient.
    """
    z_tilde = _be_target(z_curr, z_prev, h)
    return _kinetic_gradient(z - z_tilde, precomp.BMB, h, _BE_COEFF)


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


def kinetic_energy_bdf2_z(z: np.ndarray, z_curr: np.ndarray, z_prev: np.ndarray, z_prev2: np.ndarray, z_prev3: np.ndarray, h: float, precomp: KineticEnergyZPrecomp) -> float:
    """BDF2 kinetic energy of the reduced system (``x = B z``).

    Parameters
    ----------
    z : np.ndarray (r, 1)
        Candidate reduced next-step position.
    z_curr : np.ndarray (r, 1)
        Most recent known reduced position.
    z_prev : np.ndarray (r, 1)
        Reduced position one step earlier.
    z_prev2 : np.ndarray (r, 1)
        Reduced position two steps earlier.
    z_prev3 : np.ndarray (r, 1)
        Reduced position three steps earlier.
    h : float
        Timestep.
    precomp : KineticEnergyZPrecomp
        Precompute holding the reduced mass matrix.

    Returns
    -------
    k : float
        Reduced kinetic energy.
    """
    z_tilde = _bdf2_target(z_curr, z_prev, z_prev2, z_prev3, h)
    return _kinetic_energy(z - z_tilde, precomp.BMB, h, _BDF2_COEFF)


def kinetic_gradient_bdf2_z(z: np.ndarray, z_curr: np.ndarray, z_prev: np.ndarray, z_prev2: np.ndarray, z_prev3: np.ndarray, h: float, precomp: KineticEnergyZPrecomp) -> np.ndarray:
    """Gradient of the reduced BDF2 kinetic energy w.r.t. ``z``.

    Parameters
    ----------
    z : np.ndarray (r, 1)
        Candidate reduced next-step position.
    z_curr : np.ndarray (r, 1)
        Most recent known reduced position.
    z_prev : np.ndarray (r, 1)
        Reduced position one step earlier.
    z_prev2 : np.ndarray (r, 1)
        Reduced position two steps earlier.
    z_prev3 : np.ndarray (r, 1)
        Reduced position three steps earlier.
    h : float
        Timestep.
    precomp : KineticEnergyZPrecomp
        Precompute holding the reduced mass matrix.

    Returns
    -------
    g : np.ndarray (r, 1)
        Reduced energy gradient.
    """
    z_tilde = _bdf2_target(z_curr, z_prev, z_prev2, z_prev3, h)
    return _kinetic_gradient(z - z_tilde, precomp.BMB, h, _BDF2_COEFF)


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