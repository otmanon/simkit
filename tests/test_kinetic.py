"""Tests for ``simkit.energies.kinetic``.

Two implicit integrators share one quadratic inertial energy
``0.5 * c / h^2 * (x - x_tilde)^T M (x - x_tilde)``. ``x`` is the candidate
next-step position; the velocity history is reconstructed from positions:

* Backward Euler (``*_be``): ``c = 1``, velocity from ``(x_curr, x_prev)``,
  target ``x_curr + h v_curr`` (history: 2 positions).
* BDF2 (``*_bdf2``): ``c = 9/4``, velocities from the BDF2 backward difference,
  target ``(4/3) x_curr - (1/3) x_prev + (8h/9) v_curr - (2h/9) v_prev``
  (history: 4 positions).

Both are quadratic in ``x``, so the Hessian (``c M / h^2``) is independent of
``x`` and of the history.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

from simkit.energies.kinetic import (
    velocity_be,
    velocity_bdf2,
    kinetic_energy_be,
    kinetic_gradient_be,
    kinetic_hessian_be,
    kinetic_energy_bdf2,
    kinetic_gradient_bdf2,
    kinetic_hessian_bdf2,
)
from simkit.gradient_cfd import gradient_cfd


FD_STEP = 1e-6
GRAD_TOL = 1e-7
HESS_TOL = 1e-7
BDF2_COEFF = 9.0 / 4.0


def _spd_mass(rng: np.random.Generator, n: int) -> sps.csc_matrix:
    A = rng.standard_normal((n, n))
    return sps.csc_matrix(A.T @ A + n * np.eye(n))  # SPD


def _be_target(x_curr, x_prev, h):
    return x_curr + h * velocity_be(x_curr, x_prev, h)


def _bdf2_target(x_curr, x_prev, x_prev2, x_prev3, h):
    v_curr = velocity_bdf2(x_curr, x_prev, x_prev2, h)
    v_prev = velocity_bdf2(x_prev, x_prev2, x_prev3, h)
    return (4.0 / 3.0) * x_curr - (1.0 / 3.0) * x_prev + (8.0 * h / 9.0) * v_curr - (2.0 * h / 9.0) * v_prev


# --------------------------------------------------------------------------- #
# Velocity reconstruction helpers                                            #
# --------------------------------------------------------------------------- #
def test_velocity_be_exact_for_constant_velocity() -> None:
    rng = np.random.default_rng(10)
    h = 0.02
    x0 = rng.standard_normal((5, 1))
    v = rng.standard_normal((5, 1))
    x_prev = x0
    x_curr = x0 + h * v
    assert np.allclose(velocity_be(x_curr, x_prev, h), v, atol=1e-12)


def test_velocity_bdf2_exact_for_quadratic() -> None:
    rng = np.random.default_rng(11)
    h = 0.02
    a = rng.standard_normal((5, 1))
    b = rng.standard_normal((5, 1))
    c = rng.standard_normal((5, 1))

    def x_at(t):
        return a + b * t + c * t ** 2

    # samples at t = 0, h, 2h (prev2, prev, curr); analytic v at t=2h is b + 4 c h
    x_prev2 = x_at(0.0)
    x_prev = x_at(h)
    x_curr = x_at(2.0 * h)
    v_exact = b + 2.0 * c * (2.0 * h)
    assert np.allclose(velocity_bdf2(x_curr, x_prev, x_prev2, h), v_exact, atol=1e-12)


# --------------------------------------------------------------------------- #
# Backward Euler                                                              #
# --------------------------------------------------------------------------- #
def test_kinetic_be_energy_zero_at_target_and_increases() -> None:
    rng = np.random.default_rng(0)
    M = _spd_mass(rng, 6)
    x_curr = rng.standard_normal((6, 1))
    x_prev = rng.standard_normal((6, 1))
    h = 0.02

    x_tilde = _be_target(x_curr, x_prev, h)
    e_rest = kinetic_energy_be(x_tilde.copy(), x_curr, x_prev, M, h)
    e_def = kinetic_energy_be(x_tilde + 0.1 * rng.standard_normal(x_tilde.shape), x_curr, x_prev, M, h)

    assert e_rest == pytest.approx(0.0, abs=1e-12)
    assert e_def > e_rest


def test_kinetic_be_gradient_matches_fd() -> None:
    rng = np.random.default_rng(1)
    M = _spd_mass(rng, 5)
    x_curr = rng.standard_normal((5, 1))
    x_prev = rng.standard_normal((5, 1))
    h = 0.02
    x = rng.standard_normal((5, 1))
    n = x.shape[0]

    def energy_flat(x_flat: np.ndarray) -> np.ndarray:
        return np.array([kinetic_energy_be(x_flat.reshape(n, 1), x_curr, x_prev, M, h)])

    g_fd = gradient_cfd(energy_flat, x.flatten(), FD_STEP).flatten()
    g = np.asarray(kinetic_gradient_be(x, x_curr, x_prev, M, h)).flatten()
    assert np.allclose(g, g_fd, atol=GRAD_TOL)


def test_kinetic_be_hessian_matches_fd() -> None:
    rng = np.random.default_rng(2)
    M = _spd_mass(rng, 4)
    x_curr = rng.standard_normal((4, 1))
    x_prev = rng.standard_normal((4, 1))
    h = 0.02
    x = rng.standard_normal((4, 1))
    n = x.shape[0]

    def grad_flat(x_flat: np.ndarray) -> np.ndarray:
        return np.asarray(
            kinetic_gradient_be(x_flat.reshape(n, 1), x_curr, x_prev, M, h)
        ).flatten()

    H_fd = gradient_cfd(grad_flat, x.flatten(), FD_STEP)
    H = np.asarray(kinetic_hessian_be(M, h).todense())
    assert np.allclose(H, H_fd, atol=HESS_TOL)


# --------------------------------------------------------------------------- #
# BDF2                                                                        #
# --------------------------------------------------------------------------- #
def test_kinetic_bdf2_energy_zero_at_target_and_increases() -> None:
    rng = np.random.default_rng(3)
    M = _spd_mass(rng, 6)
    x_curr = rng.standard_normal((6, 1))
    x_prev = rng.standard_normal((6, 1))
    x_prev2 = rng.standard_normal((6, 1))
    x_prev3 = rng.standard_normal((6, 1))
    h = 0.02

    x_tilde = _bdf2_target(x_curr, x_prev, x_prev2, x_prev3, h)
    e_rest = kinetic_energy_bdf2(x_tilde.copy(), x_curr, x_prev, x_prev2, x_prev3, M, h)
    e_def = kinetic_energy_bdf2(
        x_tilde + 0.1 * rng.standard_normal(x_tilde.shape), x_curr, x_prev, x_prev2, x_prev3, M, h
    )

    assert e_rest == pytest.approx(0.0, abs=1e-12)
    assert e_def > e_rest


def test_kinetic_bdf2_gradient_matches_fd() -> None:
    rng = np.random.default_rng(4)
    M = _spd_mass(rng, 5)
    x_curr = rng.standard_normal((5, 1))
    x_prev = rng.standard_normal((5, 1))
    x_prev2 = rng.standard_normal((5, 1))
    x_prev3 = rng.standard_normal((5, 1))
    h = 0.02
    x = rng.standard_normal((5, 1))
    n = x.shape[0]

    def energy_flat(x_flat: np.ndarray) -> np.ndarray:
        return np.array(
            [kinetic_energy_bdf2(x_flat.reshape(n, 1), x_curr, x_prev, x_prev2, x_prev3, M, h)]
        )

    g_fd = gradient_cfd(energy_flat, x.flatten(), FD_STEP).flatten()
    g = np.asarray(kinetic_gradient_bdf2(x, x_curr, x_prev, x_prev2, x_prev3, M, h)).flatten()
    assert np.allclose(g, g_fd, atol=GRAD_TOL)


def test_kinetic_bdf2_hessian_matches_fd() -> None:
    rng = np.random.default_rng(5)
    M = _spd_mass(rng, 4)
    x_curr = rng.standard_normal((4, 1))
    x_prev = rng.standard_normal((4, 1))
    x_prev2 = rng.standard_normal((4, 1))
    x_prev3 = rng.standard_normal((4, 1))
    h = 0.02
    x = rng.standard_normal((4, 1))
    n = x.shape[0]

    def grad_flat(x_flat: np.ndarray) -> np.ndarray:
        return np.asarray(
            kinetic_gradient_bdf2(x_flat.reshape(n, 1), x_curr, x_prev, x_prev2, x_prev3, M, h)
        ).flatten()

    H_fd = gradient_cfd(grad_flat, x.flatten(), FD_STEP)
    H = np.asarray(kinetic_hessian_bdf2(M, h).todense())
    assert np.allclose(H, H_fd, atol=HESS_TOL)


def test_kinetic_bdf2_scales_be_by_nine_quarters() -> None:
    # At equal displacement (x - x_tilde) the BDF2 energy is exactly 9/4 the BE
    # one. Feed each its own target plus the same offset so the residuals match.
    rng = np.random.default_rng(6)
    M = _spd_mass(rng, 5)
    h = 0.02
    x_curr = rng.standard_normal((5, 1))
    x_prev = rng.standard_normal((5, 1))
    x_prev2 = rng.standard_normal((5, 1))
    x_prev3 = rng.standard_normal((5, 1))
    delta = 0.1 * rng.standard_normal((5, 1))

    x_be = _be_target(x_curr, x_prev, h) + delta
    x_bdf2 = _bdf2_target(x_curr, x_prev, x_prev2, x_prev3, h) + delta

    e_be = kinetic_energy_be(x_be, x_curr, x_prev, M, h)
    e_bdf2 = kinetic_energy_bdf2(x_bdf2, x_curr, x_prev, x_prev2, x_prev3, M, h)
    assert e_bdf2 == pytest.approx(BDF2_COEFF * e_be, rel=1e-12)


if __name__ == "__main__":
    test_velocity_be_exact_for_constant_velocity()
    test_velocity_bdf2_exact_for_quadratic()
    test_kinetic_be_energy_zero_at_target_and_increases()
    test_kinetic_be_gradient_matches_fd()
    test_kinetic_be_hessian_matches_fd()
    test_kinetic_bdf2_energy_zero_at_target_and_increases()
    test_kinetic_bdf2_gradient_matches_fd()
    test_kinetic_bdf2_hessian_matches_fd()
    test_kinetic_bdf2_scales_be_by_nine_quarters()