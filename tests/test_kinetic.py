"""Tests for ``simkit.energies.kinetic``.

Two implicit integrators share one quadratic inertial energy
``0.5 * c / h^2 * (x - y)^T M (x - y)``:

* Backward Euler (``*_be``): ``c = 1`` and the target ``y`` is passed in.
* BDF2 (``*_bdf2``): ``c = 9/4`` and the target ``y = (4/3) x_prev -
  (1/3) x_prev2`` is built from two history states.

Both are quadratic in ``x``, so the Hessian (``c M / h^2``) is independent of
``x``.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

from simkit.energies.kinetic import (
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


# --------------------------------------------------------------------------- #
# Backward Euler                                                              #
# --------------------------------------------------------------------------- #
def test_kinetic_be_energy_zero_at_target_and_increases() -> None:
    rng = np.random.default_rng(0)
    M = _spd_mass(rng, 6)
    y = rng.standard_normal((6, 1))
    h = 0.02

    e_rest = kinetic_energy_be(y.copy(), y, M, h)
    e_def = kinetic_energy_be(y + 0.1 * rng.standard_normal(y.shape), y, M, h)

    assert e_rest == pytest.approx(0.0, abs=1e-12)
    assert e_def > e_rest


def test_kinetic_be_gradient_matches_fd() -> None:
    rng = np.random.default_rng(1)
    M = _spd_mass(rng, 5)
    y = rng.standard_normal((5, 1))
    h = 0.02
    x = y + 0.1 * rng.standard_normal(y.shape)
    n = x.shape[0]

    def energy_flat(x_flat: np.ndarray) -> np.ndarray:
        return np.array([kinetic_energy_be(x_flat.reshape(n, 1), y, M, h)])

    g_fd = gradient_cfd(energy_flat, x.flatten(), FD_STEP).flatten()
    g = np.asarray(kinetic_gradient_be(x, y, M, h)).flatten()
    assert np.allclose(g, g_fd, atol=GRAD_TOL)


def test_kinetic_be_hessian_matches_fd() -> None:
    rng = np.random.default_rng(2)
    M = _spd_mass(rng, 4)
    y = rng.standard_normal((4, 1))
    h = 0.02
    x = y + 0.1 * rng.standard_normal(y.shape)
    n = x.shape[0]

    def grad_flat(x_flat: np.ndarray) -> np.ndarray:
        return np.asarray(kinetic_gradient_be(x_flat.reshape(n, 1), y, M, h)).flatten()

    H_fd = gradient_cfd(grad_flat, x.flatten(), FD_STEP)
    H = np.asarray(kinetic_hessian_be(M, h).todense())
    assert np.allclose(H, H_fd, atol=HESS_TOL)


# --------------------------------------------------------------------------- #
# BDF2                                                                        #
# --------------------------------------------------------------------------- #
def test_kinetic_bdf2_energy_zero_at_target_and_increases() -> None:
    rng = np.random.default_rng(3)
    M = _spd_mass(rng, 6)
    x_prev = rng.standard_normal((6, 1))
    x_prev2 = rng.standard_normal((6, 1))
    h = 0.02

    y = (4.0 / 3.0) * x_prev - (1.0 / 3.0) * x_prev2  # the internal target
    e_rest = kinetic_energy_bdf2(y.copy(), x_prev, x_prev2, M, h)
    e_def = kinetic_energy_bdf2(
        y + 0.1 * rng.standard_normal(y.shape), x_prev, x_prev2, M, h
    )

    assert e_rest == pytest.approx(0.0, abs=1e-12)
    assert e_def > e_rest


def test_kinetic_bdf2_gradient_matches_fd() -> None:
    rng = np.random.default_rng(4)
    M = _spd_mass(rng, 5)
    x_prev = rng.standard_normal((5, 1))
    x_prev2 = rng.standard_normal((5, 1))
    h = 0.02
    x = rng.standard_normal((5, 1))
    n = x.shape[0]

    def energy_flat(x_flat: np.ndarray) -> np.ndarray:
        return np.array(
            [kinetic_energy_bdf2(x_flat.reshape(n, 1), x_prev, x_prev2, M, h)]
        )

    g_fd = gradient_cfd(energy_flat, x.flatten(), FD_STEP).flatten()
    g = np.asarray(kinetic_gradient_bdf2(x, x_prev, x_prev2, M, h)).flatten()
    assert np.allclose(g, g_fd, atol=GRAD_TOL)


def test_kinetic_bdf2_hessian_matches_fd() -> None:
    rng = np.random.default_rng(5)
    M = _spd_mass(rng, 4)
    x_prev = rng.standard_normal((4, 1))
    x_prev2 = rng.standard_normal((4, 1))
    h = 0.02
    x = rng.standard_normal((4, 1))
    n = x.shape[0]

    def grad_flat(x_flat: np.ndarray) -> np.ndarray:
        return np.asarray(
            kinetic_gradient_bdf2(x_flat.reshape(n, 1), x_prev, x_prev2, M, h)
        ).flatten()

    H_fd = gradient_cfd(grad_flat, x.flatten(), FD_STEP)
    H = np.asarray(kinetic_hessian_bdf2(M, h).todense())
    assert np.allclose(H, H_fd, atol=HESS_TOL)


def test_kinetic_bdf2_scales_be_by_nine_quarters() -> None:
    # With identical (x - y), BDF2 is exactly 9/4 times the BE energy.
    rng = np.random.default_rng(6)
    M = _spd_mass(rng, 5)
    h = 0.02
    x_prev = rng.standard_normal((5, 1))
    x_prev2 = rng.standard_normal((5, 1))
    y = (4.0 / 3.0) * x_prev - (1.0 / 3.0) * x_prev2
    x = y + 0.1 * rng.standard_normal(y.shape)

    e_be = kinetic_energy_be(x, y, M, h)
    e_bdf2 = kinetic_energy_bdf2(x, x_prev, x_prev2, M, h)
    assert e_bdf2 == pytest.approx(BDF2_COEFF * e_be, rel=1e-12)


if __name__ == "__main__":
    test_kinetic_be_energy_zero_at_target_and_increases()
    test_kinetic_be_gradient_matches_fd()
    test_kinetic_be_hessian_matches_fd()
    test_kinetic_bdf2_energy_zero_at_target_and_increases()
    test_kinetic_bdf2_gradient_matches_fd()
    test_kinetic_bdf2_hessian_matches_fd()
    test_kinetic_bdf2_scales_be_by_nine_quarters()