"""Tests for ``simkit.energies.kinetic``.

The kinetic energy used in implicit Euler integration is
``0.5 * (x - y)^T M (x - y) / h^2``, where ``y`` is the inertial target
(``2 x_curr - x_prev``) and ``h`` is the time step. It is quadratic in
``x``, so the Hessian ``M / h^2`` is independent of ``x``.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

from simkit.energies.kinetic import (
    kinetic_energy,
    kinetic_gradient,
    kinetic_hessian,
)
from simkit.gradient_cfd import gradient_cfd


FD_STEP = 1e-6
GRAD_TOL = 1e-7
HESS_TOL = 1e-7


def _setup(rng: np.random.Generator, n: int):
    A = rng.standard_normal((n, n))
    M = sps.csc_matrix(A.T @ A + n * np.eye(n))  # SPD
    y = rng.standard_normal((n, 1))
    h = 0.02
    return M, y, h


def test_kinetic_energy_increases_with_deformation() -> None:
    rng = np.random.default_rng(0)
    M, y, h = _setup(rng, n=6)

    x_rest = y.copy()
    x_def = y + 0.1 * rng.standard_normal(y.shape)

    e_rest = float(np.asarray(kinetic_energy(x_rest, y, M, h)).item())
    e_def = float(np.asarray(kinetic_energy(x_def, y, M, h)).item())

    assert e_rest == pytest.approx(0.0, abs=1e-12)
    assert e_def > e_rest


def test_kinetic_gradient_matches_fd() -> None:
    rng = np.random.default_rng(1)
    M, y, h = _setup(rng, n=5)
    x = y + 0.1 * rng.standard_normal(y.shape)
    n = x.shape[0]

    def energy_flat(x_flat: np.ndarray) -> np.ndarray:
        return np.array(
            [float(np.asarray(kinetic_energy(x_flat.reshape(n, 1), y, M, h)).item())]
        )

    g_fd = gradient_cfd(energy_flat, x.flatten(), FD_STEP).flatten()
    g = np.asarray(kinetic_gradient(x, y, M, h)).flatten()

    assert np.allclose(g, g_fd, atol=GRAD_TOL)


def test_kinetic_hessian_matches_fd() -> None:
    rng = np.random.default_rng(2)
    M, y, h = _setup(rng, n=4)
    x = y + 0.1 * rng.standard_normal(y.shape)
    n = x.shape[0]

    def grad_flat(x_flat: np.ndarray) -> np.ndarray:
        return np.asarray(
            kinetic_gradient(x_flat.reshape(n, 1), y, M, h)
        ).flatten()

    H_fd = gradient_cfd(grad_flat, x.flatten(), FD_STEP)
    H = np.asarray(kinetic_hessian(M, h).todense())

    assert np.allclose(H, H_fd, atol=HESS_TOL)
