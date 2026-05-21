"""Tests for ``simkit.energies.quadratic``.

Generic quadratic energy ``0.5 * x^T Q x + b^T x``. For a positive
definite ``Q`` the minimizer is ``x* = -Q^{-1} b`` and any perturbation
about ``x*`` strictly increases the energy.
"""

from __future__ import annotations

import numpy as np
import pytest

from simkit.energies.quadratic import (
    quadratic_energy,
    quadratic_gradient,
    quadratic_hessian,
)
from simkit.gradient_cfd import gradient_cfd


FD_STEP = 1e-6
GRAD_TOL = 1e-6
HESS_TOL = 1e-6


def _setup(rng: np.random.Generator, n: int):
    A = rng.standard_normal((n, n))
    Q = A.T @ A + n * np.eye(n)
    b = rng.standard_normal((n, 1))
    return Q, b


def test_quadratic_energy_increases_with_deformation() -> None:
    rng = np.random.default_rng(0)
    Q, b = _setup(rng, n=6)

    x_min = -np.linalg.solve(Q, b)
    delta = 0.1 * rng.standard_normal(x_min.shape)
    x_def = x_min + delta

    e_min = float(np.asarray(quadratic_energy(x_min, Q, b)).item())
    e_def = float(np.asarray(quadratic_energy(x_def, Q, b)).item())

    assert e_def > e_min


def test_quadratic_gradient_matches_fd() -> None:
    rng = np.random.default_rng(1)
    Q, b = _setup(rng, n=5)
    x = rng.standard_normal((Q.shape[0], 1))
    n = x.shape[0]

    def energy_flat(x_flat: np.ndarray) -> np.ndarray:
        return np.array(
            [float(np.asarray(quadratic_energy(x_flat.reshape(n, 1), Q, b)).item())]
        )

    g_fd = gradient_cfd(energy_flat, x.flatten(), FD_STEP).flatten()
    g = np.asarray(quadratic_gradient(x, Q, b)).flatten()

    assert np.allclose(g, g_fd, atol=GRAD_TOL)


def test_quadratic_hessian_matches_fd() -> None:
    rng = np.random.default_rng(2)
    Q, b = _setup(rng, n=4)
    x = rng.standard_normal((Q.shape[0], 1))
    n = x.shape[0]

    def grad_flat(x_flat: np.ndarray) -> np.ndarray:
        return np.asarray(
            quadratic_gradient(x_flat.reshape(n, 1), Q, b)
        ).flatten()

    H_fd = gradient_cfd(grad_flat, x.flatten(), FD_STEP)
    H = np.asarray(quadratic_hessian(Q))

    assert np.allclose(H, H_fd, atol=HESS_TOL)