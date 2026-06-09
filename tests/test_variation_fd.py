"""Tests for ``simkit.variation_fd``."""

from __future__ import annotations

import numpy as np

from simkit.variation_fd import variation_fd


def test_variation_fd_matches_analytic_linear_hessian() -> None:
    # H(x) = sum_k x_k * A_k, so dH/dx : V = sum_k V_k * A_k exactly.
    rng = np.random.default_rng(0)
    n, dim = 3, 4
    A = rng.standard_normal((dim, n, n))

    def H_func(x):
        return np.einsum("k,kij->ij", x, A)

    U0 = rng.standard_normal(dim)
    V = rng.standard_normal(dim)

    dH = variation_fd(U0, V, H_func, epsilon=1e-5)
    analytic = np.einsum("k,kij->ij", V, A)

    assert dH.shape == (n, n)
    np.testing.assert_allclose(dH, analytic, atol=1e-7)


def test_variation_fd_is_central_difference() -> None:
    # For a nonlinear H the result must equal the literal central difference.
    rng = np.random.default_rng(1)
    n = 2

    def H_func(x):
        s = float(np.sum(x ** 2))
        return np.array([[np.sin(s), s], [s, np.cos(s)]])

    U0 = rng.standard_normal(3)
    V = rng.standard_normal(3)
    eps = 1e-4

    dH = variation_fd(U0, V, H_func, epsilon=eps)
    expected = (H_func(U0 + eps * V) - H_func(U0 - eps * V)) / (2 * eps)

    np.testing.assert_allclose(dH, expected, rtol=0, atol=0)


def test_variation_fd_linear_in_direction() -> None:
    # The directional derivative is linear in V: scaling V scales the result.
    rng = np.random.default_rng(2)
    n, dim = 3, 5
    A = rng.standard_normal((dim, n, n))

    def H_func(x):
        return np.einsum("k,kij->ij", x, A)

    U0 = rng.standard_normal(dim)
    V = rng.standard_normal(dim)

    dH = variation_fd(U0, V, H_func)
    dH_scaled = variation_fd(U0, 2.5 * V, H_func)

    np.testing.assert_allclose(dH_scaled, 2.5 * dH, rtol=1e-6)
