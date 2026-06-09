"""Tests for ``simkit.equillibrium_variation_fd``."""

from __future__ import annotations

import numpy as np
import scipy as sp

from simkit.equillibrium_variation_fd import equillibrium_variation_fd
from simkit.variation_fd import variation_fd


def _spd_linear_H(rng, n, dim):
    """Build H(x) = H_const + sum_k x_k * S_k with symmetric pieces."""
    H_const = rng.standard_normal((n, n))
    H_const = H_const @ H_const.T + n * np.eye(n)
    S = rng.standard_normal((dim, n, n))
    S = 0.5 * (S + np.transpose(S, (0, 2, 1)))

    def H_func(x):
        return H_const + np.einsum("k,kij->ij", x, S)

    return H_func


def test_equillibrium_variation_fd_matches_closed_form() -> None:
    # phi_i = H0^{-1} b0 and dphi = H0^{-1} (db - (dH:phi_j) @ phi_i).
    rng = np.random.default_rng(0)
    n, dim = 4, 3

    H_func = _spd_linear_H(rng, n, dim)
    B = rng.standard_normal((dim, n))
    b0 = rng.standard_normal(n)

    def b_func(x):
        return b0 + B.T @ x

    U0 = 0.05 * rng.standard_normal(dim)
    phi_j = rng.standard_normal(dim)

    dphi = equillibrium_variation_fd(U0, phi_j, H_func, b_func, epsilon=1e-5)

    H0 = H_func(U0)
    phi_i = np.linalg.solve(H0, b_func(U0))
    dH = variation_fd(U0, phi_j, H_func, epsilon=1e-5)
    db = variation_fd(U0, phi_j, b_func, epsilon=1e-5)
    expected = np.linalg.solve(H0, db - dH @ phi_i)

    assert dphi.shape == (n,)
    np.testing.assert_allclose(dphi, expected, rtol=1e-8, atol=1e-10)


def test_equillibrium_variation_fd_sparse_H_matches_dense() -> None:
    # A sparse H (routed through spsolve) must match the dense computation.
    rng = np.random.default_rng(4)
    n, dim = 5, 4

    dense_H = _spd_linear_H(rng, n, dim)
    B = rng.standard_normal((dim, n))
    b0 = rng.standard_normal(n)

    def b_func(x):
        return b0 + B.T @ x

    def H_sparse(x):
        return sp.sparse.csr_matrix(dense_H(x))

    U0 = 0.05 * rng.standard_normal(dim)
    phi_j = rng.standard_normal(dim)

    dphi_sparse = equillibrium_variation_fd(U0, phi_j, H_sparse, b_func)
    dphi_dense = equillibrium_variation_fd(U0, phi_j, dense_H, b_func)

    assert dphi_sparse.shape == (n,)
    np.testing.assert_allclose(dphi_sparse, dphi_dense, rtol=1e-8, atol=1e-10)


def test_equillibrium_variation_fd_satisfies_differentiated_equilibrium() -> None:
    # dphi must satisfy: (dH:phi_j) @ phi_i + H0 @ dphi = db:phi_j.
    rng = np.random.default_rng(1)
    n, dim = 5, 4

    H_func = _spd_linear_H(rng, n, dim)
    B = rng.standard_normal((dim, n))

    def b_func(x):
        return B.T @ x

    U0 = 0.05 * rng.standard_normal(dim)
    phi_j = rng.standard_normal(dim)

    dphi = equillibrium_variation_fd(U0, phi_j, H_func, b_func, epsilon=1e-5)

    H0 = H_func(U0)
    phi_i = np.linalg.solve(H0, b_func(U0))
    dH = variation_fd(U0, phi_j, H_func, epsilon=1e-5)
    db = variation_fd(U0, phi_j, b_func, epsilon=1e-5)

    residual = dH @ phi_i + H0 @ dphi - db
    np.testing.assert_allclose(residual, np.zeros(n), atol=1e-8)


def test_equillibrium_variation_fd_matches_total_finite_difference() -> None:
    # The analytic-style result must match a direct central difference of the
    # full equilibrium solution phi_i(x) = H(x)^{-1} b(x) along phi_j.
    rng = np.random.default_rng(2)
    n, dim = 4, 4

    H_func = _spd_linear_H(rng, n, dim)
    B = rng.standard_normal((dim, n))
    b0 = rng.standard_normal(n)

    def b_func(x):
        return b0 + B.T @ x

    def phi_i_of_x(x):
        return np.linalg.solve(H_func(x), b_func(x))

    U0 = 0.05 * rng.standard_normal(dim)
    phi_j = rng.standard_normal(dim)
    eps = 1e-6

    dphi = equillibrium_variation_fd(U0, phi_j, H_func, b_func, epsilon=1e-5)
    dphi_fd = (phi_i_of_x(U0 + eps * phi_j) - phi_i_of_x(U0 - eps * phi_j)) / (2 * eps)

    np.testing.assert_allclose(dphi, dphi_fd, rtol=1e-5, atol=1e-7)


def test_equillibrium_variation_fd_constant_rhs_drops_db_term() -> None:
    # A constant b makes db:phi_j ~ 0, so dphi -> -H0^{-1} (dH:phi_j) @ phi_i.
    rng = np.random.default_rng(3)
    n, dim = 3, 3

    H_func = _spd_linear_H(rng, n, dim)
    b0 = rng.standard_normal(n)

    def b_func(x):
        return b0

    U0 = 0.05 * rng.standard_normal(dim)
    phi_j = rng.standard_normal(dim)

    dphi = equillibrium_variation_fd(U0, phi_j, H_func, b_func)

    H0 = H_func(U0)
    phi_i = np.linalg.solve(H0, b0)
    dH = variation_fd(U0, phi_j, H_func, epsilon=1e-5)
    expected = np.linalg.solve(H0, -dH @ phi_i)

    np.testing.assert_allclose(dphi, expected, rtol=1e-8, atol=1e-10)
