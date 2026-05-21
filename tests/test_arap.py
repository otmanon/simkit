"""Tests for ``simkit.energies.arap``.

The ARAP module is the reference implementation of the standardized
three-tier energy layout. We exercise the element tier (``*_element_F``
and ``*_element_S``), the global explicit tier (``*_x``, ``*_S``), and the
self-contained tier (``arap_energy`` / ``arap_gradient`` / ``arap_hessian``)
with finite-difference checks against the analytic gradient and Hessian.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

from simkit.energies.arap import (
    arap_energy,
    arap_energy_S,
    arap_energy_element_F,
    arap_energy_element_S,
    arap_energy_x,
    arap_gradient,
    arap_gradient_S,
    arap_gradient_element_F,
    arap_gradient_element_S,
    arap_gradient_x,
    arap_hessian,
    arap_hessian_S,
    arap_hessian_element_F,
    arap_hessian_element_S,
    arap_hessian_x,
)
from simkit.deformation_jacobian import deformation_jacobian
from simkit.gradient_cfd import gradient_cfd
from simkit.volume import volume


FD_STEP = 1e-6
GRAD_TOL = 1e-5
HESS_TOL = 1e-4


def _unit_simplex(dim: int):
    """Return a single rest simplex (X, T) in ``dim`` dimensions."""
    if dim == 2:
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        T = np.array([[0, 1, 2]])
    elif dim == 3:
        X = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        T = np.array([[0, 1, 2, 3]])
    else:
        raise ValueError(dim)
    return X, T


def _random_def_F(rng: np.random.Generator, t: int, dim: int) -> np.ndarray:
    """Identity-plus-small-perturbation deformation gradients."""
    F = np.tile(np.eye(dim)[None, :, :], (t, 1, 1))
    F = F + 0.1 * rng.standard_normal((t, dim, dim))
    return F


# --------------------------------------------------------------------------- #
# Element tier: F                                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dim", [2, 3])
def test_arap_element_F_energy_increases_with_deformation(dim: int) -> None:
    rng = np.random.default_rng(0)
    t = 5
    mu = rng.uniform(0.5, 2.0, size=(t, 1))

    F_rest = np.tile(np.eye(dim)[None, :, :], (t, 1, 1))
    F_def = _random_def_F(rng, t, dim)

    psi_rest = arap_energy_element_F(F_rest, mu)
    psi_def = arap_energy_element_F(F_def, mu)

    assert np.allclose(psi_rest, 0.0)
    assert np.all(psi_def >= 0.0)
    assert psi_def.sum() > psi_rest.sum()


@pytest.mark.parametrize("dim", [2, 3])
def test_arap_element_F_gradient_matches_fd(dim: int) -> None:
    rng = np.random.default_rng(1)
    t = 3
    mu = rng.uniform(0.5, 2.0, size=(t, 1))
    F = _random_def_F(rng, t, dim)

    def energy_flat(F_flat: np.ndarray) -> np.ndarray:
        psi = arap_energy_element_F(F_flat.reshape(t, dim, dim), mu)
        return np.array([psi.sum()])

    g_fd = gradient_cfd(energy_flat, F.flatten(), FD_STEP).reshape(t, dim, dim)
    g = arap_gradient_element_F(F, mu)

    assert np.allclose(g, g_fd, atol=GRAD_TOL)


@pytest.mark.parametrize("dim", [2, 3])
def test_arap_element_F_hessian_matches_fd(dim: int) -> None:
    rng = np.random.default_rng(2)
    t = 1
    mu = rng.uniform(0.5, 2.0, size=(t, 1))
    F = _random_def_F(rng, t, dim)

    def grad_flat(F_flat: np.ndarray) -> np.ndarray:
        return arap_gradient_element_F(F_flat.reshape(t, dim, dim), mu).flatten()

    H_fd = gradient_cfd(grad_flat, F.flatten(), FD_STEP)
    H = arap_hessian_element_F(F, mu)[0]

    assert np.allclose(H, H_fd, atol=HESS_TOL)


# --------------------------------------------------------------------------- #
# Element tier: S                                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dim", [2, 3])
def test_arap_element_S_full_energy_increases_with_deformation(dim: int) -> None:
    rng = np.random.default_rng(3)
    t = 4
    mu = rng.uniform(0.5, 2.0, size=(t, 1))

    S_rest = np.tile(np.eye(dim)[None, :, :], (t, 1, 1))
    P = 0.05 * rng.standard_normal((t, dim, dim))
    S_def = S_rest + 0.5 * (P + P.transpose(0, 2, 1))

    psi_rest = arap_energy_element_S(S_rest, mu)
    psi_def = arap_energy_element_S(S_def, mu)

    assert np.allclose(psi_rest, 0.0)
    assert np.all(psi_def >= 0.0)
    assert psi_def.sum() > psi_rest.sum()


@pytest.mark.parametrize("dim", [2, 3])
def test_arap_element_S_full_gradient_and_hessian_match_fd(dim: int) -> None:
    rng = np.random.default_rng(4)
    t = 2
    mu = rng.uniform(0.5, 2.0, size=(t, 1))
    S = np.tile(np.eye(dim)[None, :, :], (t, 1, 1))
    S = S + 0.1 * rng.standard_normal((t, dim, dim))

    def energy_flat(S_flat: np.ndarray) -> np.ndarray:
        return np.array(
            [arap_energy_element_S(S_flat.reshape(t, dim, dim), mu).sum()]
        )

    g_fd = gradient_cfd(energy_flat, S.flatten(), FD_STEP).reshape(t, dim, dim)
    g = arap_gradient_element_S(S, mu)
    assert np.allclose(g, g_fd, atol=GRAD_TOL)

    def grad_flat(S_flat: np.ndarray) -> np.ndarray:
        return arap_gradient_element_S(S_flat.reshape(t, dim, dim), mu).flatten()

    H_fd = gradient_cfd(grad_flat, S.flatten(), FD_STEP)
    H_blocks = arap_hessian_element_S(S, mu)
    H = sps.block_diag(H_blocks).toarray()
    assert np.allclose(H, H_fd, atol=HESS_TOL)


# --------------------------------------------------------------------------- #
# Global explicit tier: positions x                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dim", [2, 3])
def test_arap_energy_x_increases_with_deformation(dim: int) -> None:
    rng = np.random.default_rng(5)
    X, T = _unit_simplex(dim)
    J = deformation_jacobian(X, T)
    vol = volume(X, T)
    mu = np.ones((T.shape[0], 1))

    U_def = X + 0.1 * rng.standard_normal(X.shape)

    e_rest = arap_energy_x(X, J, mu, vol)
    e_def = arap_energy_x(U_def, J, mu, vol)

    assert e_rest == pytest.approx(0.0, abs=1e-12)
    assert e_def > e_rest


@pytest.mark.parametrize("dim", [2, 3])
def test_arap_gradient_x_matches_fd(dim: int) -> None:
    rng = np.random.default_rng(6)
    X, T = _unit_simplex(dim)
    J = deformation_jacobian(X, T)
    vol = volume(X, T)
    mu = np.ones((T.shape[0], 1))

    U = X + 0.1 * rng.standard_normal(X.shape)

    def energy_flat(x_flat: np.ndarray) -> np.ndarray:
        return np.array(
            [arap_energy_x(x_flat.reshape(-1, dim), J, mu, vol)]
        )

    g_fd = gradient_cfd(energy_flat, U.flatten(), FD_STEP).flatten()
    g = arap_gradient_x(U, J, mu, vol).flatten()

    assert np.allclose(g, g_fd, atol=GRAD_TOL)


@pytest.mark.parametrize("dim", [2, 3])
def test_arap_hessian_x_matches_fd(dim: int) -> None:
    rng = np.random.default_rng(7)
    X, T = _unit_simplex(dim)
    J = deformation_jacobian(X, T)
    vol = volume(X, T)
    mu = np.ones((T.shape[0], 1))

    U = X + 0.1 * rng.standard_normal(X.shape)

    def grad_flat(x_flat: np.ndarray) -> np.ndarray:
        return arap_gradient_x(x_flat.reshape(-1, dim), J, mu, vol).flatten()

    H_fd = gradient_cfd(grad_flat, U.flatten(), FD_STEP)
    H = arap_hessian_x(U, J, mu, vol, psd=False).toarray()

    assert np.allclose(H, H_fd, atol=HESS_TOL)


# --------------------------------------------------------------------------- #
# Global explicit tier: stretch S                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dim", [2, 3])
def test_arap_energy_S_increases_with_deformation(dim: int) -> None:
    rng = np.random.default_rng(8)
    t = 3
    mu = np.ones((t, 1))
    vol = np.ones((t, 1))

    S_rest = np.tile(np.eye(dim)[None, :, :], (t, 1, 1))
    P = 0.05 * rng.standard_normal((t, dim, dim))
    S_def = S_rest + 0.5 * (P + P.transpose(0, 2, 1))

    e_rest = arap_energy_S(S_rest, mu, vol)
    e_def = arap_energy_S(S_def, mu, vol)

    assert e_rest == pytest.approx(0.0, abs=1e-12)
    assert e_def > e_rest


@pytest.mark.parametrize("dim", [2, 3])
def test_arap_gradient_and_hessian_S_match_fd(dim: int) -> None:
    rng = np.random.default_rng(9)
    t = 2
    mu = np.ones((t, 1))
    vol = np.ones((t, 1))

    S = np.tile(np.eye(dim)[None, :, :], (t, 1, 1))
    S = S + 0.1 * rng.standard_normal((t, dim, dim))

    def energy_flat(S_flat: np.ndarray) -> np.ndarray:
        return np.array(
            [arap_energy_S(S_flat.reshape(t, dim, dim), mu, vol)]
        )

    g_fd = gradient_cfd(energy_flat, S.flatten(), FD_STEP).flatten()
    g = arap_gradient_S(S, mu, vol).flatten()
    assert np.allclose(g, g_fd, atol=GRAD_TOL)

    def grad_flat(S_flat: np.ndarray) -> np.ndarray:
        return arap_gradient_S(S_flat.reshape(t, dim, dim), mu, vol).flatten()

    H_fd = gradient_cfd(grad_flat, S.flatten(), FD_STEP)
    H = arap_hessian_S(S, mu, vol).toarray()
    assert np.allclose(H, H_fd, atol=HESS_TOL)


# --------------------------------------------------------------------------- #
# Self-contained tier                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dim", [2, 3])
def test_arap_self_contained_energy_increases(dim: int) -> None:
    rng = np.random.default_rng(10)
    X, T = _unit_simplex(dim)
    mu = np.ones((T.shape[0], 1))
    U = X + 0.1 * rng.standard_normal(X.shape)

    e_rest = arap_energy(X, T, mu)
    e_def = arap_energy(X, T, mu, U=U)

    assert e_rest == pytest.approx(0.0, abs=1e-12)
    assert e_def > e_rest


@pytest.mark.parametrize("dim", [2, 3])
def test_arap_self_contained_gradient_matches_fd(dim: int) -> None:
    rng = np.random.default_rng(11)
    X, T = _unit_simplex(dim)
    mu = np.ones((T.shape[0], 1))
    U = X + 0.1 * rng.standard_normal(X.shape)

    def energy_flat(u_flat: np.ndarray) -> np.ndarray:
        return np.array([arap_energy(X, T, mu, U=u_flat.reshape(-1, dim))])

    g_fd = gradient_cfd(energy_flat, U.flatten(), FD_STEP).flatten()
    g = arap_gradient(X, T, mu, U=U).flatten()

    assert np.allclose(g, g_fd, atol=GRAD_TOL)


@pytest.mark.parametrize("dim", [2, 3])
def test_arap_self_contained_hessian_matches_fd(dim: int) -> None:
    rng = np.random.default_rng(12)
    X, T = _unit_simplex(dim)
    mu = np.ones((T.shape[0], 1))
    U = X + 0.1 * rng.standard_normal(X.shape)

    def grad_flat(u_flat: np.ndarray) -> np.ndarray:
        return arap_gradient(X, T, mu, U=u_flat.reshape(-1, dim)).flatten()

    H_fd = gradient_cfd(grad_flat, U.flatten(), FD_STEP)
    H = arap_hessian(X, T, mu, U=U, psd=False).toarray()

    assert np.allclose(H, H_fd, atol=HESS_TOL)


if __name__ == "__main__":
    test_arap_element_F_energy_increases_with_deformation(2)
    test_arap_element_F_gradient_matches_fd(2)
    test_arap_element_F_hessian_matches_fd(2)
    test_arap_element_S_full_energy_increases_with_deformation(2)
    test_arap_element_S_full_gradient_and_hessian_match_fd(2)
    test_arap_energy_x_increases_with_deformation(2)
    test_arap_gradient_x_matches_fd(2)
    test_arap_hessian_x_matches_fd(2)
    test_arap_energy_S_increases_with_deformation(2)
    test_arap_gradient_and_hessian_S_match_fd(2)
    test_arap_self_contained_energy_increases(2)