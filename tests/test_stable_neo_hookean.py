"""Tests for ``simkit.energies.stable_neo_hookean``.

Smith-Goes-Kim stable Neo-Hookean (TOG 2018):
https://dl.acm.org/doi/10.1145/3180491. The element-tier functions take
per-element ``F`` and material parameters and return the energy density /
PK1 stress / per-element Hessian blocks. The tests below check:

1. The rest state (``F = I``) is a stationary point of the energy: the
   analytic gradient vanishes, and a small random perturbation strictly
   increases the energy.
2. The analytic gradient matches a central finite-difference of the energy.
3. The analytic per-element Hessian matches a central finite-difference of
   the gradient when assembled block-diagonally.
4. Energy stays bounded as the element collapses to zero volume (this is
   the "stable" property: unlike the standard NH which uses ``-mu log J``,
   the SGK energy is finite at degenerate configurations).
5. Energy explodes as a vertex is dragged to infinity (``I_C -> infty``).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

from simkit.energies.stable_neo_hookean import (
    stable_neo_hookean_energy,
    stable_neo_hookean_energy_element_F,
    stable_neo_hookean_gradient_element_F,
    stable_neo_hookean_hessian_element_F,
)
from simkit.gradient_cfd import gradient_cfd


FD_STEP = 1e-6
GRAD_TOL = 1e-5
HESS_TOL = 1e-4


def _rest_and_perturbed(rng: np.random.Generator, t: int, dim: int):
    F_rest = np.tile(np.eye(dim)[None, :, :], (t, 1, 1))
    F_def = F_rest + 0.05 * rng.standard_normal((t, dim, dim))
    mu = rng.uniform(0.5, 2.0, size=(t, 1))
    lam = rng.uniform(0.5, 2.0, size=(t, 1))
    vol = rng.uniform(0.5, 1.5, size=(t, 1))
    return F_rest, F_def, mu, lam, vol


@pytest.mark.parametrize("dim", [2, 3])
def test_stable_neo_hookean_rest_is_minimum(dim: int) -> None:
    rng = np.random.default_rng(0)
    F_rest, F_def, mu, lam, _ = _rest_and_perturbed(rng, t=4, dim=dim)

    e_rest = float(stable_neo_hookean_energy_element_F(F_rest, mu, lam).sum().item())
    e_def = float(stable_neo_hookean_energy_element_F(F_def, mu, lam).sum().item())

    # rest energy is finite and not NaN (it is non-zero -- a feature, not a bug)
    assert np.isfinite(e_rest)
    # rest is a minimum: small perturbation strictly increases the energy
    assert e_def > e_rest

    # rest is also a stationary point: gradient at F=I is (numerically) zero
    g_rest = stable_neo_hookean_gradient_element_F(F_rest, mu, lam)
    assert np.max(np.abs(g_rest)) < 1e-12


@pytest.mark.parametrize("dim", [2, 3])
def test_stable_neo_hookean_gradient_matches_fd(dim: int) -> None:
    rng = np.random.default_rng(1)
    _, F, mu, lam, _ = _rest_and_perturbed(rng, t=3, dim=dim)
    t = F.shape[0]

    def energy_flat(F_flat: np.ndarray) -> np.ndarray:
        return np.array(
            [
                float(
                    stable_neo_hookean_energy_element_F(
                        F_flat.reshape(t, dim, dim), mu, lam
                    ).sum().item()
                )
            ]
        )

    g_fd = gradient_cfd(energy_flat, F.flatten(), FD_STEP).reshape(t, dim, dim)
    g = stable_neo_hookean_gradient_element_F(F, mu, lam)

    assert np.allclose(g, g_fd, atol=GRAD_TOL)


@pytest.mark.parametrize("dim", [2, 3])
def test_stable_neo_hookean_hessian_matches_fd(dim: int) -> None:
    rng = np.random.default_rng(2)
    _, F, mu, lam, _ = _rest_and_perturbed(rng, t=2, dim=dim)
    t = F.shape[0]

    def grad_flat(F_flat: np.ndarray) -> np.ndarray:
        return stable_neo_hookean_gradient_element_F(
            F_flat.reshape(t, dim, dim), mu, lam
        ).flatten()

    H_fd = gradient_cfd(grad_flat, F.flatten(), FD_STEP)
    H_blocks = stable_neo_hookean_hessian_element_F(F, mu, lam)
    H = sps.block_diag(H_blocks).toarray()

    assert np.allclose(H, H_fd, atol=HESS_TOL)


def _unit_triangle():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    T = np.array([[0, 1, 2]])
    mu = np.array([[1.0]])
    lam = np.array([[1.0]])
    return X, T, mu, lam


def _unit_tet():
    X = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    T = np.array([[0, 1, 2, 3]])
    mu = np.array([[1.0]])
    lam = np.array([[1.0]])
    return X, T, mu, lam


@pytest.mark.parametrize(
    "geom_factory", [_unit_triangle, _unit_tet], ids=["triangle", "tet"]
)
def test_stable_neo_hookean_collapse_remains_bounded(geom_factory) -> None:
    """The "stable" property: energy is finite as the element collapses to zero volume."""
    X, T, mu, lam = geom_factory()

    # uniform shrink toward F = 0
    scales = [1.0, 0.5, 1e-1, 1e-3, 1e-6, 0.0]
    energies = np.array(
        [stable_neo_hookean_energy(X, T, mu, lam, s * X) for s in scales]
    )
    assert np.all(np.isfinite(energies)), f"energy blew up under uniform shrink: {energies}"

    # shear collapse: squash only the last coordinate
    dim = X.shape[1]
    energies_shear = []
    for s in [1.0, 0.5, 1e-1, 1e-3, 1e-6, 0.0]:
        U = X.copy()
        U[:, dim - 1] *= s
        energies_shear.append(stable_neo_hookean_energy(X, T, mu, lam, U))
    energies_shear = np.array(energies_shear)
    assert np.all(np.isfinite(energies_shear)), (
        f"energy blew up under shear collapse: {energies_shear}"
    )


@pytest.mark.parametrize(
    "geom_factory", [_unit_triangle, _unit_tet], ids=["triangle", "tet"]
)
def test_stable_neo_hookean_stretch_explodes(geom_factory) -> None:
    """Dragging a vertex to infinity drives I_C -> inf, so energy -> inf."""
    X, T, mu, lam = geom_factory()

    scales = [1.0, 10.0, 1e3, 1e6]
    energies = np.array(
        [stable_neo_hookean_energy(X, T, mu, lam, _scaled_vertex(X, -1, s)) for s in scales]
    )
    # monotonically non-decreasing with stretch
    assert np.all(np.diff(energies) > 0), f"energy not monotonic in stretch: {energies}"
    # energy grows at least as fast as I_C ~ s^2, so E(s=1e6) >> E(s=1)
    assert energies[-1] > 1e6 * abs(energies[0]) + 1.0


def _scaled_vertex(X: np.ndarray, idx: int, s: float) -> np.ndarray:
    U = X.copy()
    U[idx] = U[idx] * s
    return U
