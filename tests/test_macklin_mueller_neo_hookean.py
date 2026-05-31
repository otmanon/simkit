"""Tests for ``simkit.energies.macklin_mueller_neo_hookean``.

Macklin-Mueller stable Neo-Hookean (quadratic volume-preserving term, MIG
2021): https://dl.acm.org/doi/10.1145/3487983.3488289. The element-tier
functions take per-element ``F`` and material parameters and return the
energy density / PK1 stress / per-element Hessian blocks. The tests below
check:

1. The energy is zero at the rest state and strictly increases under a small
   random perturbation of ``F``.
2. The analytic gradient matches a central finite-difference of the energy.
3. The analytic per-element Hessian matches a central finite-difference of
   the gradient when assembled block-diagonally.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

from simkit.energies.macklin_mueller_neo_hookean import (
    macklin_mueller_neo_hookean_energy_element_F,
    macklin_mueller_neo_hookean_gradient_element_F,
    macklin_mueller_neo_hookean_hessian_element_F,
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
def test_macklin_mueller_neo_hookean_energy_increases_with_deformation(dim: int) -> None:
    rng = np.random.default_rng(0)
    F_rest, F_def, mu, lam, vol = _rest_and_perturbed(rng, t=4, dim=dim)

    e_rest = float(macklin_mueller_neo_hookean_energy_element_F(F_rest, mu, lam).sum().item())
    e_def = float(macklin_mueller_neo_hookean_energy_element_F(F_def, mu, lam).sum().item())

    assert e_rest == pytest.approx(0.0, abs=1e-10)
    assert e_def > e_rest


@pytest.mark.parametrize("dim", [2, 3])
def test_macklin_mueller_neo_hookean_gradient_matches_fd(dim: int) -> None:
    rng = np.random.default_rng(1)
    _, F, mu, lam, vol = _rest_and_perturbed(rng, t=3, dim=dim)
    t = F.shape[0]

    def energy_flat(F_flat: np.ndarray) -> np.ndarray:
        return np.array(
            [
                float(
                    macklin_mueller_neo_hookean_energy_element_F(
                        F_flat.reshape(t, dim, dim), mu, lam
                    ).sum().item()
                )
            ]
        )

    g_fd = gradient_cfd(energy_flat, F.flatten(), FD_STEP).reshape(t, dim, dim)
    g = macklin_mueller_neo_hookean_gradient_element_F(F, mu, lam)

    assert np.allclose(g, g_fd, atol=GRAD_TOL)


@pytest.mark.parametrize("dim", [2, 3])
def test_macklin_mueller_neo_hookean_hessian_matches_fd(dim: int) -> None:
    rng = np.random.default_rng(2)
    _, F, mu, lam, vol = _rest_and_perturbed(rng, t=2, dim=dim)
    t = F.shape[0]

    def grad_flat(F_flat: np.ndarray) -> np.ndarray:
        return macklin_mueller_neo_hookean_gradient_element_F(
            F_flat.reshape(t, dim, dim), mu, lam
        ).flatten()

    H_fd = gradient_cfd(grad_flat, F.flatten(), FD_STEP)
    H_blocks = macklin_mueller_neo_hookean_hessian_element_F(F, mu, lam)
    H = sps.block_diag(H_blocks).toarray()

    assert np.allclose(H, H_fd, atol=HESS_TOL)
