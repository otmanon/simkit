"""Tests for the stretch (``S``) tier of ``macklin_mueller_neo_hookean``.

The Macklin-Mueller Neo-Hookean density is isotropic, so for a symmetric
stretch ``S`` the energy equals ``psi(F)`` at ``F = S``. The ``_S`` tier exposes
this energy in the ``d(d+1)/2`` independent stretch components used by the mixed
(MFEM) solver. The tests check:

1. Energy is zero at the rest stretch and increases under a small perturbation.
2. The compact-component energy agrees with the full-matrix ``F`` tier.
3. The analytic compact gradient matches a central finite difference.
4. The analytic compact Hessian (``psd=False``) matches a finite difference of
   the gradient.
5. The ``elastic_*_S`` dispatch routes ``'macklin-mueller-neo-hookean'`` here
   and applies the quadrature weights.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

from simkit.energies.macklin_mueller_neo_hookean import (
    macklin_mueller_neo_hookean_energy_element_F,
    macklin_mueller_neo_hookean_energy_element_S,
    macklin_mueller_neo_hookean_gradient_element_S,
    macklin_mueller_neo_hookean_hessian_element_S,
)
from simkit.energies.elastic import elastic_energy_S, elastic_gradient_S, elastic_hessian_S
from simkit.symmetric_stretch_map import symmetric_stretch_map
from simkit.gradient_cfd import gradient_cfd


FD_STEP = 1e-6
GRAD_TOL = 1e-5
HESS_TOL = 1e-4


def _rest_compact(t: int, dim: int) -> np.ndarray:
    k = dim * (dim + 1) // 2
    rest = np.zeros((t, k))
    rest[:, :dim] = 1.0   # diagonal stretches are 1 at rest, off-diagonals 0
    return rest


def _rest_and_perturbed(rng: np.random.Generator, t: int, dim: int):
    rest = _rest_compact(t, dim)
    Sc = rest + 0.05 * rng.standard_normal(rest.shape)   # symmetric, near identity
    mu = rng.uniform(0.5, 2.0, size=(t, 1))
    lam = rng.uniform(0.5, 2.0, size=(t, 1))
    vol = rng.uniform(0.5, 1.5, size=(t, 1))
    return rest, Sc, mu, lam, vol


@pytest.mark.parametrize("dim", [2, 3])
def test_energy_zero_at_rest_and_increases(dim: int) -> None:
    rng = np.random.default_rng(0)
    rest, Sc, mu, lam, vol = _rest_and_perturbed(rng, t=4, dim=dim)

    e_rest = float(macklin_mueller_neo_hookean_energy_element_S(rest, mu, lam).sum())
    e_def = float(macklin_mueller_neo_hookean_energy_element_S(Sc, mu, lam).sum())

    assert e_rest == pytest.approx(0.0, abs=1e-10)
    assert e_def > e_rest


@pytest.mark.parametrize("dim", [2, 3])
def test_compact_energy_matches_full_matrix(dim: int) -> None:
    rng = np.random.default_rng(7)
    _, Sc, mu, lam, _ = _rest_and_perturbed(rng, t=5, dim=dim)
    # embed compact components into full symmetric matrices and compare
    Se, _ = symmetric_stretch_map(1, dim)
    C0 = np.asarray(Se.todense())
    Sf = (Sc @ C0.T).reshape(-1, dim, dim)

    e_compact = macklin_mueller_neo_hookean_energy_element_S(Sc, mu, lam)
    e_full = macklin_mueller_neo_hookean_energy_element_F(Sf, mu, lam)
    assert np.allclose(e_compact, e_full)


@pytest.mark.parametrize("dim", [2, 3])
def test_gradient_matches_fd(dim: int) -> None:
    rng = np.random.default_rng(1)
    _, Sc, mu, lam, _ = _rest_and_perturbed(rng, t=3, dim=dim)
    t, k = Sc.shape

    def energy_flat(s_flat: np.ndarray) -> np.ndarray:
        return np.array([float(
            macklin_mueller_neo_hookean_energy_element_S(s_flat.reshape(t, k), mu, lam).sum()
        )])

    g_fd = gradient_cfd(energy_flat, Sc.flatten(), FD_STEP).reshape(t, k)
    g = macklin_mueller_neo_hookean_gradient_element_S(Sc, mu, lam)
    assert np.allclose(g, g_fd, atol=GRAD_TOL)


@pytest.mark.parametrize("dim", [2, 3])
def test_hessian_matches_fd(dim: int) -> None:
    rng = np.random.default_rng(2)
    _, Sc, mu, lam, _ = _rest_and_perturbed(rng, t=2, dim=dim)
    t, k = Sc.shape

    def grad_flat(s_flat: np.ndarray) -> np.ndarray:
        return macklin_mueller_neo_hookean_gradient_element_S(
            s_flat.reshape(t, k), mu, lam).flatten()

    H_fd = gradient_cfd(grad_flat, Sc.flatten(), FD_STEP)
    # psd=False gives the true (possibly indefinite) analytic Hessian
    H_blocks = macklin_mueller_neo_hookean_hessian_element_S(Sc, mu, lam, psd=False)
    H = sps.block_diag(H_blocks).toarray()
    assert np.allclose(H, H_fd, atol=HESS_TOL)


@pytest.mark.parametrize("dim", [2, 3])
def test_elastic_S_dispatch(dim: int) -> None:
    rng = np.random.default_rng(3)
    _, Sc, mu, lam, vol = _rest_and_perturbed(rng, t=4, dim=dim)
    mat = "macklin-mueller-neo-hookean"

    # energy is the volume-weighted sum of per-element densities
    psi = macklin_mueller_neo_hookean_energy_element_S(Sc, mu, lam)
    assert elastic_energy_S(Sc, mu, lam, vol, mat) == pytest.approx(float((vol * psi).sum()))

    # gradient / hessian are volume-weighted per-element blocks
    g = elastic_gradient_S(Sc, mu, lam, vol, mat)
    g_ref = macklin_mueller_neo_hookean_gradient_element_S(Sc, mu, lam) * vol
    assert np.allclose(g, g_ref)

    H = elastic_hessian_S(Sc, mu, lam, vol, mat)
    H_ref = macklin_mueller_neo_hookean_hessian_element_S(Sc, mu, lam) * vol.reshape(-1, 1, 1)
    assert np.allclose(H, H_ref)
