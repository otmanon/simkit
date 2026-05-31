"""Tests for ``simkit.energies.stvk``.

Saint Venant-Kirchhoff energy, classical form from the FEM-deformables
course notes (Sifakis & Barbic, http://barbic.usc.edu/femdefo/):

    psi(F) = mu * tr(E**2) + (lam / 2) * tr(E)**2,   E = (F^T F - I) / 2

Tests:

1. Energy is zero at rest (``F = I``) and strictly increases under a small
   random perturbation.
2. Analytic gradient matches a central finite-difference of the energy.
3. Analytic per-element Hessian matches a central finite-difference of the
   gradient when assembled block-diagonally.
4. Inversion blind-spot: confirm that ``psi(F) == psi(-F)`` (the documented
   StVK weakness; included as a regression guard so a future "fix" can't
   silently change this without updating the docstring).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

from simkit.energies.stvk import (
    stvk_energy_element_F,
    stvk_gradient_element_F,
    stvk_hessian_element_F,
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
def test_stvk_energy_increases_with_deformation(dim: int) -> None:
    rng = np.random.default_rng(0)
    F_rest, F_def, mu, lam, _ = _rest_and_perturbed(rng, t=4, dim=dim)

    e_rest = float(stvk_energy_element_F(F_rest, mu, lam).sum().item())
    e_def = float(stvk_energy_element_F(F_def, mu, lam).sum().item())

    assert e_rest == pytest.approx(0.0, abs=1e-10)
    assert e_def > e_rest

    # gradient also vanishes at rest
    g_rest = stvk_gradient_element_F(F_rest, mu, lam)
    assert np.max(np.abs(g_rest)) < 1e-12


@pytest.mark.parametrize("dim", [2, 3])
def test_stvk_gradient_matches_fd(dim: int) -> None:
    rng = np.random.default_rng(1)
    _, F, mu, lam, _ = _rest_and_perturbed(rng, t=3, dim=dim)
    t = F.shape[0]

    def energy_flat(F_flat: np.ndarray) -> np.ndarray:
        return np.array(
            [
                float(
                    stvk_energy_element_F(
                        F_flat.reshape(t, dim, dim), mu, lam
                    ).sum().item()
                )
            ]
        )

    g_fd = gradient_cfd(energy_flat, F.flatten(), FD_STEP).reshape(t, dim, dim)
    g = stvk_gradient_element_F(F, mu, lam)

    assert np.allclose(g, g_fd, atol=GRAD_TOL)


@pytest.mark.parametrize("dim", [2, 3])
def test_stvk_hessian_matches_fd(dim: int) -> None:
    rng = np.random.default_rng(2)
    _, F, mu, lam, _ = _rest_and_perturbed(rng, t=2, dim=dim)
    t = F.shape[0]

    def grad_flat(F_flat: np.ndarray) -> np.ndarray:
        return stvk_gradient_element_F(
            F_flat.reshape(t, dim, dim), mu, lam
        ).flatten()

    H_fd = gradient_cfd(grad_flat, F.flatten(), FD_STEP)
    H_blocks = stvk_hessian_element_F(F, mu, lam)
    H = sps.block_diag(H_blocks).toarray()

    assert np.allclose(H, H_fd, atol=HESS_TOL)


@pytest.mark.parametrize("dim", [2, 3])
def test_stvk_is_blind_to_inversion(dim: int) -> None:
    """StVK depends on F only through F^T F so psi(F) == psi(-F).

    This is the well-known weakness of StVK and the reason
    inversion-robust formulations like Stable Neo-Hookean exist.
    Recorded here as a regression guard.
    """
    rng = np.random.default_rng(3)
    _, F, mu, lam, _ = _rest_and_perturbed(rng, t=4, dim=dim)
    e_F = stvk_energy_element_F(F, mu, lam)
    e_negF = stvk_energy_element_F(-F, mu, lam)
    assert np.allclose(e_F, e_negF, atol=1e-12)
