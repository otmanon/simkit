"""Tests for ``simkit.energies.neo_hookean``.

Classical Neo-Hookean energy from the FEM-deformables course notes
(Sifakis & Barbic, http://barbic.usc.edu/femdefo/):

    psi(F) = (mu/2)(I_C - dim) - mu * log(J) + (lam/2) * log(J)**2

Tests:

1. Energy is zero at rest (``F = I``) and strictly increases under a small
   random perturbation.
2. Analytic gradient matches a central finite-difference of the energy.
3. Analytic per-element Hessian matches a central finite-difference of the
   gradient when assembled block-diagonally.
4. Energy diverges to ``+infty`` as the element collapses to zero volume
   (the ``-mu log(J)`` term is the singular term that makes this Neo-Hookean
   variant inversion-fragile -- and is exactly the term replaced by
   quadratic-in-J penalties in the "stable" variants).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

from simkit.energies.neo_hookean import (
    neo_hookean_energy,
    neo_hookean_energy_element_F,
    neo_hookean_gradient_element_F,
    neo_hookean_hessian_element_F,
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
def test_neo_hookean_energy_increases_with_deformation(dim: int) -> None:
    rng = np.random.default_rng(0)
    F_rest, F_def, mu, lam, _ = _rest_and_perturbed(rng, t=4, dim=dim)

    e_rest = float(neo_hookean_energy_element_F(F_rest, mu, lam).sum().item())
    e_def = float(neo_hookean_energy_element_F(F_def, mu, lam).sum().item())

    assert e_rest == pytest.approx(0.0, abs=1e-10)
    assert e_def > e_rest

    # gradient also vanishes at rest
    g_rest = neo_hookean_gradient_element_F(F_rest, mu, lam)
    assert np.max(np.abs(g_rest)) < 1e-12


@pytest.mark.parametrize("dim", [2, 3])
def test_neo_hookean_gradient_matches_fd(dim: int) -> None:
    rng = np.random.default_rng(1)
    _, F, mu, lam, _ = _rest_and_perturbed(rng, t=3, dim=dim)
    t = F.shape[0]

    def energy_flat(F_flat: np.ndarray) -> np.ndarray:
        return np.array(
            [
                float(
                    neo_hookean_energy_element_F(
                        F_flat.reshape(t, dim, dim), mu, lam
                    ).sum().item()
                )
            ]
        )

    g_fd = gradient_cfd(energy_flat, F.flatten(), FD_STEP).reshape(t, dim, dim)
    g = neo_hookean_gradient_element_F(F, mu, lam)

    assert np.allclose(g, g_fd, atol=GRAD_TOL)


@pytest.mark.parametrize("dim", [2, 3])
def test_neo_hookean_hessian_matches_fd(dim: int) -> None:
    rng = np.random.default_rng(2)
    _, F, mu, lam, _ = _rest_and_perturbed(rng, t=2, dim=dim)
    t = F.shape[0]

    def grad_flat(F_flat: np.ndarray) -> np.ndarray:
        return neo_hookean_gradient_element_F(
            F_flat.reshape(t, dim, dim), mu, lam
        ).flatten()

    H_fd = gradient_cfd(grad_flat, F.flatten(), FD_STEP)
    H_blocks = neo_hookean_hessian_element_F(F, mu, lam)
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
def test_neo_hookean_collapse_explodes(geom_factory) -> None:
    """As the element collapses to zero volume the energy diverges to +inf.

    The ``-mu log(J)`` term sends ``psi -> +inf`` as ``J -> 0+``. This is the
    canonical "inversion is unphysical" property of the classical Neo-Hookean
    energy.
    """
    X, T, mu, lam = geom_factory()

    # uniform shrink toward F = 0 (J -> 0+ for s > 0; undefined exactly at s=0).
    # Energy grows as (lam/2)*log(J)^2, which is log^2(1/s) -- unbounded but slow.
    scales = [1.0, 0.5, 0.1, 1e-3, 1e-6, 1e-9, 1e-12]
    with np.errstate(divide="ignore", invalid="ignore"):
        energies = np.array(
            [neo_hookean_energy(X, T, mu, lam, s * X) for s in scales]
        )
    # rest energy is zero, energy grows monotonically as s -> 0+
    assert energies[0] == pytest.approx(0.0, abs=1e-10)
    assert np.all(np.diff(energies) > 0), f"not monotonic under shrink: {energies}"

    # at exactly J = 0, the -mu*log(J) and (lam/2)*log(J)^2 terms diverge,
    # so the energy is +infinity (numerically: +inf or nan).
    with np.errstate(divide="ignore", invalid="ignore"):
        e_zero = neo_hookean_energy(X, T, mu, lam, 0.0 * X)
    assert not np.isfinite(e_zero), (
        f"expected energy to diverge at zero volume, got finite {e_zero}"
    )
