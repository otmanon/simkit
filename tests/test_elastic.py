"""Tests for ``simkit.energies.elastic`` (the merged dispatch module).

``elastic.py`` combines the former ``elastic_energy``/``elastic_gradient``/
``elastic_hessian`` files. It dispatches on a ``material`` string over the
per-material element tiers, applying ``vol`` and the operator at the global
tiers. These tests check:

1. The element dispatch reproduces the underlying per-material element
   functions (including ARAP's ``mu``-only signature).
2. The ``_element_F`` gradient/Hessian match a finite difference of the
   ``_element_F`` energy for every material.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

from simkit.energies.elastic import (
    elastic_energy_element_F,
    elastic_gradient_element_F,
    elastic_hessian_element_F,
)
from simkit.energies.arap import arap_energy_element_F
from simkit.energies.neo_hookean import neo_hookean_energy_element_F
from simkit.gradient_cfd import gradient_cfd


FD_STEP = 1e-6
GRAD_TOL = 1e-5
HESS_TOL = 1e-4

MATERIALS = ["linear-elasticity", "arap", "fcr", "neo-hookean"]


def _perturbed(rng: np.random.Generator, t: int, dim: int):
    F = np.tile(np.eye(dim)[None, :, :], (t, 1, 1))
    F = F + 0.05 * rng.standard_normal((t, dim, dim))
    mu = rng.uniform(0.5, 2.0, size=(t, 1))
    lam = rng.uniform(0.5, 2.0, size=(t, 1))
    return F, mu, lam


def test_dispatch_matches_underlying_material() -> None:
    rng = np.random.default_rng(0)
    F, mu, lam = _perturbed(rng, t=4, dim=3)

    # ARAP ignores lam internally; dispatch should match the mu-only call.
    assert np.allclose(
        elastic_energy_element_F(F, mu, lam, "arap"),
        arap_energy_element_F(F, mu),
    )
    assert np.allclose(
        elastic_energy_element_F(F, mu, lam, "neo-hookean"),
        neo_hookean_energy_element_F(F, mu, lam),
    )


def test_dispatch_unknown_material_raises() -> None:
    rng = np.random.default_rng(0)
    F, mu, lam = _perturbed(rng, t=2, dim=2)
    with pytest.raises(ValueError):
        elastic_energy_element_F(F, mu, lam, "not-a-material")


@pytest.mark.parametrize("material", MATERIALS)
@pytest.mark.parametrize("dim", [2, 3])
def test_elastic_element_gradient_matches_fd(material: str, dim: int) -> None:
    rng = np.random.default_rng(1)
    F, mu, lam = _perturbed(rng, t=3, dim=dim)
    t = F.shape[0]

    def energy_flat(F_flat: np.ndarray) -> np.ndarray:
        return np.array(
            [
                float(
                    np.asarray(
                        elastic_energy_element_F(
                            F_flat.reshape(t, dim, dim), mu, lam, material
                        )
                    ).sum()
                )
            ]
        )

    g_fd = gradient_cfd(energy_flat, F.flatten(), FD_STEP).reshape(t, dim, dim)
    g = elastic_gradient_element_F(F, mu, lam, material)
    assert np.allclose(g, g_fd, atol=GRAD_TOL)


@pytest.mark.parametrize("material", MATERIALS)
@pytest.mark.parametrize("dim", [2, 3])
def test_elastic_element_hessian_matches_fd(material: str, dim: int) -> None:
    rng = np.random.default_rng(2)
    F, mu, lam = _perturbed(rng, t=2, dim=dim)
    t = F.shape[0]

    def grad_flat(F_flat: np.ndarray) -> np.ndarray:
        return elastic_gradient_element_F(
            F_flat.reshape(t, dim, dim), mu, lam, material
        ).flatten()

    H_fd = gradient_cfd(grad_flat, F.flatten(), FD_STEP)
    # psd=False so the analytic Hessian equals the true second derivative.
    H_blocks = elastic_hessian_element_F(F, mu, lam, material, psd=False)
    H = sps.block_diag(H_blocks).toarray()
    assert np.allclose(H, H_fd, atol=HESS_TOL)


if __name__ == "__main__":
    test_dispatch_matches_underlying_material()
    test_dispatch_unknown_material_raises()
    for m in MATERIALS:
        test_elastic_element_gradient_matches_fd(m, 2)
        test_elastic_element_hessian_matches_fd(m, 2)
    
        test_elastic_element_gradient_matches_fd(m, 3)
        test_elastic_element_hessian_matches_fd(m, 3)
    print("All tests passed.")