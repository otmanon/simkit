"""Tests for ``simkit.energies.membrane_neo_hookean``.

Membrane Neo-Hookean: each element has a 3D ambient position but a 2D
parametric (rest) domain. The deformation gradient ``F`` is therefore
``(t, 3, 2)``. At rest, ``F = [[1,0],[0,1],[0,0]]`` so the right
Cauchy-Green tensor ``C = F^T F`` is the 2x2 identity and the energy is
zero.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

from simkit.energies.membrane_neo_hookean import (
    membrane_neo_hookean_energy_element_F,
    membrane_neo_hookean_gradient_element_F,
    membrane_neo_hookean_hessian_element_F,
)
from simkit.gradient_cfd import gradient_cfd


FD_STEP = 1e-6
GRAD_TOL = 1e-5
HESS_TOL = 1e-4


def _rest_membrane_F(t: int) -> np.ndarray:
    F = np.zeros((t, 3, 2))
    F[:, 0, 0] = 1.0
    F[:, 1, 1] = 1.0
    return F


def _rest_and_perturbed(rng: np.random.Generator, t: int):
    F_rest = _rest_membrane_F(t)
    F_def = F_rest + 0.05 * rng.standard_normal(F_rest.shape)
    mu = rng.uniform(0.5, 2.0, size=(t, 1))
    lam = rng.uniform(0.5, 2.0, size=(t, 1))
    vol = rng.uniform(0.5, 1.5, size=(t, 1))
    return F_rest, F_def, mu, lam, vol


def test_membrane_neo_hookean_energy_increases_with_deformation() -> None:
    rng = np.random.default_rng(0)
    F_rest, F_def, mu, lam, vol = _rest_and_perturbed(rng, t=4)

    e_rest = float(membrane_neo_hookean_energy_element_F(F_rest, mu, lam))
    e_def = float(membrane_neo_hookean_energy_element_F(F_def, mu, lam))

    assert e_rest == pytest.approx(0.0, abs=1e-10)
    assert e_def > e_rest


def test_membrane_neo_hookean_gradient_matches_fd() -> None:
    rng = np.random.default_rng(1)
    _, F, mu, lam, vol = _rest_and_perturbed(rng, t=3)
    t = F.shape[0]

    def energy_flat(F_flat: np.ndarray) -> np.ndarray:
        return np.array(
            [
                float(
                    membrane_neo_hookean_energy_element_F(
                        F_flat.reshape(t, 3, 2), mu, lam
                    )
                )
            ]
        )

    g_fd = gradient_cfd(energy_flat, F.flatten(), FD_STEP).reshape(t, 3, 2)
    g = membrane_neo_hookean_gradient_element_F(F, mu, lam).reshape(t, 3, 2)

    assert np.allclose(g, g_fd, atol=GRAD_TOL)


def test_membrane_neo_hookean_hessian_matches_fd() -> None:
    rng = np.random.default_rng(2)
    _, F, mu, lam, vol = _rest_and_perturbed(rng, t=2)
    t = F.shape[0]

    def grad_flat(F_flat: np.ndarray) -> np.ndarray:
        return membrane_neo_hookean_gradient_element_F(
            F_flat.reshape(t, 3, 2), mu, lam, vol
        ).flatten()

    H_fd = gradient_cfd(grad_flat, F.flatten(), FD_STEP)
    H_blocks = membrane_neo_hookean_hessian_element_F(F, mu, lam)
    H = sps.block_diag(H_blocks).toarray()

    assert np.allclose(H, H_fd, atol=HESS_TOL)


if __name__ == "__main__":
    test_membrane_neo_hookean_energy_increases_with_deformation()
    test_membrane_neo_hookean_gradient_matches_fd()
    test_membrane_neo_hookean_hessian_matches_fd()