"""Tests for ``simkit.energies.fcr``.

Fixed-Co-Rotated (FCR) elasticity: ARAP shear plus a determinant penalty.
The gradient and Hessian routines reuse the ARAP element-tier helpers.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

from simkit.energies.fcr import (
    fcr_energy_element_F,
    fcr_gradient_element_F,
    fcr_hessian_element_F,
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
def test_fcr_energy_increases_with_deformation(dim: int) -> None:
    rng = np.random.default_rng(0)
    F_rest, F_def, mu, lam, vol = _rest_and_perturbed(rng, t=4, dim=dim)

    e_rest = float((np.asarray(fcr_energy_element_F(F_rest, mu, lam)) * vol.reshape(-1, 1)).sum())
    e_def = float((np.asarray(fcr_energy_element_F(F_def, mu, lam)) * vol.reshape(-1, 1)).sum())

    assert e_rest == pytest.approx(0.0, abs=1e-10)
    assert e_def > e_rest


@pytest.mark.parametrize("dim", [2, 3])
def test_fcr_gradient_matches_fd(dim: int) -> None:
    rng = np.random.default_rng(1)
    _, F, mu, lam, vol = _rest_and_perturbed(rng, t=3, dim=dim)
    t = F.shape[0]

    def energy_flat(F_flat: np.ndarray) -> np.ndarray:
        return np.array(
            [
                float(
                    np.asarray(
                        fcr_energy_element_F(F_flat.reshape(t, dim, dim), mu, lam)
                    ).sum()
                )
            ]
        )

    g_fd = gradient_cfd(energy_flat, F.flatten(), FD_STEP).reshape(t, dim, dim)
    g = fcr_gradient_element_F(F, mu, lam)

    assert np.allclose(g, g_fd, atol=GRAD_TOL)


@pytest.mark.parametrize("dim", [2, 3])
def test_fcr_hessian_matches_fd(dim: int) -> None:
    rng = np.random.default_rng(2)
    _, F, mu, lam, vol = _rest_and_perturbed(rng, t=2, dim=dim)
    t = F.shape[0]

    def grad_flat(F_flat: np.ndarray) -> np.ndarray:
        return fcr_gradient_element_F(F_flat.reshape(t, dim, dim), mu, lam).flatten()

    H_fd = gradient_cfd(grad_flat, F.flatten(), FD_STEP)
    H_blocks = fcr_hessian_element_F(F, mu, lam)
    H = sps.block_diag(H_blocks).toarray()

    assert np.allclose(H, H_fd, atol=HESS_TOL)


if __name__ == "__main__":
    test_fcr_energy_increases_with_deformation(2)
    test_fcr_gradient_matches_fd(2)
    test_fcr_hessian_matches_fd(2)