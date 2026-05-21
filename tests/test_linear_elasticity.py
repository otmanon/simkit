"""Tests for ``simkit.energies.linear_elasticity``.

Linear elasticity from the Sifakis course notes (small-strain elasticity).
The energy is quadratic in the symmetric small-strain tensor
``eps = sym(F) - I`` and is *not* rotation invariant. For the
energy-increases-with-deformation check we perturb ``F`` symmetrically
about the rest state to obtain non-zero strain.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

from simkit.energies.linear_elasticity import (
    linear_elasticity_energy_element_F,
    linear_elasticity_gradient_element_F,
    linear_elasticity_hessian_element_F,
)
from simkit.gradient_cfd import gradient_cfd


FD_STEP = 1e-6
GRAD_TOL = 1e-5
HESS_TOL = 1e-4


def _rest_and_perturbed(rng: np.random.Generator, t: int, dim: int):
    F_rest = np.tile(np.eye(dim)[None, :, :], (t, 1, 1))
    P = 0.05 * rng.standard_normal((t, dim, dim))
    F_def = F_rest + 0.5 * (P + P.transpose(0, 2, 1))
    mu = rng.uniform(0.5, 2.0, size=(t, 1))
    lam = rng.uniform(0.5, 2.0, size=(t, 1))
    vol = rng.uniform(0.5, 1.5, size=(t, 1))
    return F_rest, F_def, mu, lam, vol


@pytest.mark.parametrize("dim", [2, 3])
def test_linear_elasticity_energy_increases_with_deformation(dim: int) -> None:
    rng = np.random.default_rng(0)
    F_rest, F_def, mu, lam, vol = _rest_and_perturbed(rng, t=4, dim=dim)

    e_rest = float(linear_elasticity_energy_element_F(F_rest, mu, lam).sum())
    e_def = float(linear_elasticity_energy_element_F(F_def, mu, lam).sum())

    assert e_rest == pytest.approx(0.0, abs=1e-10)
    assert e_def > e_rest


@pytest.mark.parametrize("dim", [2, 3])
def test_linear_elasticity_gradient_matches_fd(dim: int) -> None:
    rng = np.random.default_rng(1)
    _, F, mu, lam, vol = _rest_and_perturbed(rng, t=3, dim=dim)
    t = F.shape[0]

    def energy_flat(F_flat: np.ndarray) -> np.ndarray:
        return np.array(
            [
                float(
                    linear_elasticity_energy_element_F(
                        F_flat.reshape(t, dim, dim), mu, lam
                    ).sum()
                )
            ]
        )

    g_fd = gradient_cfd(energy_flat, F.flatten(), FD_STEP).reshape(t, dim, dim)
    g = linear_elasticity_gradient_element_F(F, mu, lam)

    assert np.allclose(g, g_fd, atol=GRAD_TOL)


@pytest.mark.parametrize("dim", [2, 3])
def test_linear_elasticity_hessian_matches_fd(dim: int) -> None:
    rng = np.random.default_rng(2)
    _, F, mu, lam, vol = _rest_and_perturbed(rng, t=2, dim=dim)
    t = F.shape[0]

    def grad_flat(F_flat: np.ndarray) -> np.ndarray:
        return linear_elasticity_gradient_element_F(
            F_flat.reshape(t, dim, dim), mu, lam
        ).flatten()

    H_fd = gradient_cfd(grad_flat, F.flatten(), FD_STEP)
    H_blocks = linear_elasticity_hessian_element_F(F, mu, lam)
    H = sps.block_diag(H_blocks).toarray()

    assert np.allclose(H, H_fd, atol=HESS_TOL)


if __name__ == "__main__":   
    test_linear_elasticity_energy_increases_with_deformation(2)
    test_linear_elasticity_gradient_matches_fd(2)
    test_linear_elasticity_hessian_matches_fd(2)