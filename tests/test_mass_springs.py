"""Tests for ``simkit.energies.mass_springs``.

The mass-springs energy is expressed in terms of per-edge displacements
``d`` (edge vectors of length ``dim``). At rest, ``||d|| = l0`` and the
energy is zero; stretching or compressing the edge increases it
quadratically about the rest length.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

from simkit.energies.mass_springs import (
    mass_springs_energy_d,
    mass_springs_gradient_dd,
    mass_springs_hessian_dd,
)
from simkit.gradient_cfd import gradient_cfd


FD_STEP = 1e-6
GRAD_TOL = 1e-5
HESS_TOL = 1e-4


def _rest_and_perturbed(rng: np.random.Generator, num_edges: int, dim: int):
    d_rest = rng.standard_normal((num_edges, dim))
    d_rest = d_rest / np.linalg.norm(d_rest, axis=1, keepdims=True)
    l0 = rng.uniform(0.5, 1.5, size=(num_edges, 1))
    d_rest = d_rest * l0
    d_def = d_rest + 0.05 * rng.standard_normal((num_edges, dim))
    ym = rng.uniform(0.5, 2.0, size=(num_edges, 1))
    vol = rng.uniform(0.5, 1.5, size=(num_edges, 1))
    return d_rest, d_def, ym, vol, l0


@pytest.mark.parametrize("dim", [2, 3])
def test_mass_springs_energy_increases_with_deformation(dim: int) -> None:
    rng = np.random.default_rng(0)
    d_rest, d_def, ym, vol, l0 = _rest_and_perturbed(rng, num_edges=5, dim=dim)

    e_rest = float(mass_springs_energy_d(d_rest, ym, vol, l0).item())
    e_def = float(mass_springs_energy_d(d_def, ym, vol, l0).item())

    assert e_rest == pytest.approx(0.0, abs=1e-10)
    assert e_def > e_rest


@pytest.mark.parametrize("dim", [2, 3])
def test_mass_springs_gradient_matches_fd(dim: int) -> None:
    rng = np.random.default_rng(1)
    _, d, ym, vol, l0 = _rest_and_perturbed(rng, num_edges=4, dim=dim)
    num_edges = d.shape[0]

    def energy_flat(d_flat: np.ndarray) -> np.ndarray:
        return np.array(
            [
                float(
                    mass_springs_energy_d(
                        d_flat.reshape(num_edges, dim), ym, vol, l0
                    ).item()
                )
            ]
        )

    g_fd = gradient_cfd(energy_flat, d.flatten(), FD_STEP).reshape(num_edges, dim)
    g = mass_springs_gradient_dd(d, ym, vol, l0)

    assert np.allclose(g, g_fd, atol=GRAD_TOL)


@pytest.mark.parametrize("dim", [2, 3])
def test_mass_springs_hessian_matches_fd(dim: int) -> None:
    rng = np.random.default_rng(2)
    _, d, ym, vol, l0 = _rest_and_perturbed(rng, num_edges=3, dim=dim)
    num_edges = d.shape[0]

    def grad_flat(d_flat: np.ndarray) -> np.ndarray:
        return mass_springs_gradient_dd(
            d_flat.reshape(num_edges, dim), ym, vol, l0
        ).flatten()

    H_fd = gradient_cfd(grad_flat, d.flatten(), FD_STEP)
    H_blocks = mass_springs_hessian_dd(d, ym, vol, l0)
    H = sps.block_diag(H_blocks).toarray()

    assert np.allclose(H, H_fd, atol=HESS_TOL)
