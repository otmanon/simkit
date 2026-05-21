"""Tests for ``simkit.energies.emu``.

The EMU energy is ``0.5 * sum_i vol_i * a_i * (d_i^T F_i^T F_i d_i)``: a
quadratic-in-``F`` penalty on the directional stretch ``F d`` along a
per-element direction ``d``. It does not have a single "rest" deformation
gradient (any ``F`` with ``Fd = 0`` minimizes it), so for the
"deformation" check we compare ``F = I`` against a stretched ``F`` along
``d`` and verify the energy grows.

The element-tier functions take per-element ``F`` and material params
(``d``, ``a``) only; quadrature weighting (``vol``) is applied at the global
tier, so it does not appear here.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

from simkit.energies.emu import (
    emu_energy_element_F,
    emu_gradient_element_F,
    emu_hessian_element_F,
)
from simkit.gradient_cfd import gradient_cfd


FD_STEP = 1e-6
GRAD_TOL = 1e-5
HESS_TOL = 1e-4


def _setup(rng: np.random.Generator, t: int, dim: int):
    d = rng.standard_normal((t, dim))
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    a = rng.uniform(0.5, 2.0, size=(t, 1))
    return d, a


@pytest.mark.parametrize("dim", [2, 3])
def test_emu_energy_increases_with_deformation(dim: int) -> None:
    rng = np.random.default_rng(0)
    t = 4
    d, a = _setup(rng, t, dim)

    F_rest = np.tile(np.eye(dim)[None, :, :], (t, 1, 1))
    F_def = F_rest + 0.2 * d[:, :, None] * d[:, None, :]

    e_rest = float(emu_energy_element_F(F_rest, d, a).sum())
    e_def = float(emu_energy_element_F(F_def, d, a).sum())

    assert e_def > e_rest


@pytest.mark.parametrize("dim", [2, 3])
def test_emu_gradient_matches_fd(dim: int) -> None:
    rng = np.random.default_rng(1)
    t = 3
    d, a = _setup(rng, t, dim)
    F = np.tile(np.eye(dim)[None, :, :], (t, 1, 1)) + 0.05 * rng.standard_normal(
        (t, dim, dim)
    )

    def energy_flat(F_flat: np.ndarray) -> np.ndarray:
        return np.array(
            [float(emu_energy_element_F(F_flat.reshape(t, dim, dim), d, a).sum())]
        )

    g_fd = gradient_cfd(energy_flat, F.flatten(), FD_STEP).reshape(t, dim, dim)
    g = emu_gradient_element_F(F, d, a)

    assert np.allclose(g, g_fd, atol=GRAD_TOL)


@pytest.mark.parametrize("dim", [2, 3])
def test_emu_hessian_matches_fd(dim: int) -> None:
    rng = np.random.default_rng(2)
    t = 2
    d, a = _setup(rng, t, dim)
    F = np.tile(np.eye(dim)[None, :, :], (t, 1, 1)) + 0.05 * rng.standard_normal(
        (t, dim, dim)
    )

    def grad_flat(F_flat: np.ndarray) -> np.ndarray:
        return emu_gradient_element_F(F_flat.reshape(t, dim, dim), d, a).flatten()

    H_fd = gradient_cfd(grad_flat, F.flatten(), FD_STEP)
    H_blocks = emu_hessian_element_F(F, d, a)
    H = sps.block_diag(H_blocks).toarray()

    assert np.allclose(H, H_fd, atol=HESS_TOL)


if __name__ == "__main__":
    test_emu_energy_increases_with_deformation(3)
    test_emu_gradient_matches_fd(3)
    test_emu_hessian_matches_fd(3)