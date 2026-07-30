"""Tests for ``simkit.energies.mass_springs``.

The mass-springs energy is expressed in terms of per-edge displacements
``d`` (edge vectors of length ``dim``). At rest, ``||d|| = l0`` and the
energy is zero; stretching or compressing the edge increases it
quadratically about the rest length.

The element-tier functions take per-edge ``d`` and material params
(``ym``, ``l0``) only; quadrature weighting (``vol``) is applied at the
global tier, so it does not appear here.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

from simkit.edge_displacement_jacobian import edge_displacement_jacobian
from simkit.energies.mass_springs import (
    mass_springs_energy_element_d,
    mass_springs_energy_l,
    mass_springs_energy_x,
    mass_springs_energy_z,
    mass_springs_gradient_element_d,
    mass_springs_gradient_l,
    mass_springs_gradient_x,
    mass_springs_gradient_z,
    mass_springs_hessian_d_l0,
    mass_springs_hessian_element_d,
    mass_springs_hessian_l,
    mass_springs_hessian_x,
    mass_springs_hessian_z,
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
    return d_rest, d_def, ym, l0


@pytest.mark.parametrize("dim", [2, 3])
def test_mass_springs_energy_increases_with_deformation(dim: int) -> None:
    rng = np.random.default_rng(0)
    d_rest, d_def, ym, l0 = _rest_and_perturbed(rng, num_edges=5, dim=dim)

    e_rest = float(mass_springs_energy_element_d(d_rest, ym, l0).sum())
    e_def = float(mass_springs_energy_element_d(d_def, ym, l0).sum())

    assert e_rest == pytest.approx(0.0, abs=1e-10)
    assert e_def > e_rest


@pytest.mark.parametrize("dim", [2, 3])
def test_mass_springs_gradient_matches_fd(dim: int) -> None:
    rng = np.random.default_rng(1)
    _, d, ym, l0 = _rest_and_perturbed(rng, num_edges=4, dim=dim)
    num_edges = d.shape[0]

    def energy_flat(d_flat: np.ndarray) -> np.ndarray:
        return np.array(
            [
                float(
                    mass_springs_energy_element_d(
                        d_flat.reshape(num_edges, dim), ym, l0
                    ).sum()
                )
            ]
        )

    g_fd = gradient_cfd(energy_flat, d.flatten(), FD_STEP).reshape(num_edges, dim)
    g = mass_springs_gradient_element_d(d, ym, l0)

    assert np.allclose(g, g_fd, atol=GRAD_TOL)


@pytest.mark.parametrize("dim", [2, 3])
def test_mass_springs_hessian_matches_fd(dim: int) -> None:
    rng = np.random.default_rng(2)
    _, d, ym, l0 = _rest_and_perturbed(rng, num_edges=3, dim=dim)
    num_edges = d.shape[0]

    def grad_flat(d_flat: np.ndarray) -> np.ndarray:
        return mass_springs_gradient_element_d(
            d_flat.reshape(num_edges, dim), ym, l0
        ).flatten()

    H_fd = gradient_cfd(grad_flat, d.flatten(), FD_STEP)
    H_blocks = mass_springs_hessian_element_d(d, ym, l0)
    H = sps.block_diag(H_blocks).toarray()

    assert np.allclose(H, H_fd, atol=HESS_TOL)


# --------------------------------------------------------------------------- #
# Material model: 0.5 * ym * (||d|| - l0)^2, no rest-length normalization      #
# --------------------------------------------------------------------------- #
def test_material_model_is_not_rest_length_normalized() -> None:
    """Density must be ``0.5*ym*(l-l0)^2`` exactly -- no ``1/l0^2`` factor."""
    d = np.array([[3.0, 0.0], [5.0, 0.0]])
    ym = np.array([[2.0], [2.0]])
    l0 = np.array([[1.0], [3.0]])

    psi = mass_springs_energy_element_d(d, ym, l0)

    # Both edges are stretched by the same absolute amount (2.0) with the same
    # stiffness, so under the unnormalized model they store the same energy.
    assert psi[0, 0] == pytest.approx(0.5 * 2.0 * 2.0 ** 2)
    assert psi[1, 0] == pytest.approx(0.5 * 2.0 * 2.0 ** 2)


def test_stiffness_scales_energy_linearly() -> None:
    """``ym`` is the stiffness outright, so energy is linear in it."""
    rng = np.random.default_rng(7)
    _, d, ym, l0 = _rest_and_perturbed(rng, num_edges=4, dim=3)

    e1 = float(mass_springs_energy_element_d(d, ym, l0).sum())
    e2 = float(mass_springs_energy_element_d(d, 3.0 * ym, l0).sum())

    assert e2 == pytest.approx(3.0 * e1)


# --------------------------------------------------------------------------- #
# Global explicit tier: positions (x)                                         #
# --------------------------------------------------------------------------- #
def _mesh(rng: np.random.Generator, num_verts: int, dim: int):
    """A perturbed open chain plus a couple of extra chords."""
    x = np.linspace(0.0, 1.0, num_verts)[:, None] * np.ones((1, dim))
    x = x + 0.1 * rng.standard_normal((num_verts, dim))
    chain = np.stack([np.arange(num_verts - 1), np.arange(1, num_verts)], axis=1)
    chords = np.array([[0, num_verts - 1], [0, num_verts - 2]])
    E = np.concatenate([chain, chords], axis=0)

    num_edges = E.shape[0]
    ym = rng.uniform(0.5, 2.0, size=(num_edges, 1))
    vol = rng.uniform(0.5, 1.5, size=(num_edges, 1))
    # Rest lengths straddle the current ones, so some edges are compressed.
    l0 = np.linalg.norm(x[E[:, 0]] - x[E[:, 1]], axis=1)[:, None]
    l0 = l0 * rng.uniform(0.7, 1.3, size=(num_edges, 1))
    return x, E, ym, vol, l0


@pytest.mark.parametrize("dim", [2, 3])
def test_energy_x_matches_element_tier(dim: int) -> None:
    rng = np.random.default_rng(10)
    x, E, ym, vol, l0 = _mesh(rng, num_verts=6, dim=dim)

    d = x[E[:, 0]] - x[E[:, 1]]
    expected = float((vol * mass_springs_energy_element_d(d, ym, l0)).sum())

    assert mass_springs_energy_x(x, E, ym, vol, l0) == pytest.approx(expected)


@pytest.mark.parametrize("dim", [2, 3])
def test_gradient_x_matches_fd(dim: int) -> None:
    rng = np.random.default_rng(11)
    x, E, ym, vol, l0 = _mesh(rng, num_verts=6, dim=dim)

    def energy_flat(x_flat: np.ndarray) -> np.ndarray:
        return np.array([mass_springs_energy_x(x_flat.reshape(-1, dim), E, ym, vol, l0)])

    g_fd = gradient_cfd(energy_flat, x.flatten(), FD_STEP).reshape(-1, 1)
    g = mass_springs_gradient_x(x, E, ym, vol, l0)

    assert g.shape == (x.size, 1)
    assert np.allclose(g, g_fd, atol=GRAD_TOL)


@pytest.mark.parametrize("dim", [2, 3])
def test_hessian_x_matches_fd(dim: int) -> None:
    rng = np.random.default_rng(12)
    x, E, ym, vol, l0 = _mesh(rng, num_verts=6, dim=dim)

    def grad_flat(x_flat: np.ndarray) -> np.ndarray:
        return mass_springs_gradient_x(x_flat.reshape(-1, dim), E, ym, vol, l0).flatten()

    H_fd = gradient_cfd(grad_flat, x.flatten(), FD_STEP)
    H = mass_springs_hessian_x(x, E, ym, vol, l0, psd=False).toarray()

    assert H.shape == (x.size, x.size)
    assert np.allclose(H, H_fd, atol=HESS_TOL)


@pytest.mark.parametrize("dim", [2, 3])
def test_hessian_x_psd_projection(dim: int) -> None:
    """Compressed springs give an indefinite Hessian; ``psd=True`` fixes it."""
    rng = np.random.default_rng(13)
    x, E, ym, vol, l0 = _mesh(rng, num_verts=6, dim=dim)
    l0 = 3.0 * l0  # every edge heavily compressed

    H_raw = mass_springs_hessian_x(x, E, ym, vol, l0, psd=False).toarray()
    H_psd = mass_springs_hessian_x(x, E, ym, vol, l0, psd=True).toarray()

    assert np.linalg.eigvalsh(H_raw).min() < -1e-8
    assert np.linalg.eigvalsh(H_psd).min() > -1e-8


@pytest.mark.parametrize("dim", [2, 3])
def test_rest_state_x_tier_is_a_minimum(dim: int) -> None:
    rng = np.random.default_rng(14)
    x, E, ym, vol, _ = _mesh(rng, num_verts=6, dim=dim)
    l0 = np.linalg.norm(x[E[:, 0]] - x[E[:, 1]], axis=1)[:, None]

    assert mass_springs_energy_x(x, E, ym, vol, l0) == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(mass_springs_gradient_x(x, E, ym, vol, l0), 0.0, atol=1e-12)


# --------------------------------------------------------------------------- #
# x tier vs z tier: same energy through the edge-difference operator          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dim", [2, 3])
def test_x_tier_matches_z_tier(dim: int) -> None:
    rng = np.random.default_rng(15)
    x, E, ym, vol, l0 = _mesh(rng, num_verts=6, dim=dim)

    J = sps.kron(edge_displacement_jacobian(x, E), sps.identity(dim))
    z = x.reshape(-1, 1)

    assert mass_springs_energy_z(z, J, ym, vol, l0) == pytest.approx(
        mass_springs_energy_x(x, E, ym, vol, l0)
    )
    assert np.allclose(
        mass_springs_gradient_z(z, J, ym, vol, l0),
        mass_springs_gradient_x(x, E, ym, vol, l0),
    )
    for psd in (False, True):
        assert np.allclose(
            mass_springs_hessian_z(z, J, ym, vol, l0, psd=psd).toarray(),
            mass_springs_hessian_x(x, E, ym, vol, l0, psd=psd).toarray(),
        )


# --------------------------------------------------------------------------- #
# Length-space helpers (l)                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dim", [2, 3])
def test_energy_l_matches_x_tier(dim: int) -> None:
    rng = np.random.default_rng(16)
    x, E, ym, vol, l0 = _mesh(rng, num_verts=6, dim=dim)
    length = np.linalg.norm(x[E[:, 0]] - x[E[:, 1]], axis=1)[:, None]

    assert mass_springs_energy_l(length, ym, vol, l0) == pytest.approx(
        mass_springs_energy_x(x, E, ym, vol, l0)
    )


def test_gradient_and_hessian_l_match_fd() -> None:
    rng = np.random.default_rng(17)
    num_edges = 5
    l0 = rng.uniform(0.5, 1.5, size=(num_edges, 1))
    length = l0 + 0.1 * rng.standard_normal((num_edges, 1))
    ym = rng.uniform(0.5, 2.0, size=(num_edges, 1))
    vol = rng.uniform(0.5, 1.5, size=(num_edges, 1))

    def energy_flat(l_flat: np.ndarray) -> np.ndarray:
        return np.array([mass_springs_energy_l(l_flat.reshape(-1, 1), ym, vol, l0)])

    g_fd = gradient_cfd(energy_flat, length.flatten(), FD_STEP).reshape(-1, 1)
    assert np.allclose(mass_springs_gradient_l(length, ym, vol, l0), g_fd, atol=GRAD_TOL)

    def grad_flat(l_flat: np.ndarray) -> np.ndarray:
        return mass_springs_gradient_l(l_flat.reshape(-1, 1), ym, vol, l0).flatten()

    H_fd = gradient_cfd(grad_flat, length.flatten(), FD_STEP)
    assert np.allclose(mass_springs_hessian_l(ym, vol, l0).toarray(), H_fd, atol=HESS_TOL)


@pytest.mark.parametrize("dim", [2, 3])
def test_hessian_d_l0_matches_fd(dim: int) -> None:
    """Mixed derivative of the weighted gradient w.r.t. the rest lengths."""
    rng = np.random.default_rng(18)
    _, d, ym, l0 = _rest_and_perturbed(rng, num_edges=4, dim=dim)
    num_edges = d.shape[0]
    vol = rng.uniform(0.5, 1.5, size=(num_edges, 1))

    def grad_flat(l0_flat: np.ndarray) -> np.ndarray:
        g = mass_springs_gradient_element_d(d, ym, l0_flat.reshape(-1, 1)) * vol
        return g.flatten()

    # (num_edges*dim, num_edges); only the per-edge diagonal blocks are nonzero.
    J_fd = gradient_cfd(grad_flat, l0.flatten(), FD_STEP)
    expected = np.stack(
        [J_fd[i * dim : (i + 1) * dim, i] for i in range(num_edges)], axis=0
    )

    assert np.allclose(mass_springs_hessian_d_l0(d, ym, vol, l0), expected, atol=HESS_TOL)


if __name__ == "__main__":
    test_mass_springs_energy_increases_with_deformation(2)
    test_mass_springs_gradient_matches_fd(2)
    test_mass_springs_hessian_matches_fd(2)
    test_material_model_is_not_rest_length_normalized()
    test_stiffness_scales_energy_linearly()
    test_energy_x_matches_element_tier(2)
    test_gradient_x_matches_fd(2)
    test_hessian_x_matches_fd(2)
    test_hessian_x_psd_projection(2)
    test_rest_state_x_tier_is_a_minimum(2)
    test_x_tier_matches_z_tier(2)
    test_energy_l_matches_x_tier(2)
    test_gradient_and_hessian_l_match_fd()
    test_hessian_d_l0_matches_fd(2)