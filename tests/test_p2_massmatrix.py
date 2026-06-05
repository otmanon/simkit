"""Tests for ``simkit.p2_massmatrix`` and ``simkit.p2_gravity_force``."""

from __future__ import annotations

import numpy as np
import pytest

from simkit.gauss_legendre_quadrature import gauss_legendre_quadrature
from simkit.linear_to_quadratic_elements import linear_to_quadratic_elements
from simkit.p2_gravity_force import p2_gravity_force
from simkit.p2_massmatrix import p2_massmatrix
from simkit.volume import volume


def _two_element_mesh(dim: int):
    if dim == 2:
        X = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        T = np.array([[0, 1, 2], [0, 2, 3]])
    elif dim == 3:
        X = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )
        # Both tets positively oriented (second apex below the shared face).
        T = np.array([[0, 1, 2, 3], [0, 2, 1, 4]])
    else:
        raise ValueError(dim)
    return X, T


@pytest.mark.parametrize("dim", [2, 3])
def test_total_mass_matches_volume(dim: int) -> None:
    X, T = _two_element_mesh(dim)
    V2, T2 = linear_to_quadratic_elements(X, T)
    # N_i N_j is quartic (product of two quadratic shape funcs) -> order 4.
    bary, w = gauss_legendre_quadrature(X, T, 4)

    rho = 2.5
    M = p2_massmatrix(V2, T2, bary, w, rho=rho)
    assert M.shape == (V2.shape[0], V2.shape[0])
    # Partition of unity => total mass equals rho * total volume.
    assert M.sum() == pytest.approx(rho * float(volume(X, T).sum()), rel=1e-10)


@pytest.mark.parametrize("dim", [2, 3])
def test_mass_matrix_is_symmetric_psd(dim: int) -> None:
    X, T = _two_element_mesh(dim)
    V2, T2 = linear_to_quadratic_elements(X, T)
    bary, w = gauss_legendre_quadrature(X, T, 4)
    M = p2_massmatrix(V2, T2, bary, w).toarray()

    assert np.allclose(M, M.T, atol=1e-12)
    eigs = np.linalg.eigvalsh(M)
    assert eigs.min() > -1e-12  # consistent mass matrix is SPD


@pytest.mark.parametrize("dim", [2, 3])
def test_rigid_translation_kinetic_energy(dim: int) -> None:
    # For a uniform velocity v, the kinetic energy 0.5 vᵀ M v summed over
    # components equals 0.5 * total_mass * |v|^2.
    X, T = _two_element_mesh(dim)
    V2, T2 = linear_to_quadratic_elements(X, T)
    bary, w = gauss_legendre_quadrature(X, T, 4)
    M = p2_massmatrix(V2, T2, bary, w)

    total_mass = float(volume(X, T).sum())
    v = np.ones(V2.shape[0])
    ke = 0.5 * v @ (M @ v)
    assert ke == pytest.approx(0.5 * total_mass, rel=1e-10)


@pytest.mark.parametrize("dim", [2, 3])
def test_gravity_total_force(dim: int) -> None:
    X, T = _two_element_mesh(dim)
    V2, T2 = linear_to_quadratic_elements(X, T)
    bary, w = gauss_legendre_quadrature(X, T, 4)

    a = -9.8
    rho = 1.3
    g = p2_gravity_force(V2, T2, bary, w, a=a, rho=rho)
    assert g.shape == (V2.shape[0], dim)
    # Net vertical force equals (total mass) * a; other components vanish.
    total_mass = rho * float(volume(X, T).sum())
    assert g[:, 1].sum() == pytest.approx(total_mass * a, rel=1e-10)
    for c in range(dim):
        if c != 1:
            assert np.allclose(g[:, c], 0.0, atol=1e-12)
