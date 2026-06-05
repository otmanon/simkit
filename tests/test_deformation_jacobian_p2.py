"""Tests for ``simkit.deformation_jacobian_p2``."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

from simkit.deformation_gradient_p2 import deformation_gradient_p2
from simkit.deformation_jacobian import deformation_jacobian
from simkit.deformation_jacobian_p2 import deformation_jacobian_p2
from simkit.gauss_legendre_quadrature import gauss_legendre_quadrature
from simkit.linear_to_quadratic_elements import linear_to_quadratic_elements
from simkit.volume import volume
from simkit.energies.arap import arap_energy_element_F


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
        T = np.array([[0, 1, 2, 3], [0, 2, 1, 4]])
    else:
        raise ValueError(dim)
    return X, T


def _rest_points(V2, T2, bary):
    """Rest-space positions of every cubature point, shape (t*n_quad, dim)."""
    t, n_quad, s = bary.shape
    pts = np.zeros((t * n_quad, V2.shape[1]))
    for e in range(t):
        corners = T2[e, :s]
        for q in range(n_quad):
            pts[e * n_quad + q] = bary[e, q] @ V2[corners]
    return pts


@pytest.mark.parametrize("dim", [2, 3])
def test_shape_and_type(dim: int) -> None:
    X, T = _two_element_mesh(dim)
    V2, T2 = linear_to_quadratic_elements(X, T)
    bary, w = gauss_legendre_quadrature(X, T, 2)
    J = deformation_jacobian_p2(V2, T2, bary, w)

    t, n_quad = bary.shape[0], bary.shape[1]
    assert isinstance(J, sps.csc_matrix)
    assert J.shape == (t * n_quad * dim * dim, V2.shape[0] * dim)


@pytest.mark.parametrize("dim", [2, 3])
def test_affine_patch_test(dim: int) -> None:
    # An affine deformation U = A X + b must yield F == A at EVERY cubature
    # point (P2 reproduces linear fields exactly).
    rng = np.random.default_rng(0)
    X, T = _two_element_mesh(dim)
    V2, T2 = linear_to_quadratic_elements(X, T)
    bary, w = gauss_legendre_quadrature(X, T, 2)
    J = deformation_jacobian_p2(V2, T2, bary, w)

    A = np.eye(dim) + 0.2 * rng.standard_normal((dim, dim))
    b = rng.standard_normal(dim)
    U2 = V2 @ A.T + b

    F = (J @ U2.reshape(-1, 1)).reshape(-1, dim, dim)
    assert np.allclose(F, A[None], atol=1e-12)


@pytest.mark.parametrize("dim", [2, 3])
def test_quadratic_field_gradient_matches_analytic(dim: int) -> None:
    # For a quadratic deformation phi_k(X) = b_k . X + X^T C_k X, P2 is exact, so
    # F at each cubature point must equal the analytic gradient there:
    #   grad phi_k (X) = b_k + (C_k + C_k^T) X.
    rng = np.random.default_rng(3)
    X, T = _two_element_mesh(dim)
    V2, T2 = linear_to_quadratic_elements(X, T)
    bary, w = gauss_legendre_quadrature(X, T, 2)
    J = deformation_jacobian_p2(V2, T2, bary, w)

    B = rng.standard_normal((dim, dim))        # linear coeffs, rows = component
    C = rng.standard_normal((dim, dim, dim))   # quadratic coeffs per component

    def phi(P):
        out = P @ B.T
        for k in range(dim):
            out[:, k] += np.einsum("ni,ij,nj->n", P, C[k], P)
        return out

    U2 = phi(V2)
    F = (J @ U2.reshape(-1, 1)).reshape(-1, dim, dim)

    pts = _rest_points(V2, T2, bary)
    F_exact = np.zeros_like(F)
    for p in range(pts.shape[0]):
        for k in range(dim):
            F_exact[p, k, :] = B[k] + (C[k] + C[k].T) @ pts[p]
    assert np.allclose(F, F_exact, atol=1e-10)

    # And F genuinely varies across cubature points (it is not constant per
    # element the way P1 would be).
    F_per_elem = F.reshape(T.shape[0], bary.shape[1], dim, dim)
    assert np.abs(F_per_elem - F_per_elem[:, :1]).max() > 1e-6


@pytest.mark.parametrize("dim", [2, 3])
def test_operator_is_state_independent(dim: int) -> None:
    # The operator is built from rest geometry + fixed quadrature points only;
    # rebuilding it never depends on any deformed configuration.
    X, T = _two_element_mesh(dim)
    V2, T2 = linear_to_quadratic_elements(X, T)
    bary, w = gauss_legendre_quadrature(X, T, 2)
    J1 = deformation_jacobian_p2(V2, T2, bary, w)
    J2 = deformation_jacobian_p2(V2, T2, bary, w)
    assert (J1 - J2).nnz == 0
    # weights do not affect the operator either.
    J3 = deformation_jacobian_p2(V2, T2, bary, 0.0 * w)
    assert np.allclose((J1 - J3).toarray(), 0.0)


@pytest.mark.parametrize("dim", [2, 3])
def test_dense_gradient_matches_sparse_operator(dim: int) -> None:
    # deformation_gradient_p2 (dense, direct) must agree with the sparse
    # operator J applied to the flattened deformed positions.
    rng = np.random.default_rng(5)
    X, T = _two_element_mesh(dim)
    V2, T2 = linear_to_quadratic_elements(X, T)
    bary, w = gauss_legendre_quadrature(X, T, 2)

    U2 = V2 + 0.1 * rng.standard_normal(V2.shape)

    F_dense = deformation_gradient_p2(V2, T2, bary, U2)
    assert F_dense.shape == (T.shape[0], bary.shape[1], dim, dim)

    J = deformation_jacobian_p2(V2, T2, bary, w)
    F_sparse = (J @ U2.reshape(-1, 1)).reshape(-1, dim, dim)

    assert np.allclose(F_dense.reshape(-1, dim, dim), F_sparse, atol=1e-12)


@pytest.mark.parametrize("dim", [2, 3])
def test_p1_p2_energy_agree_for_affine(dim: int) -> None:
    # Under an affine deformation F == A everywhere, so the P2 multi-point
    # quadrature Σ_q w_q ψ(A) collapses to vol·ψ(A) -- the P1 answer.
    rng = np.random.default_rng(7)
    X, T = _two_element_mesh(dim)
    V2, T2 = linear_to_quadratic_elements(X, T)
    bary, w = gauss_legendre_quadrature(X, T, 2)

    mu_val = 1.7
    A = np.eye(dim) + 0.15 * rng.standard_normal((dim, dim))
    b = rng.standard_normal(dim)

    # P1 energy on the linear mesh.
    J1 = deformation_jacobian(X, T)
    vol1 = volume(X, T)
    U1 = X @ A.T + b
    F1 = (J1 @ U1.reshape(-1, 1)).reshape(-1, dim, dim)
    E1 = float((vol1 * arap_energy_element_F(F1, np.full((F1.shape[0], 1), mu_val))).sum())

    # P2 energy on the quadratic mesh.
    J2 = deformation_jacobian_p2(V2, T2, bary, w)
    U2 = V2 @ A.T + b
    F2 = (J2 @ U2.reshape(-1, 1)).reshape(-1, dim, dim)
    wcol = w.reshape(-1, 1)
    E2 = float((wcol * arap_energy_element_F(F2, np.full((F2.shape[0], 1), mu_val))).sum())

    assert E2 == pytest.approx(E1, rel=1e-10, abs=1e-12)
