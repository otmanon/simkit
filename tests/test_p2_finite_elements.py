"""Tests for the P2 (quadratic element) modules.

Covers ``p2_shape_functions``, ``deformation_gradient_p2`` and
``p2_gravity_force``, which had no by-name test files. The properties checked
are the ones that pin the element down: partition of unity, nodal
interpolation, exactness of the deformation gradient on affine maps, and
consistency of the gravity load with the P2 mass matrix.
"""

from __future__ import annotations

import numpy as np
import pytest

from simkit import (
    deformation_gradient_p2,
    gauss_legendre_quadrature,
    linear_to_quadratic_elements,
    p2_gravity_force,
    p2_massmatrix,
)
from simkit.p2_shape_functions import p2_num_nodes, p2_shape_functions


TRI = 3
TET = 4


@pytest.fixture
def tri_mesh():
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    T = np.array([[0, 1, 2], [1, 3, 2]])
    return V, T


@pytest.fixture
def tet_mesh():
    V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    T = np.array([[0, 1, 2, 3]])
    return V, T


def _random_barycentric(s, rng):
    L = rng.random(s)
    return L / L.sum()


# --------------------------------------------------------------------------- #
# p2_shape_functions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("s,n_nodes", [(TRI, 6), (TET, 10)])
def test_node_count(s, n_nodes):
    assert p2_num_nodes(s) == n_nodes


@pytest.mark.parametrize("s", [TRI, TET])
def test_shape_functions_form_a_partition_of_unity(s):
    rng = np.random.default_rng(0)
    for _ in range(10):
        N, _ = p2_shape_functions(_random_barycentric(s, rng), s)
        assert np.isclose(N.sum(), 1.0)


@pytest.mark.parametrize("s", [TRI, TET])
def test_shape_functions_are_nodal(s):
    """``N_i`` is 1 at node ``i`` and 0 at every other node."""
    n_nodes = p2_num_nodes(s)
    # Corner nodes sit at the barycentric basis vectors.
    for i in range(s):
        L = np.zeros(s)
        L[i] = 1.0
        N, _ = p2_shape_functions(L, s)
        expected = np.zeros(n_nodes)
        expected[i] = 1.0
        assert np.allclose(N, expected, atol=1e-12)


@pytest.mark.parametrize("s", [TRI, TET])
def test_shape_function_gradients_are_consistent_with_partition_of_unity(s):
    """``sum_i dN_i/dL_j`` must not depend on ``j``.

    The barycentric coordinates are constrained to sum to 1, so only
    differences between columns are meaningful. Equal column sums are exactly
    the statement that ``sum_i N_i`` is constant along the simplex. (The shared
    value is 3 for both the triangle and the tetrahedron.)
    """
    rng = np.random.default_rng(1)
    for _ in range(10):
        _, dNdL = p2_shape_functions(_random_barycentric(s, rng), s)
        col_sums = dNdL.sum(axis=0)
        assert np.allclose(col_sums, col_sums[0], atol=1e-12)
        assert np.isclose(col_sums[0], 3.0)


@pytest.mark.parametrize("s", [TRI, TET])
def test_shape_function_gradients_match_finite_differences(s):
    """Differentiate along the barycentric simplex (perturbations sum to 0)."""
    rng = np.random.default_rng(2)
    L = _random_barycentric(s, rng)
    _, dNdL = p2_shape_functions(L, s)

    h = 1e-6
    for a in range(s):
        for b in range(s):
            if a == b:
                continue
            d = np.zeros(s)
            d[a], d[b] = h, -h
            Np, _ = p2_shape_functions(L + d, s)
            Nm, _ = p2_shape_functions(L - d, s)
            fd = (Np - Nm) / (2 * h)
            analytic = dNdL[:, a] - dNdL[:, b]
            assert np.allclose(fd, analytic, atol=1e-6)


# --------------------------------------------------------------------------- #
# deformation_gradient_p2
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mesh_name", ["tri_mesh", "tet_mesh"])
def test_rest_configuration_gives_identity(mesh_name, request):
    V, T = request.getfixturevalue(mesh_name)
    V2, T2 = linear_to_quadratic_elements(V, T)
    bary, _ = gauss_legendre_quadrature(V, T, 2)

    F = deformation_gradient_p2(V2, T2, bary, V2)
    dim = V.shape[1]
    assert F.shape[-2:] == (dim, dim)
    assert np.allclose(F, np.eye(dim), atol=1e-10)


@pytest.mark.parametrize("mesh_name", ["tri_mesh", "tet_mesh"])
def test_affine_map_is_reproduced_exactly(mesh_name, request):
    """A P2 element must represent any affine deformation exactly.

    Under ``x = A X + b`` the deformation gradient is ``A`` at every
    cubature point.
    """
    V, T = request.getfixturevalue(mesh_name)
    dim = V.shape[1]
    rng = np.random.default_rng(3)
    A = np.eye(dim) + 0.15 * rng.standard_normal((dim, dim))
    b = rng.standard_normal(dim)

    V2, T2 = linear_to_quadratic_elements(V, T)
    bary, _ = gauss_legendre_quadrature(V, T, 2)
    U2 = V2 @ A.T + b

    F = deformation_gradient_p2(V2, T2, bary, U2)
    assert np.allclose(F, A, atol=1e-10)


def test_rigid_translation_leaves_F_at_identity(tri_mesh):
    V, T = tri_mesh
    V2, T2 = linear_to_quadratic_elements(V, T)
    bary, _ = gauss_legendre_quadrature(V, T, 2)

    F = deformation_gradient_p2(V2, T2, bary, V2 + np.array([2.0, -3.0]))
    assert np.allclose(F, np.eye(2), atol=1e-10)


def test_uniform_scaling_scales_F(tri_mesh):
    V, T = tri_mesh
    V2, T2 = linear_to_quadratic_elements(V, T)
    bary, _ = gauss_legendre_quadrature(V, T, 2)

    F = deformation_gradient_p2(V2, T2, bary, 2.5 * V2)
    assert np.allclose(F, 2.5 * np.eye(2), atol=1e-10)


# --------------------------------------------------------------------------- #
# p2_gravity_force
# --------------------------------------------------------------------------- #
def test_gravity_acts_only_along_the_second_axis(tri_mesh):
    V, T = tri_mesh
    V2, T2 = linear_to_quadratic_elements(V, T)
    bary, weights = gauss_legendre_quadrature(V, T, 4)

    g = p2_gravity_force(V2, T2, bary, weights, a=-9.8)
    assert g.shape == V2.shape
    assert np.allclose(g[:, 0], 0.0, atol=1e-12)
    assert not np.allclose(g[:, 1], 0.0)


def test_total_gravity_equals_total_mass_times_acceleration(tri_mesh):
    """Consistency with the P2 mass matrix: sum(g_y) == a * total mass."""
    V, T = tri_mesh
    V2, T2 = linear_to_quadratic_elements(V, T)
    bary, weights = gauss_legendre_quadrature(V, T, 4)

    a, rho = -9.8, 2.0
    g = p2_gravity_force(V2, T2, bary, weights, a=a, rho=rho)
    M = p2_massmatrix(V2, T2, bary, weights, rho=rho)

    total_mass = M.sum()
    assert np.isclose(g[:, 1].sum(), a * total_mass, rtol=1e-9)


def test_gravity_scales_linearly_with_density_and_acceleration(tri_mesh):
    V, T = tri_mesh
    V2, T2 = linear_to_quadratic_elements(V, T)
    bary, weights = gauss_legendre_quadrature(V, T, 4)

    base = p2_gravity_force(V2, T2, bary, weights, a=-1.0, rho=1.0)
    scaled = p2_gravity_force(V2, T2, bary, weights, a=-2.0, rho=3.0)
    assert np.allclose(scaled, 6.0 * base)
