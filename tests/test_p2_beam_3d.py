"""End-to-end 3D test of the quadratic (P2) finite-element pipeline.

The per-component P2 helpers are unit-tested elsewhere (``test_deformation_jacobian_p2``,
``test_p2_massmatrix``, ``test_linear_to_quadratic_elements``,
``test_gauss_legendre_quadrature``). This test wires them together the way a real
simulation does and checks the *physics* on a genuinely 3D problem: a tetrahedral
beam, pinned along one end, sagging under gravity.

Pipeline exercised (all in 3D / tets):
    linear_to_quadratic_elements   -> P2 mesh (10-node tets)
    gauss_legendre_quadrature      -> per-cubature barycentric points + weights
    deformation_jacobian_p2        -> sparse F operator
    deformation_gradient_p2        -> dense F (cross-check)
    p2_massmatrix / p2_gravity_force -> consistent mass + body force
    macklin_mueller_neo_hookean_*  -> elastic energy/grad/hess (unchanged P1 code)
    newton_solver                  -> static gravity equilibrium
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

import simkit
from simkit.deformation_gradient_p2 import deformation_gradient_p2
from simkit.deformation_jacobian_p2 import deformation_jacobian_p2
from simkit.dirichlet_penalty import dirichlet_penalty
from simkit.gauss_legendre_quadrature import gauss_legendre_quadrature
from simkit.linear_to_quadratic_elements import linear_to_quadratic_elements
from simkit.p2_gravity_force import p2_gravity_force
from simkit.p2_massmatrix import p2_massmatrix
from simkit.solvers import newton_solver
from simkit.volume import volume
from simkit.energies.macklin_mueller_neo_hookean import (
    macklin_mueller_neo_hookean_energy_x,
    macklin_mueller_neo_hookean_gradient_x,
    macklin_mueller_neo_hookean_hessian_x,
)
from simkit.energies.arap import arap_energy_element_F


# --------------------------------------------------------------------------- #
# A small tetrahedralized brick (5 tets per hex cell), no examples dependency.
# --------------------------------------------------------------------------- #
def _tet_beam(nx=8, ny=2, nz=2, width=4.0, height=0.4, depth=0.4):
    xs = np.linspace(-width / 2.0, width / 2.0, nx)
    ys = np.linspace(-height / 2.0, height / 2.0, ny)
    zs = np.linspace(-depth / 2.0, depth / 2.0, nz)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
    X = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)

    pattern_even = np.array([
        [0, 1, 3, 4], [1, 2, 3, 6], [1, 4, 5, 6], [3, 4, 6, 7], [1, 3, 4, 6],
    ])
    pattern_odd = np.array([
        [0, 1, 2, 5], [0, 2, 3, 7], [0, 5, 7, 4], [2, 5, 6, 7], [0, 2, 7, 5],
    ])
    ii, jj, kk = np.meshgrid(
        np.arange(nx - 1), np.arange(ny - 1), np.arange(nz - 1), indexing="ij"
    )
    ii, jj, kk = ii.ravel(), jj.ravel(), kk.ravel()

    def vid(i, j, k):
        return (i * ny + j) * nz + k

    corners = np.stack([
        vid(ii, jj, kk), vid(ii + 1, jj, kk), vid(ii + 1, jj + 1, kk),
        vid(ii, jj + 1, kk), vid(ii, jj, kk + 1), vid(ii + 1, jj, kk + 1),
        vid(ii + 1, jj + 1, kk + 1), vid(ii, jj + 1, kk + 1),
    ], axis=1)
    even = ((ii + jj + kk) % 2) == 0
    tets = np.empty((corners.shape[0], 5, 4), dtype=corners.dtype)
    tets[even] = corners[even][:, pattern_even]
    tets[~even] = corners[~even][:, pattern_odd]
    return X, tets.reshape(-1, 4)


def _build_p2_beam(order=2):
    X, T = _tet_beam()
    V2, T2 = linear_to_quadratic_elements(X, T)
    bary, w = gauss_legendre_quadrature(X, T, order=order)
    return X, T, V2, T2, bary, w


def _solve_gravity_sag(V2, T2, bary, w, ym=3e4, pr=0.4, grav=-9.8, gamma=1e8):
    """Static neo-Hookean equilibrium of the P2 beam pinned at x == x_min."""
    n2, dim = V2.shape
    J = deformation_jacobian_p2(V2, T2, bary, w)
    vol = w.reshape(-1, 1)
    nb = vol.shape[0]
    mu0, lam0 = simkit.ympr_to_lame(ym, pr)
    mu = np.full((nb, 1), mu0)
    lam = np.full((nb, 1), lam0)

    fg = p2_gravity_force(V2, T2, bary, w, a=grav, rho=1.0).reshape(-1, 1)

    pin = np.where(V2[:, 0] <= V2[:, 0].min() + 1e-9)[0]
    Q, b = dirichlet_penalty(pin, V2[pin], n2, gamma)

    def E(x):
        xn, xc = x.reshape(-1, dim), x.reshape(-1, 1)
        return (macklin_mueller_neo_hookean_energy_x(xn, J, mu, lam, vol)
                + 0.5 * (xc.T @ (Q @ xc))[0, 0] + (b.T @ xc)[0, 0]
                - (fg.T @ xc)[0, 0])

    def G(x):
        xn, xc = x.reshape(-1, dim), x.reshape(-1, 1)
        return (macklin_mueller_neo_hookean_gradient_x(xn, J, mu, lam, vol)
                + Q @ xc + b - fg)

    def Hf(x):
        return macklin_mueller_neo_hookean_hessian_x(
            x.reshape(-1, dim), J, mu, lam, vol, psd=True) + Q

    x = newton_solver(V2.reshape(-1, 1).copy(), E, G, Hf,
                      max_iter=200, tolerance=1e-9)
    return x.reshape(n2, dim), pin, E


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_pipeline_shapes_3d():
    X, T, V2, T2, bary, w = _build_p2_beam()
    n2 = V2.shape[0]
    # 3D tets -> 10-node P2 elements, 4-component barycentric coords.
    assert T2.shape[1] == 10
    assert bary.shape[1:] == (w.shape[1], 4)

    J = deformation_jacobian_p2(V2, T2, bary, w)
    assert isinstance(J, sps.csc_matrix)
    assert J.shape == (T2.shape[0] * w.shape[1] * 9, n2 * 3)

    M = p2_massmatrix(V2, T2, bary, w, rho=1.0)
    assert M.shape == (n2, n2)


def test_rest_state_is_identity_and_mass_conserved_3d():
    X, T, V2, T2, bary, w = _build_p2_beam()
    J = deformation_jacobian_p2(V2, T2, bary, w)
    # F == I at every cubature point in the undeformed rest state.
    F = (J @ V2.reshape(-1, 1)).reshape(-1, 3, 3)
    assert np.allclose(F, np.eye(3), atol=1e-10)

    # Consistent P2 mass matrix conserves total mass (partition of unity).
    M = p2_massmatrix(V2, T2, bary, w, rho=2.5)
    assert M.sum() == pytest.approx(2.5 * volume(X, T).sum(), rel=1e-12)
    # Total gravitational force == total mass * g, distributed over all nodes.
    fg = p2_gravity_force(V2, T2, bary, w, a=-9.8, rho=2.5)
    assert fg[:, 1].sum() == pytest.approx(-9.8 * 2.5 * volume(X, T).sum(), rel=1e-10)
    assert np.allclose(fg[:, [0, 2]], 0.0)


def test_dense_and_sparse_F_agree_3d():
    X, T, V2, T2, bary, w = _build_p2_beam()
    rng = np.random.default_rng(0)
    U2 = V2 + 0.05 * rng.standard_normal(V2.shape)

    F_dense = deformation_gradient_p2(V2, T2, bary, U2).reshape(-1, 3, 3)
    J = deformation_jacobian_p2(V2, T2, bary, w)
    F_sparse = (J @ U2.reshape(-1, 1)).reshape(-1, 3, 3)
    assert np.allclose(F_dense, F_sparse, atol=1e-12)


def test_beam_sags_under_gravity_3d():
    """The headline physics test: a pinned 3D beam droops under gravity."""
    X, T, V2, T2, bary, w = _build_p2_beam()
    U, pin, E = _solve_gravity_sag(V2, T2, bary, w)

    # 1. Pinned end stays put (penalty is stiff).
    assert np.allclose(U[pin], V2[pin], atol=1e-3)

    # 2. The free end sags: it ends up well below the rest beam.
    assert U[:, 1].min() < V2[:, 1].min() - 0.3

    # 3. Deflection grows monotonically along the beam (root -> free tip).
    x_rest = V2[:, 0]
    free_tip = np.where(x_rest >= x_rest.max() - 1e-9)[0]
    root = np.where(x_rest <= x_rest.min() + 1e-9)[0]
    tip_drop = (V2[free_tip, 1] - U[free_tip, 1]).mean()
    root_drop = (V2[root, 1] - U[root, 1]).mean()
    assert tip_drop > root_drop > -1e-6
    assert tip_drop > 0.2

    # 4. Result is finite and elements stay inverted-free (positive det F).
    assert np.all(np.isfinite(U))
    F = (deformation_jacobian_p2(V2, T2, bary, w) @ U.reshape(-1, 1)).reshape(-1, 3, 3)
    assert np.all(np.linalg.det(F) > 0.0)

    # 5. The solver actually lowered the total potential vs the rest state.
    assert E(U.reshape(-1, 1)) < E(V2.reshape(-1, 1))


def test_stronger_gravity_sags_more_3d():
    """Monotonic response: a heavier pull produces a lower tip."""
    X, T, V2, T2, bary, w = _build_p2_beam()
    U_weak, _, _ = _solve_gravity_sag(V2, T2, bary, w, grav=-2.0)
    U_strong, _, _ = _solve_gravity_sag(V2, T2, bary, w, grav=-12.0)
    assert U_strong[:, 1].min() < U_weak[:, 1].min()


def test_load_acts_primarily_along_gravity_3d():
    """Gravity points along -y, so the response is dominated by vertical sag.

    The 5-tet-per-hex pattern is parity-flipped and therefore *not* exactly
    symmetric about z == 0, so a tiny z drift is an expected meshing artifact --
    but it must be small compared with the vertical motion the load drives."""
    X, T, V2, T2, bary, w = _build_p2_beam()
    U, _, _ = _solve_gravity_sag(V2, T2, bary, w)
    y_motion = np.abs(U[:, 1] - V2[:, 1]).sum()
    z_drift = abs((U[:, 2] - V2[:, 2]).sum())
    assert z_drift < 0.05 * y_motion


def test_p1_p2_energy_agree_for_affine_3d():
    """Under an affine map F == A everywhere, so the P2 multi-point quadrature
    Σ_q w_q ψ(A) must collapse to the single-point P1 energy vol·ψ(A)."""
    rng = np.random.default_rng(11)
    X, T = _tet_beam(nx=3, ny=2, nz=2)
    V2, T2 = linear_to_quadratic_elements(X, T)
    bary, w = gauss_legendre_quadrature(X, T, order=2)

    mu_val = 1.3
    A = np.eye(3) + 0.1 * rng.standard_normal((3, 3))
    b = rng.standard_normal(3)

    J1 = simkit.deformation_jacobian(X, T)
    vol1 = volume(X, T)
    U1 = X @ A.T + b
    F1 = (J1 @ U1.reshape(-1, 1)).reshape(-1, 3, 3)
    E1 = float((vol1 * arap_energy_element_F(
        F1, np.full((F1.shape[0], 1), mu_val))).sum())

    J2 = deformation_jacobian_p2(V2, T2, bary, w)
    U2 = V2 @ A.T + b
    F2 = (J2 @ U2.reshape(-1, 1)).reshape(-1, 3, 3)
    E2 = float((w.reshape(-1, 1) * arap_energy_element_F(
        F2, np.full((F2.shape[0], 1), mu_val))).sum())

    assert E2 == pytest.approx(E1, rel=1e-9, abs=1e-12)
