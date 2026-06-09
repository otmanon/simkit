"""Finite-difference verification of the subspace Mixed-FEM (MFEM) blocks.

The augmented-Lagrangian MFEM solve (see
``examples/interactive_demos/012_interactive_mixed_fem.py`` and
``examples/tutorials/24_subspace_mixed_fem.ipynb``) stacks the state
``p = [u, a, ll]`` (subspace positions, per-cubature symmetric stretch, constraint
multipliers) and minimises the merit

    E = elastic_S(a) + kinetic_be(u) + quad(u)
        + ll^T (W c) + 0.5 * rho_aug * c^T W c,   c = Ci stretch(F(u)) - a

via ``simkit.solvers.sqp_mfem``, which condenses the stretch update and adds the
multiplier coupling (``+ G_u ll`` to ``f_u``, ``+ G_z ll`` to ``f_z``) internally.

These tests rebuild the blocks self-contained on a tiny beam and check, by central
finite differences:

1. ``G_u`` is the (weighted) constraint Jacobian:  ``G_u^T == d(W c)/du``.
2. ``grad_blocks`` are the merit gradient:  ``[f_u + G_u ll, f_z + G_z ll, f_ll]``
   equals ``[dE/du, dE/da, dE/dll]`` -- for ``rho_aug = 0`` and ``rho_aug > 0``.
3. ``H_z`` is the exact stretch-block Hessian:  ``H_z == d^2E/da^2`` (any rho_aug).
4. ``H_u`` is the exact position-block Hessian in the regime where its
   Gauss-Newton form is exact (``ll = 0, rho_aug = 0``), and is SPD for rho_aug >= 0.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy as sp

import simkit as sk
from simkit.energies.elastic import (
    elastic_energy_S, elastic_gradient_S, elastic_hessian_S,
)
from simkit.energies.kinetic import (
    kinetic_energy_be, kinetic_gradient_be, kinetic_hessian_be,
)
from simkit.energies.quadratic import (
    quadratic_energy, quadratic_gradient, quadratic_hessian,
)
from simkit.gradient_cfd import gradient_cfd


MATERIAL = "macklin-mueller-neo-hookean"
FD = 1e-6
TOL = 1e-5


def _grid(nx=5, ny=3, width=2.0, height=0.5):
    xs = np.linspace(-width / 2, width / 2, nx)
    ys = np.linspace(-height / 2, height / 2, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    X = np.stack([XX.ravel(), YY.ravel()], axis=1)
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="xy")
    v00 = (j * nx + i).ravel(); v01 = (j * nx + i + 1).ravel()
    v10 = ((j + 1) * nx + i).ravel(); v11 = ((j + 1) * nx + i + 1).ravel()
    T = np.stack([np.stack([v00, v01, v11], 1), np.stack([v00, v11, v10], 1)], 1).reshape(-1, 3)
    return X, T


def _build(rho_aug):
    """Build a tiny full-space MFEM problem and return its block closures + a state."""
    X, T = _grid()
    n, dim = X.shape
    q = X.reshape(-1, 1)
    B = sp.sparse.identity(n * dim).tocsc()
    nz = B.shape[1]

    cI = np.arange(T.shape[0])               # all elements are cubature points here
    vol = sk.volume(X, T).reshape(-1, 1)
    mu_l, lam_l = sk.ympr_to_lame(1e4, 0.45)
    mu = np.full((T.shape[0], 1), mu_l)[cI]
    lam = np.full((T.shape[0], 1), lam_l)[cI]

    Mv = sp.sparse.kron(sk.massmatrix(X, T, rho=1e3), sp.sparse.identity(dim))
    kin_pre = B.T @ Mv @ B

    Ge = sp.sparse.kron(sk.selection_matrix(cI, T.shape[0]), sp.sparse.identity(dim * dim))
    J = sk.deformation_jacobian(X, T)
    GJB = (Ge @ J @ B).tocsc()
    GJq = Ge @ J @ q

    C, Ci = sk.symmetric_stretch_map(cI.shape[0], dim)
    w = np.kron(vol, np.array([[1.0, 1.0, 2.0]]).T)
    W = sp.sparse.diags(w.flatten())
    Wi = sp.sparse.diags(1.0 / w.flatten())
    na = Ci.shape[0]
    h = 1e-2

    # a soft pin so quad is non-trivial; gravity as a linear term
    Q = (1e3 * sp.sparse.identity(nz)).tocsc()
    b = np.ones((nz, 1)) * 0.1

    # previous/current reduced state for the backward-Euler inertia term
    rng = np.random.default_rng(0)
    z_curr = 0.01 * rng.standard_normal((nz, 1))
    z_prev = 0.01 * rng.standard_normal((nz, 1))

    def split(p):
        return p[:nz], p[nz:nz + na], p[nz + na:]

    def energy(p):
        u, a, ll = split(p)
        A = a.reshape(-1, dim * (dim + 1) // 2)
        F = (GJB @ u + GJq).reshape(-1, dim, dim)
        c = (Ci @ sk.stretch(F) - a)
        wc = w * c
        el = elastic_energy_S(A, mu, lam, vol, MATERIAL)
        kin = kinetic_energy_be(u, z_curr, z_prev, kin_pre, h)
        quad = quadratic_energy(u, Q, b)
        lag = float((ll.T @ wc).item())
        aug = 0.5 * rho_aug * float((c.T @ wc).item())
        return el + kin + quad + lag + aug

    def grad_blocks(p):
        u, a, ll = split(p)
        A = a.reshape(-1, dim * (dim + 1) // 2)
        F = (GJB @ u + GJq).reshape(-1, dim, dim)
        c = (Ci @ sk.stretch(F) - a)
        wc = w * c
        dsdz = sk.stretch_gradient_dz(u, GJB, Ci=Ci, dim=dim, GJq=GJq)
        G_u = dsdz @ W
        # penalty gradient of 0.5 rho c^T W c w.r.t u is rho * dsdz @ (W c) = rho * G_u @ c
        f_u = (kinetic_gradient_be(u, z_curr, z_prev, kin_pre, h)
               + quadratic_gradient(u, Q, b)
               + rho_aug * (G_u @ c))
        f_z = (elastic_gradient_S(A, mu, lam, vol, MATERIAL).reshape(-1, 1)
               - rho_aug * wc)            # dc/da = -I
        f_ll = wc
        return [f_u, f_z, f_ll]

    def hess_blocks(p):
        u, a, ll = split(p)
        A = a.reshape(-1, dim * (dim + 1) // 2)
        dsdz = sk.stretch_gradient_dz(u, GJB, Ci=Ci, dim=dim, GJq=GJq)
        G_u = dsdz @ W
        H_u = kinetic_hessian_be(kin_pre, h) + quadratic_hessian(Q)
        H_u = H_u + rho_aug * (G_u @ Wi @ G_u.T)
        H_z = sp.sparse.block_diag([hh for hh in elastic_hessian_S(A, mu, lam, vol, MATERIAL)])
        H_z = H_z + rho_aug * W
        G_z = -W
        G_zi = sp.sparse.diags(1.0 / G_z.diagonal())
        return [H_u, H_z, G_u, G_z, G_zi]

    # a representative perturbed state
    rng = np.random.default_rng(1)
    u = 0.03 * rng.standard_normal((nz, 1))
    a = (np.tile(np.array([[1.0, 1.0, 0.0]]), (cI.shape[0], 1)).reshape(-1, 1)
         + 0.03 * rng.standard_normal((na, 1)))
    ll = 0.05 * rng.standard_normal((na, 1))

    return dict(energy=energy, grad_blocks=grad_blocks, hess_blocks=hess_blocks,
                split=split, nz=nz, na=na, u=u, a=a, ll=ll, W=W)


@pytest.mark.parametrize("rho_aug", [0.0, 10.0])
def test_constraint_jacobian(rho_aug):
    m = _build(rho_aug)
    nz, na = m["nz"], m["na"]
    p = np.vstack([m["u"], m["a"], m["ll"]])
    G_u = m["hess_blocks"](p)[2]

    # f_ll = W c, so d(f_ll)/du should equal G_u^T
    def f_ll_of_u(u_flat):
        pp = np.vstack([u_flat.reshape(-1, 1), m["a"], m["ll"]])
        return m["grad_blocks"](pp)[2].flatten()
    J_fd = gradient_cfd(f_ll_of_u, m["u"].flatten(), FD)       # (na, nz)
    assert np.allclose(J_fd, np.asarray(G_u.todense()).T, atol=TOL)


@pytest.mark.parametrize("rho_aug", [0.0, 10.0])
def test_grad_blocks_are_merit_gradient(rho_aug):
    m = _build(rho_aug)
    nz, na = m["nz"], m["na"]
    p = np.vstack([m["u"], m["a"], m["ll"]])
    f_u, f_z, f_ll = m["grad_blocks"](p)
    H_u, H_z, G_u, G_z, G_zi = m["hess_blocks"](p)
    ll = m["ll"]

    g_fd = gradient_cfd(lambda pp: np.array([m["energy"](pp.reshape(-1, 1))]),
                        p.flatten(), FD).reshape(-1, 1)
    # merit gradient = blocks + multiplier coupling that sqp_mfem adds internally
    assert np.allclose(g_fd[:nz], f_u + G_u @ ll, atol=TOL)
    assert np.allclose(g_fd[nz:nz + na], f_z + G_z @ ll, atol=TOL)
    assert np.allclose(g_fd[nz + na:], f_ll, atol=TOL)


@pytest.mark.parametrize("rho_aug", [0.0, 10.0])
def test_Hz_is_exact_stretch_hessian(rho_aug):
    m = _build(rho_aug)
    nz, na = m["nz"], m["na"]
    p = np.vstack([m["u"], m["a"], m["ll"]])
    H_z = m["hess_blocks"](p)[1]
    G_z = m["hess_blocks"](p)[3]
    ll = m["ll"]

    # dE/da = f_z + G_z ll ; its Jacobian w.r.t a is H_z
    def dEda(a_flat):
        pp = np.vstack([m["u"], a_flat.reshape(-1, 1), m["ll"]])
        f_z = m["grad_blocks"](pp)[1]
        return (f_z + G_z @ ll).flatten()
    H_fd = gradient_cfd(dEda, m["a"].flatten(), FD)
    assert np.allclose(H_fd, np.asarray(H_z.todense()), atol=1e-4)


def test_Hu_exact_at_zero_multiplier_and_SPD():
    # At ll = 0, rho_aug = 0 the Gauss-Newton H_u equals the exact position Hessian.
    m = _build(0.0)
    nz, na = m["nz"], m["na"]
    p = np.vstack([m["u"], m["a"], np.zeros((na, 1))])
    H_u = m["hess_blocks"](p)[0]

    def dEdu(u_flat):
        pp = np.vstack([u_flat.reshape(-1, 1), m["a"], np.zeros((na, 1))])
        return m["grad_blocks"](pp)[0].flatten()
    H_fd = gradient_cfd(dEdu, m["u"].flatten(), FD)
    assert np.allclose(H_fd, np.asarray(H_u.todense()), atol=1e-4)

    # H_u must stay SPD as the augmentation grows (this is what sqp_mfem relies on).
    for rho_aug in (0.0, 10.0, 1e3):
        mm = _build(rho_aug)
        pp = np.vstack([mm["u"], mm["a"], mm["ll"]])
        Hu = np.asarray(mm["hess_blocks"](pp)[0].todense())
        assert np.linalg.eigvalsh(0.5 * (Hu + Hu.T)).min() > 0
