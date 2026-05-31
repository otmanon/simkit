"""Sympy verification aid for StVK and classical Neo-Hookean energies.

Defines the two energies from the FEM-deformables course notes
(Sifakis & Barbic; http://barbic.usc.edu/femdefo/) symbolically:

    StVK:   psi(F) = mu * tr(E**2) + (lam/2) * tr(E)**2
            E      = (1/2) * (F^T F - I)

    NH:     psi(F) = (mu/2)(I_C - dim) - mu * log(J) + (lam/2) * log(J)**2
            I_C    = ||F||_F**2,  J = det(F)

Cross-checks the closed-form gradient (and Hessian) implementations against
sympy's lambdify on a random non-degenerate F. Build aid only --- the
runtime modules do not import sympy.

Usage
-----
    py scripts/derive_stvk_and_neo_hookean.py
"""
from __future__ import annotations

import sys

import numpy as np
import sympy as sp


def _F_symbols(dim: int) -> sp.Matrix:
    F = sp.zeros(dim, dim)
    for i in range(dim):
        for j in range(dim):
            F[i, j] = sp.Symbol(f"F_{i}{j}")
    return F


# --------------------------------------------------------------------------- #
# Symbolic energies                                                           #
# --------------------------------------------------------------------------- #
def stvk_psi(F: sp.Matrix, mu, lam, dim: int):
    Fmat = sp.Matrix(F)
    E = sp.Rational(1, 2) * (Fmat.T * Fmat - sp.eye(dim))
    tr_E = sum(E[i, i] for i in range(dim))
    tr_E2 = sum(E[i, j] ** 2 for i in range(dim) for j in range(dim))  # E:E
    return mu * tr_E2 + (lam / 2) * tr_E ** 2


def neo_hookean_psi(F: sp.Matrix, mu, lam, dim: int):
    I_C = sum(F[i, j] ** 2 for i in range(dim) for j in range(dim))
    J = F.det()
    return (mu / 2) * (I_C - dim) - mu * sp.log(J) + (lam / 2) * sp.log(J) ** 2


# --------------------------------------------------------------------------- #
# Reference vectorized numpy implementations (match runtime modules)          #
# --------------------------------------------------------------------------- #
def stvk_gradient_np(F, mu, lam):
    dim = F.shape[-1]
    Id = np.eye(dim)
    mu_b = mu.reshape(-1, 1, 1)
    lam_b = lam.reshape(-1, 1, 1)
    C = F.swapaxes(-1, -2) @ F
    E = 0.5 * (C - Id)
    tr_E = np.trace(E, axis1=-2, axis2=-1).reshape(-1, 1, 1)
    S = 2 * mu_b * E + lam_b * tr_E * Id
    return F @ S


def stvk_hessian_np(F, mu, lam):
    t, dim, _ = F.shape
    Id = np.eye(dim)
    mu_b = mu.reshape(-1, 1, 1, 1, 1)
    lam_b = lam.reshape(-1, 1, 1, 1, 1)
    C = F.swapaxes(-1, -2) @ F
    E = 0.5 * (C - Id)
    FFT = F @ F.swapaxes(-1, -2)
    tr_E = np.trace(E, axis1=-2, axis2=-1).reshape(-1, 1, 1, 1, 1)
    H5 = (
        2 * mu_b * np.einsum("ik,tlj->tijkl", Id, E)
        + mu_b * np.einsum("til,tkj->tijkl", F, F)
        + mu_b * np.einsum("tik,jl->tijkl", FFT, Id)
        + lam_b * np.einsum("tij,tkl->tijkl", F, F)
        + lam_b * tr_E * np.einsum("ik,jl->ijkl", Id, Id)[None]
    )
    return H5.reshape(t, dim * dim, dim * dim)


def neo_hookean_gradient_np(F, mu, lam):
    dim = F.shape[-1]
    mu_b = mu.reshape(-1, 1, 1)
    lam_b = lam.reshape(-1, 1, 1)
    J = np.linalg.det(F).reshape(-1, 1, 1)
    log_J = np.log(J)
    F_invT = np.linalg.inv(F).swapaxes(-1, -2)
    return mu_b * F + (lam_b * log_J - mu_b) * F_invT


def neo_hookean_hessian_np(F, mu, lam):
    t, dim, _ = F.shape
    Id = np.eye(dim)
    mu_b = mu.reshape(-1, 1, 1, 1, 1)
    lam_b = lam.reshape(-1, 1, 1, 1, 1)
    J = np.linalg.det(F).reshape(-1, 1, 1, 1, 1)
    log_J = np.log(J)
    F_invT = np.linalg.inv(F).swapaxes(-1, -2)
    H5 = (
        mu_b * np.einsum("ik,jl->ijkl", Id, Id)[None]
        + lam_b * np.einsum("tij,tkl->tijkl", F_invT, F_invT)
        + (mu_b - lam_b * log_J) * np.einsum("til,tkj->tijkl", F_invT, F_invT)
    )
    return H5.reshape(t, dim * dim, dim * dim)


# --------------------------------------------------------------------------- #
# Numerical cross-check                                                       #
# --------------------------------------------------------------------------- #
def _verify(name: str, psi_fn, np_grad_fn, np_hess_fn, dim: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    t = 3
    F_np = np.eye(dim)[None] + 0.1 * rng.standard_normal((t, dim, dim))
    mu_np = rng.uniform(0.5, 2.0, size=(t, 1))
    lam_np = rng.uniform(0.5, 2.0, size=(t, 1))

    mu_s = sp.Symbol("mu", positive=True)
    lam_s = sp.Symbol("lam", positive=True)
    F_s = _F_symbols(dim)
    psi = psi_fn(F_s, mu_s, lam_s, dim)
    vars_flat = [F_s[i, j] for i in range(dim) for j in range(dim)] + [mu_s, lam_s]
    grad_exprs = [sp.diff(psi, F_s[i, j]) for i in range(dim) for j in range(dim)]
    grad_fn = sp.lambdify(vars_flat, grad_exprs, modules="numpy")

    g_sym = np.empty((t, dim, dim))
    for k in range(t):
        args = list(F_np[k].flatten()) + [float(mu_np[k, 0]), float(lam_np[k, 0])]
        g_sym[k] = np.array(grad_fn(*args)).reshape(dim, dim)

    g_np = np_grad_fn(F_np, mu_np, lam_np)
    g_err = float(np.max(np.abs(g_sym - g_np)))

    # Hessian: row-major flat, (i,j)->row, (k,l)->col, all from row-major reshape.
    hess_exprs = [
        [sp.diff(grad_exprs[a], F_s[i, j]) for i in range(dim) for j in range(dim)]
        for a in range(dim * dim)
    ]
    hess_fn = sp.lambdify(vars_flat, hess_exprs, modules="numpy")
    H_sym = np.empty((t, dim * dim, dim * dim))
    for k in range(t):
        args = list(F_np[k].flatten()) + [float(mu_np[k, 0]), float(lam_np[k, 0])]
        H_sym[k] = np.array(hess_fn(*args))
    H_np = np_hess_fn(F_np, mu_np, lam_np)
    h_err = float(np.max(np.abs(H_sym - H_np)))

    print(f"{name:>8s} dim={dim}: grad err {g_err:.3e},  hess err {h_err:.3e}")
    assert g_err < 1e-10, f"{name} dim={dim}: gradient mismatch"
    assert h_err < 1e-10, f"{name} dim={dim}: Hessian mismatch"


def main() -> int:
    print("Cross-checking vectorized numpy implementations vs sympy:")
    for dim in (2, 3):
        _verify("StVK", stvk_psi, stvk_gradient_np, stvk_hessian_np, dim, seed=0)
        _verify("NH",   neo_hookean_psi, neo_hookean_gradient_np, neo_hookean_hessian_np, dim, seed=1)
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
