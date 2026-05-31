"""Sympy codegen + verification aid for the Smith-Goes-Kim stable Neo-Hookean energy.

Defines the energy

    psi(F) = (mu/2)(I_C - d) - (mu/2) log(I_C + 1) + (lam/2)(J - alpha)^2
    alpha  = 1 + d*mu / ((d+1)*lam)
    I_C    = ||F||_F^2 = sum_{i,j} F_{ij}^2
    J      = det(F)

symbolically for dim in {2, 3}, prints the symbolic gradient and Hessian
(raw, no costly simplify), and numerically cross-checks a vectorized numpy
reference implementation against sympy's lambdify on a random F.

This is a build aid only --- the runtime stable_neo_hookean.py imports
nothing from sympy.

Usage
-----
    py scripts/derive_stable_neo_hookean.py
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


def _energy(F: sp.Matrix, mu, lam, dim: int):
    I_C = sum(F[i, j] ** 2 for i in range(dim) for j in range(dim))
    J = F.det()
    alpha = 1 + dim * mu / ((dim + 1) * lam)
    psi = (mu / 2) * (I_C - dim) - (mu / 2) * sp.log(I_C + 1) + (lam / 2) * (J - alpha) ** 2
    return psi, I_C, J, alpha


def _cofactor(F: np.ndarray) -> np.ndarray:
    """Vectorized cofactor matrix cof(F): cof(F)_ij = dJ/dF_ij. Shape (t, dim, dim)."""
    dim = F.shape[-1]
    if dim == 2:
        cof = np.empty_like(F)
        cof[..., 0, 0] = F[..., 1, 1]
        cof[..., 0, 1] = -F[..., 1, 0]
        cof[..., 1, 0] = -F[..., 0, 1]
        cof[..., 1, 1] = F[..., 0, 0]
        return cof
    if dim == 3:
        cof = np.empty_like(F)
        cof[..., 0, 0] = F[..., 1, 1] * F[..., 2, 2] - F[..., 1, 2] * F[..., 2, 1]
        cof[..., 0, 1] = F[..., 1, 2] * F[..., 2, 0] - F[..., 1, 0] * F[..., 2, 2]
        cof[..., 0, 2] = F[..., 1, 0] * F[..., 2, 1] - F[..., 1, 1] * F[..., 2, 0]
        cof[..., 1, 0] = F[..., 0, 2] * F[..., 2, 1] - F[..., 0, 1] * F[..., 2, 2]
        cof[..., 1, 1] = F[..., 0, 0] * F[..., 2, 2] - F[..., 0, 2] * F[..., 2, 0]
        cof[..., 1, 2] = F[..., 0, 1] * F[..., 2, 0] - F[..., 0, 0] * F[..., 2, 1]
        cof[..., 2, 0] = F[..., 0, 1] * F[..., 1, 2] - F[..., 0, 2] * F[..., 1, 1]
        cof[..., 2, 1] = F[..., 0, 2] * F[..., 1, 0] - F[..., 0, 0] * F[..., 1, 2]
        cof[..., 2, 2] = F[..., 0, 0] * F[..., 1, 1] - F[..., 0, 1] * F[..., 1, 0]
        return cof
    raise ValueError("dim must be 2 or 3")


def numpy_gradient(F: np.ndarray, mu: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Reference vectorized numpy gradient (P = dpsi/dF)."""
    dim = F.shape[-1]
    I_C = (F ** 2).sum(axis=(-1, -2), keepdims=True)
    J = np.linalg.det(F).reshape(-1, 1, 1)
    mu_b = mu.reshape(-1, 1, 1)
    lam_b = lam.reshape(-1, 1, 1)
    alpha = 1.0 + dim * mu_b / ((dim + 1) * lam_b)
    cof = _cofactor(F)
    return mu_b * F * (I_C / (I_C + 1.0)) + lam_b * (J - alpha) * cof


def _print_symbolic(dim: int) -> None:
    mu = sp.Symbol("mu", positive=True)
    lam = sp.Symbol("lam", positive=True)
    F = _F_symbols(dim)
    psi, I_C, J, alpha = _energy(F, mu, lam, dim)

    print("=" * 78)
    print(f"dim = {dim}")
    print("=" * 78)
    print("psi =")
    sp.pprint(psi)
    print()
    print(f"alpha = {alpha}")
    print()
    print("dpsi/dF (column-major flatten):")
    g = sp.zeros(dim * dim, 1)
    a = 0
    for j in range(dim):
        for i in range(dim):
            g[a, 0] = sp.diff(psi, F[i, j])
            a += 1
    sp.pprint(g)
    print()
    print("d2psi/dF2 (column-major flatten, raw, no simplify):")
    H = sp.zeros(dim * dim, dim * dim)
    a = 0
    for j in range(dim):
        for i in range(dim):
            b = 0
            for jj in range(dim):
                for ii in range(dim):
                    H[a, b] = sp.diff(g[a, 0], F[ii, jj])
                    b += 1
            a += 1
    # only pprint for dim=2 (4x4); 9x9 is too wide to read
    if dim == 2:
        sp.pprint(H)
    else:
        print("  [9x9 Hessian; omitted for readability --- numerical check follows]")
    print()


def _numerical_verify(dim: int, t: int = 3, seed: int = 0) -> None:
    """Verify the numpy reference gradient matches sympy lambdified gradient."""
    rng = np.random.default_rng(seed)
    F_np = np.eye(dim)[None] + 0.1 * rng.standard_normal((t, dim, dim))
    mu_np = rng.uniform(0.5, 2.0, size=(t, 1))
    lam_np = rng.uniform(0.5, 2.0, size=(t, 1))

    # Build sympy lambdified gradient
    mu_s = sp.Symbol("mu", positive=True)
    lam_s = sp.Symbol("lam", positive=True)
    F_s = _F_symbols(dim)
    psi, *_ = _energy(F_s, mu_s, lam_s, dim)
    vars_flat = [F_s[i, j] for i in range(dim) for j in range(dim)] + [mu_s, lam_s]
    grad_exprs = [sp.diff(psi, F_s[i, j]) for i in range(dim) for j in range(dim)]
    grad_fn = sp.lambdify(vars_flat, grad_exprs, modules="numpy")

    g_sym = np.empty((t, dim, dim))
    for k in range(t):
        args = list(F_np[k].flatten()) + [float(mu_np[k, 0]), float(lam_np[k, 0])]
        g_sym[k] = np.array(grad_fn(*args)).reshape(dim, dim)

    g_np = numpy_gradient(F_np, mu_np, lam_np)
    err = np.max(np.abs(g_sym - g_np))
    print(f"dim={dim}: max gradient error (sympy vs numpy) = {err:.3e}")
    assert err < 1e-10, f"sympy/numpy gradient mismatch for dim={dim}"


def main() -> int:
    for dim in (2, 3):
        _print_symbolic(dim)
    print("=" * 78)
    print("Numerical cross-check: sympy lambdified vs vectorized numpy gradient")
    print("=" * 78)
    for dim in (2, 3):
        _numerical_verify(dim)
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
