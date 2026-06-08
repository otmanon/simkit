"""Tests for the flat solver functions in ``simkit.solvers``.

Each solver minimizes a simple convex quadratic ``0.5 x^T Q x + b^T x`` whose
unique minimizer is ``x* = -Q^{-1} b``, so we can check convergence directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from simkit.solvers import (
    newton_solver,
    gradient_descent_solver,
    block_coord_solver,
)


def _quadratic(n: int, seed: int):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    Q = A.T @ A + n * np.eye(n)
    b = rng.standard_normal((n, 1))
    x_star = -np.linalg.solve(Q, b)

    energy = lambda x: float((0.5 * x.T @ Q @ x + b.T @ x).item())
    gradient = lambda x: Q @ x + b
    hessian = lambda x: Q
    return Q, b, x_star, energy, gradient, hessian


def test_newton_solver_converges_in_one_step() -> None:
    # Newton's method is exact on a quadratic in a single step.
    Q, b, x_star, energy, gradient, hessian = _quadratic(n=6, seed=0)
    x = newton_solver(np.zeros_like(b), energy, gradient, hessian,
                      max_iter=1, do_line_search=True)
    assert np.allclose(x, x_star, atol=1e-8)


def test_newton_solver_no_line_search() -> None:
    Q, b, x_star, energy, gradient, hessian = _quadratic(n=4, seed=1)
    x = newton_solver(np.zeros_like(b), energy, gradient, hessian,
                      max_iter=5, do_line_search=False)
    assert np.allclose(x, x_star, atol=1e-8)


def test_newton_solver_return_info() -> None:
    Q, b, x_star, energy, gradient, hessian = _quadratic(n=3, seed=2)
    x, info = newton_solver(np.zeros_like(b), energy, gradient, hessian,
                            max_iter=10, return_info=True)
    assert np.allclose(x, x_star, atol=1e-8)
    assert info["iters"] >= 0
    assert len(info["g"]) == info["iters"] + 1
    assert len(info["alphas"]) == info["iters"] + 1


def test_gradient_descent_solver_converges() -> None:
    Q, b, x_star, energy, gradient, hessian = _quadratic(n=5, seed=3)
    x = gradient_descent_solver(np.zeros_like(b), energy, gradient,
                                max_iter=2000, tolerance=1e-10, do_line_search=True)
    assert np.allclose(x, x_star, atol=1e-5)


def test_block_coord_solver_runs_to_fixed_point() -> None:
    # A trivial local/global pair whose fixed point is the target ``t``.
    t = np.array([[1.0], [-2.0], [3.0]])
    local_step = lambda x: t - x          # auxiliary residual
    global_step = lambda x, r: x + r       # move all the way to the target
    x = block_coord_solver(np.zeros_like(t), global_step, local_step,
                           tolerance=1e-12, max_iter=10)
    assert np.allclose(x, t, atol=1e-10)
