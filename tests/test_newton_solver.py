"""Tests for ``simkit.solvers.newton_solver``.

The numerics modules are finite-difference verified in their own test files;
this covers the layer every demo and integrator routes through. We check
convergence on problems with known minimizers (a quadratic, where Newton is
exact in one step, and a non-quadratic where it is not), the ``max_iter=1``
stepper default, the line search, sparse/dense Hessian handling, and the
``return_info`` diagnostics.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

from simkit.solvers import newton_solver


def _spd(n: int, seed: int = 0) -> np.ndarray:
    """A well-conditioned symmetric positive-definite matrix."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return A @ A.T + n * np.eye(n)


def _quadratic(A: np.ndarray, b: np.ndarray):
    """0.5 x^T A x - b^T x, minimized at ``A x = b``."""

    def energy(x):
        return float((0.5 * x.T @ A @ x - b.T @ x).item())

    def gradient(x):
        return A @ x - b

    def hessian(x):
        return A

    return energy, gradient, hessian


# --------------------------------------------------------------------------- #
# Convergence
# --------------------------------------------------------------------------- #
def test_quadratic_exact_in_one_step():
    """Newton solves a quadratic exactly in a single full step."""
    n = 8
    A = _spd(n)
    b = np.random.default_rng(1).standard_normal((n, 1))
    energy, gradient, hessian = _quadratic(A, b)
    x_star = np.linalg.solve(A, b)

    x = newton_solver(np.zeros((n, 1)), energy, gradient, hessian,
                      max_iter=1, do_line_search=False)
    assert np.allclose(x, x_star, atol=1e-10)


def test_quadratic_with_line_search_still_exact():
    """The line search must accept the full Newton step on a quadratic."""
    n = 6
    A = _spd(n, seed=2)
    b = np.random.default_rng(3).standard_normal((n, 1))
    energy, gradient, hessian = _quadratic(A, b)
    x_star = np.linalg.solve(A, b)

    x = newton_solver(np.zeros((n, 1)), energy, gradient, hessian,
                      max_iter=20, do_line_search=True, tolerance=1e-12)
    assert np.allclose(x, x_star, atol=1e-8)


def test_nonquadratic_converges_to_stationary_point():
    """On a non-quadratic energy the gradient must vanish at the answer.

    ``sum(cosh(x_i) - 1)`` is strictly convex with its unique minimum at 0,
    but Newton needs several iterations to get there.
    """
    n = 5

    def energy(x):
        return float(np.sum(np.cosh(x) - 1.0))

    def gradient(x):
        return np.sinh(x)

    def hessian(x):
        return np.diag(np.cosh(x).ravel())

    x0 = np.full((n, 1), 0.8)
    x = newton_solver(x0, energy, gradient, hessian,
                      max_iter=50, tolerance=1e-12)

    assert np.allclose(x, 0.0, atol=1e-8)
    assert np.linalg.norm(gradient(x)) < 1e-8


def test_more_iterations_never_worsen_the_energy():
    """Newton with a line search is monotone: energy must not increase."""
    n = 5

    def energy(x):
        return float(np.sum(np.cosh(x) - 1.0))

    def gradient(x):
        return np.sinh(x)

    def hessian(x):
        return np.diag(np.cosh(x).ravel())

    x0 = np.full((n, 1), 1.2)
    energies = [
        energy(newton_solver(x0, energy, gradient, hessian, max_iter=k,
                             tolerance=1e-14))
        for k in range(1, 8)
    ]
    assert all(b <= a + 1e-12 for a, b in zip(energies, energies[1:]))


# --------------------------------------------------------------------------- #
# The max_iter=1 default (documented stepper behaviour)
# --------------------------------------------------------------------------- #
def test_default_max_iter_is_a_single_step():
    """``max_iter`` defaults to 1: one step, not a solve to convergence.

    Pinning this because it is the library's most surprising default -- the
    implicit integrators rely on it.
    """
    n = 4
    counter = {"grad": 0}

    def energy(x):
        return float(np.sum(np.cosh(x) - 1.0))

    def gradient(x):
        counter["grad"] += 1
        return np.sinh(x)

    def hessian(x):
        return np.diag(np.cosh(x).ravel())

    x0 = np.full((n, 1), 1.5)
    x = newton_solver(x0, energy, gradient, hessian)

    assert counter["grad"] == 1, "default must take exactly one Newton step"
    # One step from 1.5 makes progress but does not reach the minimum.
    assert np.linalg.norm(x) < np.linalg.norm(x0)
    assert np.linalg.norm(x) > 1e-6


# --------------------------------------------------------------------------- #
# Hessian representations
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sparse", [False, True])
def test_dense_and_sparse_hessians_agree(sparse):
    """A sparse Hessian must produce the same iterate as its dense twin."""
    n = 10
    A = _spd(n, seed=4)
    b = np.random.default_rng(5).standard_normal((n, 1))
    energy, gradient, dense_hess = _quadratic(A, b)
    hessian = (lambda x: sps.csc_matrix(A)) if sparse else dense_hess

    x = newton_solver(np.zeros((n, 1)), energy, gradient, hessian,
                      max_iter=5, tolerance=1e-12)
    assert np.allclose(x, np.linalg.solve(A, b), atol=1e-8)


def test_does_not_mutate_the_initial_guess():
    """``x0`` is copied, not stepped in place."""
    n = 4
    A = _spd(n, seed=6)
    b = np.ones((n, 1))
    energy, gradient, hessian = _quadratic(A, b)

    x0 = np.zeros((n, 1))
    x0_backup = x0.copy()
    newton_solver(x0, energy, gradient, hessian, max_iter=5)
    assert np.array_equal(x0, x0_backup)


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def test_return_info_reports_iterations_and_history():
    n = 5

    def energy(x):
        return float(np.sum(np.cosh(x) - 1.0))

    def gradient(x):
        return np.sinh(x)

    def hessian(x):
        return np.diag(np.cosh(x).ravel())

    x, info = newton_solver(np.full((n, 1), 0.9), energy, gradient, hessian,
                            max_iter=50, tolerance=1e-12, return_info=True)

    assert set(info) == {"g", "dx", "alphas", "iters"}
    assert info["iters"] >= 1
    assert len(info["g"]) == len(info["dx"]) == len(info["alphas"])
    assert len(info["g"]) == info["iters"] + 1
    # Gradient norms decrease monotonically on this convex problem.
    norms = [np.linalg.norm(g) for g in info["g"]]
    assert all(b <= a + 1e-12 for a, b in zip(norms, norms[1:]))


def test_early_exit_before_max_iter_when_converged():
    """Tolerance must short-circuit the loop rather than burning max_iter."""
    n = 6
    A = _spd(n, seed=7)
    b = np.random.default_rng(8).standard_normal((n, 1))
    energy, gradient, hessian = _quadratic(A, b)

    _, info = newton_solver(np.zeros((n, 1)), energy, gradient, hessian,
                            max_iter=100, tolerance=1e-8, return_info=True)
    # A quadratic is solved in one step; the second step is null and exits.
    assert info["iters"] < 5
