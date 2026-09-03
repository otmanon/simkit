"""Tests for the flat solver functions in ``simkit.solvers``.

Each solver minimizes a simple convex quadratic ``0.5 x^T Q x + b^T x`` whose
unique minimizer is ``x* = -Q^{-1} b``, so we can check convergence directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from simkit.solvers import (
    newton_solver,
    gradient_descent,
    block_coord,
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
    x = gradient_descent(np.zeros_like(b), energy, gradient,
                                max_iter=2000, tolerance=1e-10, do_line_search=True)
    assert np.allclose(x, x_star, atol=1e-5)


def test_block_coord_solver_runs_to_fixed_point() -> None:
    # A trivial local/global pair whose fixed point is the target ``t``.
    t = np.array([[1.0], [-2.0], [3.0]])
    local_step = lambda x: t - x          # auxiliary residual
    global_step = lambda x, r: x + r       # move all the way to the target
    x = block_coord(np.zeros_like(t), global_step, local_step,
                           tolerance=1e-12, max_iter=10)
    assert np.allclose(x, t, atol=1e-10)


# --------------------------------------------------------------------------- #
# Export surface
# --------------------------------------------------------------------------- #
def test_all_solvers_are_exported() -> None:
    """The documented solver set must stay importable from the package."""
    from simkit import solvers

    for name in ("newton_solver", "gradient_descent", "block_coord",
                 "sqp_mfem", "local_global", "cmaes"):
        assert callable(getattr(solvers, name)), name


def test_solvers_available_without_optional_dependencies() -> None:
    """``simkit.solvers`` is numpy/scipy only -- ``cma`` is guarded internally.

    Regression test: the package used to be imported inside the ``[cmaes]``
    optional bucket in ``simkit/__init__.py``, which made *every* solver
    disappear on a lean install whenever ``cma`` was absent.
    """
    import simkit

    assert callable(simkit.solvers.newton_solver)


# --------------------------------------------------------------------------- #
# block_coord: step ordering and argument handling
# --------------------------------------------------------------------------- #
def test_block_coord_alternates_local_then_global() -> None:
    trace = []

    def local(x):
        trace.append("local")
        return -0.5 * x

    def global_(x, r):
        trace.append("global")
        return x + r

    x = block_coord(np.array([[8.0]]), global_, local, max_iter=3)
    assert trace == ["local", "global"] * 3
    # x <- x - x/2 halves each iteration: 8 -> 4 -> 2 -> 1
    assert np.allclose(x, 1.0)


def test_block_coord_does_not_mutate_the_initial_guess() -> None:
    x0 = np.array([[2.0], [3.0]])
    backup = x0.copy()
    block_coord(x0, lambda x, r: x + r, lambda x: -0.1 * x, max_iter=5)
    assert np.array_equal(x0, backup)


# --------------------------------------------------------------------------- #
# local_global (not implemented)
# --------------------------------------------------------------------------- #
def test_local_global_raises_not_implemented() -> None:
    """It used to silently return ``x + 1``; it must not return garbage."""
    from simkit.solvers import local_global

    with pytest.raises(NotImplementedError, match="block_coord"):
        local_global(np.zeros((3, 1)), lambda x: x, lambda x: x)


# --------------------------------------------------------------------------- #
# cmaes
# --------------------------------------------------------------------------- #
def test_cmaes_minimizes_a_simple_objective() -> None:
    """Derivative-free minimization of a separable quadratic."""
    pytest.importorskip("cma")  # pip install 'simkit[cmaes]'
    from simkit.solvers import cmaes

    objective = lambda x: float(np.sum((np.asarray(x).ravel() - 0.5) ** 2))
    x0 = np.zeros(3)
    x = cmaes(x0, objective, maxiter=60, sigma=0.5, popsize=12, seed=0)
    assert objective(x) < objective(x0)


def test_cmaes_default_call_does_not_raise() -> None:
    """Regression: the single-process branch appended to ``running_history``
    unconditionally, so any call with ``return_history=False`` (the default)
    died with ``UnboundLocalError``."""
    pytest.importorskip("cma")
    from simkit.solvers import cmaes

    objective = lambda x: float(np.sum(np.asarray(x).ravel() ** 2))
    out = cmaes(np.zeros(2), objective, maxiter=3, sigma=0.3, popsize=6, seed=0)
    assert not isinstance(out, tuple)


def test_cmaes_return_result_and_history_options() -> None:
    pytest.importorskip("cma")
    from simkit.solvers import cmaes

    objective = lambda x: float(np.sum(np.asarray(x).ravel() ** 2))
    kw = dict(maxiter=3, sigma=0.3, popsize=6, seed=0)

    _, hist = cmaes(np.zeros(2), objective, return_history=True, **kw)
    assert isinstance(hist, list) and hist

    _, res = cmaes(np.zeros(2), objective, return_result=True, **kw)
    assert isinstance(res, dict)

    _, res, hist = cmaes(np.zeros(2), objective, return_result=True,
                         return_history=True, **kw)
    assert isinstance(res, dict) and isinstance(hist, list)


def test_cmaes_without_cma_raises_with_install_hint(monkeypatch) -> None:
    """Missing ``cma`` must fail at call time with a usable message, rather
    than at import time -- that is what keeps ``simkit.solvers`` lean."""
    import importlib

    mod = importlib.import_module("simkit.solvers.cmaes")
    monkeypatch.setattr(mod, "CMA_AVAILABLE", False)
    with pytest.raises(ImportError, match="cma"):
        mod.cmaes(np.zeros(2), lambda x: 0.0, maxiter=1)


# --------------------------------------------------------------------------- #
# sqp_mfem
# --------------------------------------------------------------------------- #
def test_sqp_mfem_public_signature() -> None:
    """Exercised end-to-end by the subspace-MFEM example, which needs the data
    submodule; here we just pin the calling convention."""
    import inspect

    from simkit.solvers import sqp_mfem

    sig = inspect.signature(sqp_mfem)
    assert list(sig.parameters)[:4] == [
        "p0", "energy_func", "hess_blocks_func", "grad_blocks_func"
    ]
    assert sig.parameters["max_iter"].default == 100
    assert sqp_mfem.__doc__
