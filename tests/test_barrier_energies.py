"""Tests for ``simkit.energies.barrier_energies``.

The barrier-energy module exposes three scalar barrier potentials defined on a
per-element distance ``d`` and an activation threshold ``d_hat``:

* ``quadratic_barrier``  -- ``(d - d_hat)**2``           for ``d < d_hat``
* ``cubic_barrier``       -- ``|d - d_hat|**3``           for ``d < d_hat``
* ``ipc_barrier``         -- ``-(d - d_hat)**2 log(d/d_hat)`` for ``d < d_hat``

It also exposes two friction interpolants used by IPC-style contact:

* ``sticking_friction_interpolant``   -- a simple quadratic in ``y``.
* ``stick_slip_friction_interpolant`` -- a piece-wise smooth ramp.

All of these functions return per-element densities; the gradient and Hessian
are the per-element ``d/d d`` and ``d^2/d d^2`` of the density and so should
match a 1-D central finite difference of the energy density (taken
element-wise).
"""

from __future__ import annotations

import numpy as np
import pytest

from simkit.energies.barrier_energies import (
    quadratic_barrier_energy,
    quadratic_barrier_gradient,
    quadratic_barrier_hessian,
    cubic_barrier_energy,
    cubic_barrier_gradient,
    cubic_barrier_hessian,
    ipc_barrier_energy,
    ipc_barrier_gradient,
    ipc_barrier_hessian,
    sticking_friction_interpolant,
    sticking_friction_interpolant_gradient,
    sticking_friction_interpolant_hessian,
    stick_slip_friction_interpolant,
    stick_slip_friction_interpolant_gradient,
    stick_slip_friction_interpolant_hessian,
)


FD_STEP = 1e-6
TOL = 1e-5


def _fd_diag(f, d, h=FD_STEP):
    """Element-wise central finite difference of a scalar-per-element map.

    ``f`` is expected to take a 1-D array of length ``n`` and return either a
    1-D array of length ``n`` or a column ``(n, 1)`` array.
    """
    d = d.reshape(-1, 1).copy()
    n = d.shape[0]
    out = np.zeros((n, 1))
    for i in range(n):
        dp = d.copy()
        dm = d.copy()
        dp[i, 0] += h
        dm[i, 0] -= h
        fp = np.asarray(f(dp)).reshape(-1, 1)
        fm = np.asarray(f(dm)).reshape(-1, 1)
        out[i, 0] = (fp[i, 0] - fm[i, 0]) / (2.0 * h)
    return out


# ---------------------------------------------------------------------------
# Quadratic barrier
# ---------------------------------------------------------------------------


def test_quadratic_barrier_energy_increases_when_close() -> None:
    d_hat = 1.0
    d_far = np.array([2.0, 1.5, 3.0])
    d_near = np.array([0.5, 0.2, 0.1])

    e_far = np.asarray(quadratic_barrier_energy(d_far, d_hat)).sum()
    e_near = np.asarray(quadratic_barrier_energy(d_near, d_hat)).sum()

    assert e_far == pytest.approx(0.0, abs=1e-12)
    assert e_near > e_far


def test_quadratic_barrier_gradient_matches_fd() -> None:
    d_hat = 1.0
    d = np.array([0.2, 0.5, 0.8])
    g = np.asarray(quadratic_barrier_gradient(d, d_hat)).reshape(-1, 1)
    g_fd = _fd_diag(lambda x: quadratic_barrier_energy(x, d_hat), d)
    assert np.allclose(g, g_fd, atol=TOL)


def test_quadratic_barrier_hessian_matches_fd() -> None:
    d_hat = 1.0
    d = np.array([0.2, 0.5, 0.8])
    h_ana = np.asarray(quadratic_barrier_hessian(d, d_hat)).reshape(-1, 1)
    h_fd = _fd_diag(lambda x: quadratic_barrier_gradient(x, d_hat), d)
    assert np.allclose(h_ana, h_fd, atol=TOL)


# ---------------------------------------------------------------------------
# Cubic barrier
# ---------------------------------------------------------------------------


def test_cubic_barrier_energy_increases_when_close() -> None:
    d_hat = 1.0
    d_far = np.array([2.0, 1.5, 3.0])
    d_near = np.array([0.5, 0.2, 0.1])

    e_far = np.asarray(cubic_barrier_energy(d_far, d_hat)).sum()
    e_near = np.asarray(cubic_barrier_energy(d_near, d_hat)).sum()

    assert e_far == pytest.approx(0.0, abs=1e-12)
    assert e_near > e_far


def test_cubic_barrier_gradient_matches_fd() -> None:
    d_hat = 1.0
    d = np.array([0.2, 0.5, 0.8])
    g = np.asarray(cubic_barrier_gradient(d, d_hat)).reshape(-1, 1)
    g_fd = _fd_diag(lambda x: cubic_barrier_energy(x, d_hat), d)
    assert np.allclose(g, g_fd, atol=TOL)


def test_cubic_barrier_hessian_matches_fd() -> None:
    d_hat = 1.0
    d = np.array([0.2, 0.5, 0.8])
    h_ana = np.asarray(cubic_barrier_hessian(d, d_hat)).reshape(-1, 1)
    h_fd = _fd_diag(lambda x: cubic_barrier_gradient(x, d_hat), d)
    assert np.allclose(h_ana, h_fd, atol=TOL)


# ---------------------------------------------------------------------------
# IPC log barrier
# ---------------------------------------------------------------------------


def test_ipc_barrier_energy_increases_when_close() -> None:
    d_hat = 1.0
    d_far = np.array([2.0, 1.5, 3.0])
    d_near = np.array([0.5, 0.2, 0.1])

    e_far = np.asarray(ipc_barrier_energy(d_far, d_hat)).sum()
    e_near = np.asarray(ipc_barrier_energy(d_near, d_hat)).sum()

    assert e_far == pytest.approx(0.0, abs=1e-12)
    assert e_near > e_far


def test_ipc_barrier_gradient_matches_fd() -> None:
    d_hat = 1.0
    d = np.array([0.2, 0.5, 0.8])
    g = np.asarray(ipc_barrier_gradient(d, d_hat)).reshape(-1, 1)
    g_fd = _fd_diag(lambda x: ipc_barrier_energy(x, d_hat), d)
    assert np.allclose(g, g_fd, atol=TOL)


def test_ipc_barrier_hessian_matches_fd() -> None:
    d_hat = 1.0
    d = np.array([0.2, 0.5, 0.8])
    h_ana = np.asarray(ipc_barrier_hessian(d, d_hat)).reshape(-1, 1)
    h_fd = _fd_diag(lambda x: ipc_barrier_gradient(x, d_hat), d)
    assert np.allclose(h_ana, h_fd, atol=TOL)


# ---------------------------------------------------------------------------
# Sticking friction interpolant
# ---------------------------------------------------------------------------


def test_sticking_friction_increases_with_y() -> None:
    eps_v, h = 0.1, 1.0
    y_small = np.array([0.0, 0.0, 0.0]).reshape(-1, 1)
    y_large = np.array([0.5, 1.0, 1.5]).reshape(-1, 1)

    e_small = float(np.asarray(sticking_friction_interpolant(y_small)).sum())
    e_large = float(np.asarray(sticking_friction_interpolant(y_large)).sum())

    assert e_small == pytest.approx(0.0, abs=1e-12)
    assert e_large > e_small


def test_sticking_friction_gradient_matches_fd() -> None:

    y = np.array([0.2, 0.5, 1.2]).reshape(-1, 1)
    g = np.asarray(sticking_friction_interpolant_gradient(y)).reshape(-1, 1)
    g_fd = _fd_diag(lambda x: sticking_friction_interpolant(x), y)
    assert np.allclose(g, g_fd, atol=TOL)


def test_sticking_friction_hessian_matches_fd() -> None:

    y = np.array([0.2, 0.5, 1.2]).reshape(-1, 1)
    h_ana = np.asarray(sticking_friction_interpolant_hessian(y)).reshape(-1, 1)
    h_fd = _fd_diag(lambda x: sticking_friction_interpolant_gradient(x), y)
    assert np.allclose(h_ana, h_fd, atol=TOL)


# ---------------------------------------------------------------------------
# Stick-slip friction interpolant
# ---------------------------------------------------------------------------


def test_stick_slip_friction_increases_with_y() -> None:
    eps_v, h = 0.1, 1.0
    y_small = np.array([1e-3, 2e-3]).reshape(-1, 1)
    y_large = np.array([0.5, 1.0]).reshape(-1, 1)
    e_small = float(np.asarray(stick_slip_friction_interpolant(y_small, eps_v, h)).sum())
    e_large = float(np.asarray(stick_slip_friction_interpolant(y_large, eps_v, h)).sum())
    assert e_large > e_small


def test_stick_slip_friction_gradient_matches_fd() -> None:
    eps_v, h = 0.1, 1.0
    # Sample below the threshold (h*eps_v = 0.1) and above it so we exercise both
    # branches of the piecewise interpolant.
    y = np.array([0.02, 0.05, 0.3, 0.8]).reshape(-1, 1)
    g = np.asarray(stick_slip_friction_interpolant_gradient(y, eps_v, h)).reshape(-1, 1)
    g_fd = _fd_diag(lambda x: stick_slip_friction_interpolant(x, eps_v, h), y)
    assert np.allclose(g, g_fd, atol=TOL)


def test_stick_slip_friction_hessian_matches_fd() -> None:
    eps_v, h = 0.1, 1.0
    # Stay strictly below the threshold so the hessian branch is non-trivial
    # and the function is smooth in the FD stencil.
    y = np.array([0.02, 0.05, 0.08]).reshape(-1, 1)
    h_ana = np.asarray(stick_slip_friction_interpolant_hessian(y, eps_v, h)).reshape(-1, 1)
    h_fd = _fd_diag(lambda x: stick_slip_friction_interpolant_gradient(x, eps_v, h), y)
    assert np.allclose(h_ana, h_fd, atol=TOL)