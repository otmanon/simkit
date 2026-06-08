"""Tests for ``simkit.integrators``.

Three time integrators advance ``M x'' = -grad V(x)`` by one step from a
position history:

* ``backward_euler`` -- implicit, 1st order, unconditionally stable, damps.
* ``bdf2`` -- implicit, 2nd order, stable, barely damps.
* ``forward_euler`` -- explicit, 1st order, conditionally stable, carries v.

The implicit pair build a Newton solve from the supplied potential plus an
inertial term; the explicit one needs neither a hessian nor solver info. We
check them on a harmonic oscillator with a known analytic solution.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.sparse as sps

from simkit.integrators import backward_euler, bdf2, forward_euler


# --------------------------------------------------------------------------- #
# A diagonal spring system V(x) = 0.5 x^T K x with diagonal mass M.            #
# Each dof is an independent SHO with frequency omega_i = sqrt(k_i / m_i).     #
# --------------------------------------------------------------------------- #
@pytest.fixture
def sho():
    k = np.array([4.0, 1.0, 9.0])
    m = np.array([1.0, 2.0, 1.0])
    K = sps.diags(k).tocsc()
    M = sps.diags(m).tocsc()

    def energy(x):
        x = x.reshape(-1, 1)
        return float((0.5 * x.T @ (K @ x)).item())

    def gradient(x):
        return (K @ x.reshape(-1, 1)).reshape(-1, 1)

    def hessian(x):
        return K

    omega = np.sqrt(k / m)
    return dict(K=K, M=M, m=m, omega=omega, energy=energy,
                gradient=gradient, hessian=hessian)


def _analytic(x0, omega, t):
    """Position of an undamped SHO released from rest at ``x0``."""
    return (x0.flatten() * np.cos(omega * t)).reshape(-1, 1)


# --------------------------------------------------------------------------- #
# Shapes / return contracts                                                   #
# --------------------------------------------------------------------------- #
def test_backward_euler_returns_column(sho):
    x = np.array([[1.0], [0.0], [-0.5]])
    x_next = backward_euler(x, x, sho["energy"], sho["gradient"],
                            sho["hessian"], sho["M"], 1e-3, max_iter=10)
    assert x_next.shape == (3, 1)
    assert np.isfinite(x_next).all()


def test_bdf2_returns_column(sho):
    x = np.array([[1.0], [0.0], [-0.5]])
    x_next = bdf2(x, x, x, x, sho["energy"], sho["gradient"],
                  sho["hessian"], sho["M"], 1e-3, max_iter=10)
    assert x_next.shape == (3, 1)
    assert np.isfinite(x_next).all()


def test_forward_euler_returns_position_and_velocity(sho):
    x = np.array([[1.0], [0.0], [-0.5]])
    v = np.zeros((3, 1))
    out = forward_euler(x, v, sho["gradient"], sho["M"], 1e-3)
    assert isinstance(out, tuple) and len(out) == 2
    x_next, v_next = out
    assert x_next.shape == (3, 1) and v_next.shape == (3, 1)


def test_implicit_return_info_passthrough(sho):
    x = np.array([[1.0], [0.0], [-0.5]])
    x_next, info = backward_euler(x, x, sho["energy"], sho["gradient"],
                                  sho["hessian"], sho["M"], 1e-3,
                                  max_iter=5, return_info=True)
    assert x_next.shape == (3, 1)
    assert set(info.keys()) == {"g", "dx", "alphas", "iters"}


# --------------------------------------------------------------------------- #
# Physical accuracy against the analytic oscillator                           #
# --------------------------------------------------------------------------- #
def _roll_backward_euler(sho, x0, h, n_steps):
    x_prev = x0.copy()
    x_curr = x0.copy()
    for _ in range(n_steps):
        x_next = backward_euler(x_curr, x_prev, sho["energy"], sho["gradient"],
                                sho["hessian"], sho["M"], h, max_iter=20)
        x_prev, x_curr = x_curr, x_next
    return x_curr


def _roll_bdf2(sho, x0, h, n_steps):
    # Seed the four-level history from the exact analytic trajectory (the
    # constant-step BDF2 stencil needs a consistent start to stay 2nd order).
    omega = sho["omega"]
    xc = _analytic(x0, omega, 0.0)
    xp1 = _analytic(x0, omega, -h)
    xp2 = _analytic(x0, omega, -2.0 * h)
    xp3 = _analytic(x0, omega, -3.0 * h)
    for _ in range(n_steps):
        x_next = bdf2(xc, xp1, xp2, xp3, sho["energy"], sho["gradient"],
                      sho["hessian"], sho["M"], h, max_iter=20)
        xp3, xp2, xp1, xc = xp2, xp1, xc, x_next
    return xc


def _roll_forward_euler(sho, x0, h, n_steps):
    x = x0.copy()
    v = np.zeros_like(x0)
    for _ in range(n_steps):
        x, v = forward_euler(x, v, sho["gradient"], sho["M"], h)
    return x


def test_all_integrators_track_analytic_oscillator(sho):
    x0 = np.array([[1.0], [0.7], [-0.3]])
    h = 1e-3
    n = 100
    t = h * n
    ref = _analytic(x0, sho["omega"], t)

    be = _roll_backward_euler(sho, x0, h, n)
    b2 = _roll_bdf2(sho, x0, h, n)
    fe = _roll_forward_euler(sho, x0, h, n)

    # At a small timestep every scheme is close to the exact solution.
    assert np.linalg.norm(be - ref) < 5e-2
    assert np.linalg.norm(b2 - ref) < 5e-2
    assert np.linalg.norm(fe - ref) < 5e-2


def test_bdf2_more_accurate_than_backward_euler(sho):
    x0 = np.array([[1.0], [0.7], [-0.3]])
    h = 5e-3
    n = 100
    ref = _analytic(x0, sho["omega"], h * n)
    err_be = np.linalg.norm(_roll_backward_euler(sho, x0, h, n) - ref)
    err_b2 = np.linalg.norm(_roll_bdf2(sho, x0, h, n) - ref)
    assert err_b2 < err_be


def test_backward_euler_order_one(sho):
    """Halving h roughly halves backward-Euler error (1st order)."""
    x0 = np.array([[1.0], [0.7], [-0.3]])
    T = 0.5
    errs = []
    for h in (4e-3, 2e-3, 1e-3):
        n = int(round(T / h))
        ref = _analytic(x0, sho["omega"], h * n)
        errs.append(np.linalg.norm(_roll_backward_euler(sho, x0, h, n) - ref))
    # Empirical order from the finest pair; ~1 for backward Euler.
    p = math.log(errs[0] / errs[2]) / math.log(4.0)
    assert 0.7 < p < 1.4


def test_bdf2_order_two(sho):
    """Halving h roughly quarters BDF2 error (2nd order), seeded exactly."""
    x0 = np.array([[1.0], [0.7], [-0.3]])
    T = 0.5
    errs = []
    for h in (4e-3, 2e-3, 1e-3):
        n = int(round(T / h))
        ref = _analytic(x0, sho["omega"], h * n)
        errs.append(np.linalg.norm(_roll_bdf2(sho, x0, h, n) - ref))
    p = math.log(errs[0] / errs[2]) / math.log(4.0)
    assert 1.6 < p < 2.4


def test_backward_euler_dissipates_energy_monotonically(sho):
    """Backward Euler never *adds* mechanical energy to a free oscillation."""
    x0 = np.array([[1.0], [0.7], [-0.3]])
    h = 0.02

    def total_energy(x_curr, x_prev):
        v = (x_curr - x_prev) / h
        return sho["energy"](x_curr) + 0.5 * float((v.T @ (sho["M"] @ v)).item())

    x_prev = x0.copy()
    x_curr = x0.copy()
    e_prev = total_energy(x_curr, x_prev)
    energies_seen = [e_prev]
    for _ in range(200):
        x_next = backward_euler(x_curr, x_prev, sho["energy"], sho["gradient"],
                                sho["hessian"], sho["M"], h, max_iter=20)
        x_prev, x_curr = x_curr, x_next
        e = total_energy(x_curr, x_prev)
        assert e <= e_prev + 1e-9        # monotone, non-increasing
        e_prev = e
        energies_seen.append(e)
    # And it really has bled energy away, not just held flat.
    assert energies_seen[-1] < 0.95 * energies_seen[0]


def test_forward_euler_uses_lumped_mass_and_force(sho):
    """One explicit step matches the closed-form update x+hv, v+h M^-1 f."""
    x = np.array([[1.0], [0.5], [-0.2]])
    v = np.array([[0.1], [-0.3], [0.4]])
    h = 1e-3
    x_next, v_next = forward_euler(x, v, sho["gradient"], sho["M"], h)

    f = -sho["gradient"](x)
    a = f / sho["m"].reshape(-1, 1)
    np.testing.assert_allclose(x_next, x + h * v, rtol=0, atol=1e-12)
    np.testing.assert_allclose(v_next, v + h * a, rtol=0, atol=1e-12)
