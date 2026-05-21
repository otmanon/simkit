"""Tests for ``simkit.energies.bending_energy``.

Bending energy for a 2-D beam, parameterised by hinge angles ``theta`` at
interior vertices. We build a simple straight beam (rest angles ``theta0 = 0``),
verify the energy is zero in the rest configuration, increases when the beam
is bent, and that the analytic gradient and Hessian match a central finite
difference of the energy in the flattened-position layout.

The global tier uses the ``_x`` suffix (position variable). The shipped
Hessian is the *true* second derivative: the Gauss-Newton term plus the
geometric ``d^2 theta / dx^2`` term, so the comparison against a finite
difference of the analytic gradient is valid away from the rest state.
"""

from __future__ import annotations

import numpy as np
import pytest

from simkit.energies.bending_energy import (
    bending_energy_x,
    bending_gradient_x,
    bending_hessian_x,
)
from simkit.gradient_cfd import gradient_cfd


FD_STEP = 1e-6
TOL = 1e-5


def _straight_beam(n_nodes: int = 5):
    """A horizontal beam with ``n_nodes`` nodes at x = 0, 1, ..., n-1."""
    X = np.zeros((n_nodes, 2))
    X[:, 0] = np.arange(n_nodes)
    # Interior nodes are hinge centers. Each hinge is (A, B, C) with B central.
    H = np.array(
        [[i, i + 1, i + 2] for i in range(n_nodes - 2)],
        dtype=np.int64,
    )
    theta0 = np.zeros((H.shape[0], 1))
    ymI = np.ones((H.shape[0], 1))
    l = np.ones((H.shape[0], 1))
    return X, H, theta0, ymI, l


def test_bending_energy_zero_at_rest_and_increases_on_bend() -> None:
    X, H, theta0, ymI, l = _straight_beam(n_nodes=5)
    x_rest = X.flatten().reshape(-1, 1)
    e_rest = float(bending_energy_x(x_rest, H, theta0, ymI, l))
    assert e_rest == pytest.approx(0.0, abs=1e-12)

    # Bend the beam: push the middle node up.
    X_bent = X.copy()
    X_bent[2, 1] += 0.5
    x_bent = X_bent.flatten().reshape(-1, 1)
    e_bent = float(bending_energy_x(x_bent, H, theta0, ymI, l))
    assert e_bent > e_rest


def test_bending_gradient_matches_fd() -> None:
    X, H, theta0, ymI, l = _straight_beam(n_nodes=5)
    # Perturb the straight beam by a small but nonzero amount so we are away
    # from the gradient zero, then test gradient vs FD of the energy.
    rng = np.random.default_rng(0)
    X_def = X + 0.05 * rng.standard_normal(X.shape)
    x = X_def.flatten().reshape(-1, 1)

    def energy_flat(x_flat: np.ndarray) -> np.ndarray:
        return np.array([float(bending_energy_x(x_flat, H, theta0, ymI, l))])

    g = np.asarray(bending_gradient_x(x, H, theta0, ymI, l)).flatten()
    g_fd = gradient_cfd(energy_flat, x.flatten(), FD_STEP).flatten()
    assert np.allclose(g, g_fd, atol=TOL)


def test_bending_hessian_matches_fd() -> None:
    # The shipped Hessian includes the geometric ``d^2 theta / dx^2`` term, so
    # it should match a finite difference of the gradient at a deformed (non
    # rest) configuration, not just at rest.
    X, H, theta0, ymI, l = _straight_beam(n_nodes=5)
    rng = np.random.default_rng(0)
    X_def = X + 0.05 * rng.standard_normal(X.shape)
    x = X_def.flatten().reshape(-1, 1)

    def grad_flat(x_flat: np.ndarray) -> np.ndarray:
        return np.asarray(
            bending_gradient_x(x_flat, H, theta0, ymI, l)
        ).flatten()

    H_ana = np.asarray(bending_hessian_x(x, H, theta0, ymI, l).todense())
    H_fd = gradient_cfd(grad_flat, x.flatten(), FD_STEP)
    assert np.allclose(H_ana, H_fd, atol=1e-4)


if __name__ == "__main__":
    test_bending_energy_zero_at_rest_and_increases_on_bend()
    test_bending_gradient_matches_fd()
    test_bending_hessian_matches_fd()