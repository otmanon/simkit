"""Tests for ``simkit.energies.contact_springs_plane``.

A spring-style penalty that activates per-vertex when a vertex is below
the half-space defined by ``(p, n)``. Above the plane the energy is zero;
below the plane it grows quadratically in the signed distance.

For finite-difference checks we deliberately push some vertices below the
plane so the gradient and Hessian are non-zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from simkit.energies.contact_springs_plane import (
    contact_springs_plane_energy,
    contact_springs_plane_gradient,
    contact_springs_plane_hessian,
)
from simkit.gradient_cfd import gradient_cfd


FD_STEP = 1e-6
GRAD_TOL = 1e-6
HESS_TOL = 1e-6


def _setup(rng: np.random.Generator, n_points: int, dim: int):
    p = np.zeros(dim)
    n = np.zeros(dim)
    n[-1] = 1.0
    X_above = np.abs(rng.standard_normal((n_points, dim))) + 0.5
    return X_above, p, n


@pytest.mark.parametrize("dim", [2, 3])
def test_contact_springs_plane_energy_increases_below_plane(dim: int) -> None:
    rng = np.random.default_rng(0)
    X_above, p, n = _setup(rng, n_points=4, dim=dim)
    X_below = X_above.copy()
    X_below[:, -1] = -np.abs(X_below[:, -1]) - 0.1

    k = 1.5
    e_above = float(contact_springs_plane_energy(X_above, k, p, n))
    e_below = float(contact_springs_plane_energy(X_below, k, p, n))

    assert e_above == pytest.approx(0.0, abs=1e-12)
    assert e_below > e_above


@pytest.mark.parametrize("dim", [2, 3])
def test_contact_springs_plane_gradient_matches_fd(dim: int) -> None:
    rng = np.random.default_rng(1)
    _, p, n = _setup(rng, n_points=3, dim=dim)
    X = rng.standard_normal((3, dim))
    X[:, -1] = -0.5 - 0.1 * rng.standard_normal(X.shape[0])
    n_pts, _ = X.shape
    k = 1.7

    def energy_flat(x_flat: np.ndarray) -> np.ndarray:
        return np.array(
            [float(contact_springs_plane_energy(x_flat.reshape(n_pts, dim), k, p, n))]
        )

    g_fd = gradient_cfd(energy_flat, X.flatten(), FD_STEP).flatten()
    g = np.asarray(contact_springs_plane_gradient(X, k, p, n)).flatten()

    assert np.allclose(g, g_fd, atol=GRAD_TOL)


@pytest.mark.parametrize("dim", [2, 3])
def test_contact_springs_plane_hessian_matches_fd(dim: int) -> None:
    rng = np.random.default_rng(2)
    _, p, n = _setup(rng, n_points=2, dim=dim)
    X = rng.standard_normal((2, dim))
    X[:, -1] = -0.5 - 0.1 * rng.standard_normal(X.shape[0])
    n_pts, _ = X.shape
    k = 0.8

    def grad_flat(x_flat: np.ndarray) -> np.ndarray:
        return np.asarray(
            contact_springs_plane_gradient(x_flat.reshape(n_pts, dim), k, p, n)
        ).flatten()

    H_fd = gradient_cfd(grad_flat, X.flatten(), FD_STEP)
    H = np.asarray(contact_springs_plane_hessian(X, k, p, n).todense())

    assert np.allclose(H, H_fd, atol=HESS_TOL)