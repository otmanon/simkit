"""Tests for ``simkit.energies.contact_springs_sphere``.

A spring-style penalty that activates per-vertex when a vertex is strictly
inside the sphere of radius ``r`` centered at ``p``. Outside the sphere
the energy is zero; inside it grows quadratically in the signed
distance-to-boundary.

The Hessian shipped in this module is a Gauss-Newton style approximation
that treats the radial normal ``(x - p)/||x - p||`` as fixed. A central
finite-difference of the analytic gradient picks up the additional
"curvature" term proportional to ``(||x - p|| - r)``, so to compare apples
to apples we place the vertices very close to the boundary
(``||x - p|| ≈ r``) before doing the Hessian FD check.
"""

from __future__ import annotations

import numpy as np
import pytest

from simkit.energies.contact_springs_sphere import (
    contact_springs_sphere_energy,
    contact_springs_sphere_gradient,
    contact_springs_sphere_hessian,
)
from simkit.gradient_cfd import gradient_cfd


FD_STEP = 1e-6
GRAD_TOL = 1e-6
HESS_TOL = 5e-3


def _setup(rng: np.random.Generator, n_points: int, dim: int):
    p = np.zeros(dim)
    r = 1.0
    direction = rng.standard_normal((n_points, dim))
    direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
    radii = rng.uniform(2.0, 3.0, size=(n_points, 1))
    X_outside = radii * direction
    return X_outside, p, r


@pytest.mark.parametrize("dim", [2, 3])
def test_contact_springs_sphere_energy_increases_inside(dim: int) -> None:
    rng = np.random.default_rng(0)
    X_outside, p, r = _setup(rng, n_points=4, dim=dim)
    direction = X_outside / np.linalg.norm(X_outside, axis=1, keepdims=True)
    X_inside = 0.3 * direction

    k = 1.4
    e_out = float(contact_springs_sphere_energy(X_outside, k, p, r))
    e_in = float(contact_springs_sphere_energy(X_inside, k, p, r))

    assert e_out == pytest.approx(0.0, abs=1e-12)
    assert e_in > e_out


@pytest.mark.parametrize("dim", [2, 3])
def test_contact_springs_sphere_gradient_matches_fd(dim: int) -> None:
    rng = np.random.default_rng(1)
    p = np.zeros(dim)
    r = 1.0
    direction = rng.standard_normal((3, dim))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    X = 0.5 * direction
    n_pts = X.shape[0]
    k = 1.9

    def energy_flat(x_flat: np.ndarray) -> np.ndarray:
        return np.array(
            [float(contact_springs_sphere_energy(x_flat.reshape(n_pts, dim), k, p, r))]
        )

    g_fd = gradient_cfd(energy_flat, X.flatten(), FD_STEP).flatten()
    g = np.asarray(contact_springs_sphere_gradient(X, k, p, r)).flatten()

    assert np.allclose(g, g_fd, atol=GRAD_TOL)


@pytest.mark.parametrize("dim", [2, 3])
def test_contact_springs_sphere_hessian_matches_fd_near_boundary(dim: int) -> None:
    rng = np.random.default_rng(2)
    p = np.zeros(dim)
    r = 1.0
    direction = rng.standard_normal((2, dim))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    # Push vertices just inside the sphere so the gap (||x-p|| - r) is tiny.
    # In that regime the omitted curvature term ~ (||x-p|| - r) * d n/d x
    # is small and the shipped (Gauss-Newton) Hessian agrees with FD.
    X = (r - 1e-3) * direction
    n_pts = X.shape[0]
    k = 1.1

    def grad_flat(x_flat: np.ndarray) -> np.ndarray:
        return np.asarray(
            contact_springs_sphere_gradient(x_flat.reshape(n_pts, dim), k, p, r)
        ).flatten()

    H_fd = gradient_cfd(grad_flat, X.flatten(), FD_STEP)
    H = np.asarray(contact_springs_sphere_hessian(X, k, p, r).todense())

    assert np.allclose(H, H_fd, atol=HESS_TOL)
