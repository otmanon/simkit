"""Tests for ``simkit.limit_actuation_dirichlet_energy``."""

from __future__ import annotations

import numpy as np

from simkit.deformation_jacobian import deformation_jacobian
from simkit.limit_actuation_dirichlet_energy import limit_actuation_dirichlet_energy


def _unit_triangle_mesh() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    T = np.array([[0, 1, 2]], dtype=int)
    return X, T


def test_limit_actuation_dirichlet_energy_scales_are_nonnegative() -> None:
    X, T = _unit_triangle_mesh()
    dim = X.shape[1]
    rng = np.random.default_rng(0)
    D = rng.standard_normal((X.shape[0] * dim, 2))
    max_s = 1.0

    a = limit_actuation_dirichlet_energy(X, T, D, max_s)
    assert a.shape == (D.shape[1],)
    assert np.all(a >= 0.0)


def test_limit_actuation_dirichlet_energy_bounds_density() -> None:
    X, T = _unit_triangle_mesh()
    dim = X.shape[1]
    rng = np.random.default_rng(1)
    D = rng.standard_normal((X.shape[0] * dim, 1))
    max_s = 0.5

    a = limit_actuation_dirichlet_energy(X, T, D, max_s)
    J = deformation_jacobian(X, T)
    JD = (J @ (a[0] * D)).reshape(T.shape[0], dim, dim)
    density = np.sqrt(np.sum(JD**2, axis=(1, 2)))
    assert np.all(density <= max_s + 1e-10)
