"""Tests for ``simkit.energies.sdf_contact`` and ``simkit.energies.tangential_friction``:
finite-difference gradients, and the friction's exact quadratic Hessian."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps

from simkit.energies.sdf_contact import sdf_contact_energy_x, sdf_contact_gradient_x, sdf_contact_hessian_x
from simkit.energies.tangential_friction import (
    tangential_friction_energy_x,
    tangential_friction_gradient_x,
    tangential_friction_hessian_x,
)
from simkit.gradient_cfd import gradient_cfd
from simkit.sphere_sdf import sphere_sdf


def _setup():
    rng = np.random.default_rng(1)
    x = 0.9 * rng.standard_normal(3 * 8)                      # some points inside the unit sphere
    S = sps.random(3 * 6, 3 * 8, density=0.3, random_state=2, format="csr")
    return rng, x, S


def test_sdf_contact_gradient_matches_finite_differences() -> None:
    rng, x, S = _setup()
    sdf = lambda P: sphere_sdf(P, np.zeros(3), 1.0)
    a = rng.random(6) + 0.5
    E = lambda y: sdf_contact_energy_x(y, S, sdf, 3.0, a)
    assert E(x) > 0
    g = sdf_contact_gradient_x(x, S, sdf, 3.0, a).ravel()
    assert np.allclose(g, gradient_cfd(lambda y: np.array([E(y)]), x.reshape(-1, 1), 1e-6).ravel(), atol=1e-6)
    H = sdf_contact_hessian_x(x, S, sdf, 3.0, a).toarray()
    assert np.allclose(H, H.T) and np.linalg.eigvalsh(H).min() > -1e-9


def test_tangential_friction_is_an_exact_quadratic() -> None:
    rng, x, S = _setup()
    n = rng.standard_normal((6, 3))
    n /= np.linalg.norm(n, axis=1)[:, None]
    p0, w = rng.standard_normal((6, 3)), rng.random(6)
    E = lambda y: tangential_friction_energy_x(y, S, p0, n, w)
    g = tangential_friction_gradient_x(x, S, p0, n, w).ravel()
    assert np.allclose(g, gradient_cfd(lambda y: np.array([E(y)]), x.reshape(-1, 1), 1e-6).ravel(), atol=1e-6)
    H = tangential_friction_hessian_x(x, S, p0, n, w)
    v = rng.standard_normal(x.size)
    assert np.isclose(E(x + v), E(x) + g @ v + 0.5 * v @ (H @ v))
    # motion along the normal of every point is free
    P = (S @ x).reshape(-1, 3)
    assert np.isclose(tangential_friction_energy_x(x, S, P + 0.3 * n, n, w), 0.0)
