"""Tests for ``simkit.dirichlet_penalty``."""

from __future__ import annotations

import numpy as np

from simkit.dirichlet_penalty import dirichlet_penalty


def test_quadratic_shapes_and_pinned_gradient_vanishes() -> None:
    nv = 4
    dim = 2
    bI = np.array([0, 2])
    y = np.array([[1.0, 0.0], [0.0, 1.0]])
    gamma = 10.0

    Q, b = dirichlet_penalty(bI, y, nv, gamma)
    assert Q.shape == (nv * dim, nv * dim)
    assert b.shape == (nv * dim, 1)

    x = np.zeros(nv * dim)
    for k, vi in enumerate(bI):
        x[vi * dim : (vi + 1) * dim] = y[k]

    grad = Q @ x.reshape(-1, 1) + b
    assert np.allclose(grad, 0.0, atol=1e-12)

    xc = x.reshape(-1, 1)
    energy = 0.5 * (xc.T @ Q @ xc).item() + (xc.T @ b).item()
    assert energy <= 1e-20
