"""Tests for ``simkit.sphere_sdf`` and ``simkit.cup_sdf``."""

from __future__ import annotations

import numpy as np

from simkit.cup_sdf import cup_sdf
from simkit.sphere_sdf import sphere_sdf


def test_sphere_sdf_values_and_normals() -> None:
    P = np.array([[2.0, 0, 0], [0, 0.5, 0], [0, 0, -1.0]])
    phi, g = sphere_sdf(P, np.zeros(3), 1.0)
    assert np.allclose(phi, [1.0, -0.5, 0.0])
    assert np.allclose(g, [[1, 0, 0], [0, 1, 0], [0, 0, -1]])


def test_cup_sdf_wall_cavity_and_rotation() -> None:
    c, r, wall, base, L = np.zeros(3), 1.0, 0.1, 0.2, 2.0
    P = np.array([[0.0, 0.0, 1.5],      # outside, 0.5 from the outer wall
                  [0.0, 0.0, 0.95],     # inside the wall
                  [0.0, 0.0, 0.0],      # in the open cavity: outside the material
                  [-0.95, 0.0, 0.0]])   # inside the base
    phi, g = cup_sdf(P, c, r, wall, base, L)
    assert np.isclose(phi[0], 0.5) and phi[1] < 0 and phi[2] > 0 and phi[3] < 0
    assert np.allclose(g[0], [0, 0, 1], atol=1e-5)
    Rz = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])    # cup axis turned onto y
    phi_r, _ = cup_sdf(P @ Rz.T, c, r, wall, base, L, R=Rz)
    assert np.allclose(phi_r, phi)
