"""Tests for ``simkit.energies.discrete_shells_bending``.

Discrete-shells bending energy on triangle-pair hinges. We build a flat
two-triangle configuration, verify energy is zero at rest, increases when
the hinge is bent, and that the analytic gradient/Hessian match a central
finite difference of the energy in the flattened-position layout.
"""

from __future__ import annotations

import numpy as np
import pytest

from simkit.dihedral_angles import dihedral_angles
from simkit.energies.discrete_shells_bending import (
    discrete_shells_bending_energy_dx,
    discrete_shells_bending_gradient_dx,
    discrete_shells_bending_hessian_d2x,
)
from simkit.gradient_cfd import gradient_cfd


FD_STEP = 1e-6
TOL = 1e-4


def _two_triangle_hinge():
    """Two triangles sharing an edge in a flat configuration.

    Vertices laid out as::

            x2
           /  \\
         x0----x1
           \\  /
            x3

    ``D[i] = (v_on_face1, shared_v0, shared_v1, v_on_face2)`` with the shared
    edge ``(x0, x1)`` and opposite vertices ``x2`` / ``x3``.
    """
    X = np.array(
        [
            [0.0, 0.0, 0.0],   # x0
            [1.0, 0.0, 0.0],   # x1
            [0.5, 1.0, 0.0],   # x2
            [0.5, -1.0, 0.0],  # x3
        ],
        dtype=np.float64,
    )
    D = np.array([[2, 0, 1, 3]], dtype=np.int64)
    ym_bending = np.ones((1, 1))
    he = np.ones((1, 1))
    le = np.ones((1, 3))
    theta0 = dihedral_angles(X, D)
    return X, D, theta0, ym_bending, he, le


def test_dsb_energy_zero_at_rest_and_increases_on_bend() -> None:
    X, D, theta0, ym_bending, he, le = _two_triangle_hinge()
    e_rest = float(discrete_shells_bending_energy_dx(X, D, theta0, ym_bending, he, le))
    assert e_rest == pytest.approx(0.0, abs=1e-12)

    X_bent = X.copy()
    X_bent[2, 2] += 0.5  # lift x2 out of the plane to introduce a dihedral
    e_bent = float(
        discrete_shells_bending_energy_dx(X_bent, D, theta0, ym_bending, he, le)
    )
    assert e_bent > e_rest


def test_dsb_gradient_matches_fd() -> None:
    X, D, theta0, ym_bending, he, le = _two_triangle_hinge()
    rng = np.random.default_rng(0)
    X_def = X + 0.05 * rng.standard_normal(X.shape)

    def energy_flat(x_flat: np.ndarray) -> np.ndarray:
        return np.array(
            [float(discrete_shells_bending_energy_dx(
                x_flat.reshape(-1, 3), D, theta0, ym_bending, he, le
            ))]
        )

    g = np.asarray(
        discrete_shells_bending_gradient_dx(X_def, D, theta0, ym_bending, he, le)
    ).flatten()
    g_fd = gradient_cfd(energy_flat, X_def.flatten(), FD_STEP).flatten()
    assert np.allclose(g, g_fd, atol=TOL)


def test_dsb_hessian_matches_fd_at_rest() -> None:
    # At ``dtheta = 0`` the curvature term in the Hessian vanishes, so the rest
    # configuration is the cleanest place to compare to a finite difference of
    # the gradient.
    X, D, theta0, ym_bending, he, le = _two_triangle_hinge()

    def grad_flat(x_flat: np.ndarray) -> np.ndarray:
        return np.asarray(
            discrete_shells_bending_gradient_dx(
                x_flat.reshape(-1, 3), D, theta0, ym_bending, he, le
            )
        ).flatten()

    H_ana = np.asarray(
        discrete_shells_bending_hessian_d2x(X, D, theta0, ym_bending, he, le).todense()
    )
    H_fd = gradient_cfd(grad_flat, X.flatten(), FD_STEP)
    assert np.allclose(H_ana, H_fd, atol=TOL)


if __name__ == "__main__":
    test_dsb_energy_zero_at_rest_and_increases_on_bend()
    test_dsb_gradient_matches_fd()
    test_dsb_hessian_matches_fd_at_rest()
    print("All tests passed.")