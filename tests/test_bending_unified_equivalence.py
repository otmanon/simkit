"""Regression + dispatch tests for the unified bending energy.

The 2D hinge energy and 3D dihedral energy were unified behind one flat,
``dim``-inferring energy (`bending_energy`) that dispatches through
:mod:`simkit.dihedral_angles` and gathers with :func:`simkit.wedge_map`. These
tests pin that the unification is numerically a no-op against goldens captured
from the pre-refactor implementation, and that the ``dihedral_angles`` dispatch
and the generic ``wedge_map`` agree with the per-dimension primitives.

The single documented behavior change is 2D ``bending_hessian_x(..., psd=True)``:
it now projects the *combined* per-hinge block (Gauss-Newton + geometric) to PSD
to match the 3D module, instead of the legacy 2D behavior of projecting only the
geometric term. The default ``psd=False`` path is numerically exact, so the
goldens below only check ``psd=False`` for 2D.
"""

from __future__ import annotations

import os

import numpy as np
import scipy as sp
import pytest

from simkit.wedge_map import wedge_map
from simkit.dihedral_wedge_map import dihedral_wedge_map
from simkit.dihedral_angles import (
    dihedral_angles,
    dihedral_angles_gradient_element,
    dihedral_angles_hessian_element,
)
from simkit.dihedral_angles_2d import (
    dihedral_angles_2d,
    dihedral_angles_2d_gradient_element,
    dihedral_angles_2d_hessian_element,
)
from simkit.dihedral_angles_3d import dihedral_angles_3d
from simkit.energies.bending_energy import (
    bending_energy_x,
    bending_gradient_x,
    bending_hessian_x,
)
from simkit.energies.discrete_shells_bending import (
    discrete_shells_bending_energy_x,
    discrete_shells_bending_gradient_x,
    discrete_shells_bending_hessian_x,
)

GOLDEN = os.path.join(os.path.dirname(__file__), "data", "bending_golden.npz")


def _beam(n_nodes=5):
    X = np.zeros((n_nodes, 2))
    X[:, 0] = np.arange(n_nodes)
    H = np.array([[i, i + 1, i + 2] for i in range(n_nodes - 2)], dtype=np.int64)
    return X, H


def _shell():
    X = np.array([[0.0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, -1, 0]])
    D = np.array([[2, 0, 1, 3]], dtype=np.int64)
    return X, D


# --------------------------------------------------------------------------- #
# Golden regression: the unification is a numerical no-op                      #
# --------------------------------------------------------------------------- #
def test_matches_pre_refactor_golden() -> None:
    g = np.load(GOLDEN)

    # 2D: new signature takes kappa = ymI/l (= 1 here).
    _, H = _beam(5)
    theta0 = np.zeros((H.shape[0], 1))
    kappa = np.ones((H.shape[0], 1))
    x2 = g["x2"]
    assert float(bending_energy_x(x2, H, theta0, kappa)) == pytest.approx(float(g["e2"]), abs=1e-12)
    assert np.allclose(np.asarray(bending_gradient_x(x2, H, theta0, kappa)).flatten(), g["g2"], atol=1e-10)
    assert np.allclose(np.asarray(bending_hessian_x(x2, H, theta0, kappa, psd=False).todense()), g["Hh2_false"], atol=1e-10)

    # 3D: back-compat shim preserves the original signature.
    X, D = _shell()
    theta0_3 = dihedral_angles(X, D)
    ym_b, he, le = np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 3))
    X3d = g["X3d"]
    assert float(discrete_shells_bending_energy_x(X3d, D, theta0_3, ym_b, he, le)) == pytest.approx(float(g["e3"]), abs=1e-12)
    assert np.allclose(np.asarray(discrete_shells_bending_gradient_x(X3d, D, theta0_3, ym_b, he, le)).flatten(), g["g3"], atol=1e-10)
    assert np.allclose(np.asarray(discrete_shells_bending_hessian_x(X3d, D, theta0_3, ym_b, he, le, psd=False).todense()), g["Hh3_false"], atol=1e-10)
    assert np.allclose(np.asarray(discrete_shells_bending_hessian_x(X3d, D, theta0_3, ym_b, he, le, psd=True).todense()), g["Hh3_true"], atol=1e-10)


# --------------------------------------------------------------------------- #
# Dispatch + generic wedge_map agree with per-dimension primitives            #
# --------------------------------------------------------------------------- #
def test_dihedral_angles_dispatch_matches_primitives() -> None:
    X2, H = _beam(5)
    rng = np.random.default_rng(1)
    X2 = X2 + 0.05 * rng.standard_normal(X2.shape)
    assert np.allclose(dihedral_angles(X2, H), dihedral_angles_2d(X2, H))
    assert np.allclose(dihedral_angles_gradient_element(X2, H), dihedral_angles_2d_gradient_element(X2, H))
    assert np.allclose(dihedral_angles_hessian_element(X2, H), dihedral_angles_2d_hessian_element(X2, H))

    X3, D = _shell()
    X3 = X3 + 0.05 * rng.standard_normal(X3.shape)
    assert np.allclose(dihedral_angles(X3, D), dihedral_angles_3d(X3, D))


def test_wedge_map_matches_dihedral_and_gathers_correctly() -> None:
    X3, D = _shell()
    assert np.allclose(
        wedge_map(D, X3.shape[0]).todense(),
        dihedral_wedge_map(D, X3.shape[0]).todense(),
    )

    # kron(wedge_map, eye(2)) @ x reproduces the per-hinge [Ax,Ay,Bx,By,Cx,Cy] gather
    X2, H = _beam(5)
    x = X2.flatten()
    W = sp.sparse.kron(wedge_map(H, X2.shape[0]), sp.sparse.identity(2))
    gathered = (W @ x).reshape(H.shape[0], 6)
    expected = np.concatenate([X2[H[:, 0]], X2[H[:, 1]], X2[H[:, 2]]], axis=1)
    assert np.allclose(gathered, expected)


if __name__ == "__main__":
    test_matches_pre_refactor_golden()
    test_dihedral_angles_dispatch_matches_primitives()
    test_wedge_map_matches_dihedral_and_gathers_correctly()
    print("All unified-bending equivalence tests passed.")
