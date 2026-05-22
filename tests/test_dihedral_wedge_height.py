"""Tests for ``simkit.dihedral_wedge_height``."""

from __future__ import annotations

import numpy as np

from simkit.dihedral_wedge_height import (
    dihedral_wedge_height,
    dihedral_wedge_heights,
    dihedral_wedge_heights_element,
)
from simkit.dihedral_wedges import dihedral_wedges


def _right_angle_hinge():
    X = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    F = np.array([[0, 1, 2], [1, 2, 3]])
    return X, dihedral_wedges(F)


def test_dihedral_wedge_heights_element_shared_edge() -> None:
    x0 = np.array([[0.0, 1.0, 0.0]])
    x1 = np.array([[0.0, 0.0, 0.0]])
    x2 = np.array([[1.0, 0.0, 0.0]])
    x3 = np.array([[0.0, 0.0, 1.0]])
    h0, h1, h2, h0_tilde, h1_tilde, h2_tilde = dihedral_wedge_heights_element(
        x0, x1, x2, x3
    )
    assert h0.shape == (1, 1)
    assert np.isclose(h0[0, 0], 1.0)
    assert np.isclose(h0_tilde[0, 0], 1.0)


def test_dihedral_wedge_height_is_mean_of_shared_edge_heights() -> None:
    X, D = _right_angle_hinge()
    h0, _, _, h0_tilde, _, _ = dihedral_wedge_heights(X, D)
    h = dihedral_wedge_height(X, D)
    assert np.allclose(h, 0.5 * (h0 + h0_tilde))
