"""Wedge (altitude) heights of the two triangles meeting at a hinge edge."""

from typing import Tuple

import numpy as np


def dihedral_wedge_heights(
    X: np.ndarray, D: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """All six wedge heights per hinge, gathered from global vertex positions.

    Parameters
    ----------
    X : np.ndarray (n, 3)
        Vertex positions.
    D : np.ndarray (nd, 4)
        Hinge vertex indices ``(x0, x1, x2, x3)``.

    Returns
    -------
    h0, h1, h2 : np.ndarray (nd, 1)
        Altitudes of triangle 1 from edges ``e0, e1, e2``.
    h0_tilde, h1_tilde, h2_tilde : np.ndarray (nd, 1)
        Altitudes of triangle 2 from its corresponding edges.
    """
    x0 = X[D[:, 0]]
    x1 = X[D[:, 1]]
    x2 = X[D[:, 2]]
    x3 = X[D[:, 3]]
    return dihedral_wedge_heights_element(x0, x1, x2, x3)


def dihedral_wedge_height(X: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Mean of the two shared-edge altitudes per hinge.

    Parameters
    ----------
    X : np.ndarray (n, 3)
        Vertex positions.
    D : np.ndarray (nd, 4)
        Hinge vertex indices ``(x0, x1, x2, x3)``.

    Returns
    -------
    h : np.ndarray (nd, 1)
        ``(h0 + h0_tilde) / 2``, the average altitude onto the shared edge.
    """
    h0, h1, h2, h0_tilde, h1_tilde, h2_tilde = dihedral_wedge_heights(X, D)
    h = (h0 + h0_tilde) / 2
    return h


def dihedral_wedge_heights_element(
    x0: np.ndarray, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-hinge wedge heights from explicit corner positions.

    Each height is ``2 * area / edge_length``, the triangle altitude onto the
    corresponding edge.

    Parameters
    ----------
    x0 : np.ndarray (nd, 3)
        Apex vertex of triangle 1.
    x1, x2 : np.ndarray (nd, 3)
        Shared edge vertices.
    x3 : np.ndarray (nd, 3)
        Apex vertex of triangle 2.

    Returns
    -------
    h0, h1, h2 : np.ndarray (nd, 1)
        Altitudes of triangle 1 from edges ``e0, e1, e2``.
    h0_tilde, h1_tilde, h2_tilde : np.ndarray (nd, 1)
        Altitudes of triangle 2 from its corresponding edges.
    """
    # Triangle 1 edges (e0 is the shared edge); triangle 2 reuses e0.
    e0 = x2 - x1
    e1 = x0 - x2
    e2 = x0 - x1
    e1_tilde = x3 - x2
    e2_tilde = x3 - x1

    # Areas via the cross-product magnitudes.
    n = np.cross(e0, e2)
    n_tilde = np.cross(e2_tilde, e0)
    area = np.linalg.norm(n, axis=1).reshape(-1, 1) / 2
    area_tilde = np.linalg.norm(n_tilde, axis=1).reshape(-1, 1) / 2

    # Altitude onto each edge = 2 * area / |edge|.
    h0 = 2.0 * area / np.linalg.norm(e0, axis=1).reshape(-1, 1)
    h1 = 2.0 * area / np.linalg.norm(e1, axis=1).reshape(-1, 1)
    h2 = 2.0 * area / np.linalg.norm(e2, axis=1).reshape(-1, 1)

    h0_tilde = 2.0 * area_tilde / np.linalg.norm(e0, axis=1).reshape(-1, 1)
    h1_tilde = 2.0 * area_tilde / np.linalg.norm(e1_tilde, axis=1).reshape(-1, 1)
    h2_tilde = 2.0 * area_tilde / np.linalg.norm(e2_tilde, axis=1).reshape(-1, 1)
    return h0, h1, h2, h0_tilde, h1_tilde, h2_tilde