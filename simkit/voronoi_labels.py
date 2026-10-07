"""Voronoi partition of a mesh by nearest control node (Faure et al. 2011)."""

from __future__ import annotations

import numpy as np


def voronoi_labels(D_nodes: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Voronoi region label for every vertex and every triangle.

    Given the distance from each control node to each vertex (in any metric,
    e.g. the compliance distances from :func:`simkit.compliance_distances`),
    assigns each vertex and each triangle to its nearest node.

    Parameters
    ----------
    D_nodes : np.ndarray (k, n)
        Distances from each of the ``k`` nodes to every vertex.
    T : np.ndarray (t, 3)
        Triangle connectivity.

    Returns
    -------
    vertex_labels : np.ndarray (n,)
        Index of the nearest node for each vertex.
    face_labels : np.ndarray (t,)
        Voronoi region of each triangle (cell), the node minimising the summed
        distance to the triangle's three corners.
    """
    D_nodes = np.asarray(D_nodes, dtype=float)
    T = np.asarray(T)
    vertex_labels = np.argmin(D_nodes, axis=0)
    tri_cost = D_nodes[:, T].sum(axis=2)  # (k, t)
    face_labels = np.argmin(tri_cost, axis=0)
    return vertex_labels, face_labels
