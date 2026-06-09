"""Boundary edges of a triangle mesh.

Standalone replacement for the triangle case of ``igl.boundary_facets`` so that
boundary extraction does not pull in libigl. Returns the oriented edges that
belong to exactly one triangle.
"""

import numpy as np

# Per-triangle directed edges (opposite each vertex), in the winding of F.
_TRI_EDGES = np.array([[0, 1], [1, 2], [2, 0]])


def boundary_edges(F: np.ndarray) -> np.ndarray:
    """Oriented boundary edges of a triangle mesh.

    An edge lies on the boundary when it is incident to exactly one triangle.
    Boundary edges keep their original orientation from ``F`` (counter-clockwise
    for a consistently wound mesh), so they can be drawn as an outline directly.

    Parameters
    ----------
    F : np.ndarray (f, 3)
        Triangle connectivity.

    Returns
    -------
    E : np.ndarray (b, 2)
        Boundary edges as oriented vertex-index pairs.
    """
    F = np.asarray(F)
    # Every directed edge, in triangle winding order.
    E = F[:, _TRI_EDGES].reshape(-1, 2)
    # Count each undirected edge; the boundary ones appear exactly once.
    key = np.sort(E, axis=1)
    _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)  # numpy 2.x can return a column; flatten for indexing
    return E[counts[inv] == 1]
