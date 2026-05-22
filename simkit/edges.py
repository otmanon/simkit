"""Unique undirected edges of a triangle or tetrahedron mesh.

Standalone replacement for ``igl.edges`` so that edge extraction does not pull
in libigl. Given a simplex list (triangles or tets), returns each undirected
edge exactly once, with the smaller vertex index first and rows sorted
lexicographically. This matches libigl's convention so it can be swapped in
without changing downstream edge ordering.
"""

import numpy as np

# Local edges (vertex-pair index positions) within a single simplex, by the
# number of vertices per simplex. Triangles have 3 edges; tets have 6.
_SIMPLEX_EDGE_LOCAL = {
    3: np.array([[0, 1], [1, 2], [2, 0]]),
    4: np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]),
}


def edges(T: np.ndarray) -> np.ndarray:
    """Unique undirected edges of a triangle or tet mesh.

    Parameters
    ----------
    T : np.ndarray (t, 3) or (t, 4)
        Simplex connectivity. 3 columns for triangles, 4 for tets.

    Returns
    -------
    E : np.ndarray (m, 2)
        Unique edges, each as a sorted vertex-index pair (smaller index first),
        with rows sorted lexicographically.

    Raises
    ------
    ValueError
        If ``T`` does not have 3 or 4 columns.
    """
    T = np.asarray(T)
    s = T.shape[1]
    if s not in _SIMPLEX_EDGE_LOCAL:
        raise ValueError(f"edges expects 3 (tri) or 4 (tet) columns, got {s}")

    # Gather every edge from every simplex as global vertex-index pairs.
    local = _SIMPLEX_EDGE_LOCAL[s]
    all_edges = T[:, local].reshape(-1, 2)

    # Make each edge undirected by sorting its two endpoints, then drop
    # duplicates. np.unique also returns the rows in lexicographic order.
    all_edges = np.sort(all_edges, axis=1)
    E = np.unique(all_edges, axis=0)
    return E