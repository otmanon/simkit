"""Vertex-to-simplex incidence matrix for volumetric meshes.

Each column lists which vertices belong to that simplex; entries are ``1`` when
vertex ``i`` appears in simplex ``j``.
"""

import numpy as np
import scipy as sp


def vertex_to_simplex_adjacency(T: np.ndarray, nv: int) -> sp.sparse.csc_matrix:
    """CSC adjacency matrix from vertices to simplices (columns).

    Parameters
    ----------
    T : np.ndarray (m, k)
        Simplex vertex indices (e.g. tetrahedra with ``k = 4``).
    nv : int
        Number of mesh vertices (row count).

    Returns
    -------
    A : scipy.sparse.csc_matrix (nv, m)
        ``A[i, j] = 1`` if vertex ``i`` is a corner of simplex ``j``.
    """
    # Initialize empty lists to store row, column, and data for the triplet list
    rows, cols, data = [], [], []

    # Iterate through the tetrahedra and populate the triplet list
    for tetrahedron_idx, vertices in enumerate(T):
        for vertex in vertices:
            rows.append(vertex)
            cols.append(tetrahedron_idx)
            data.append(1)

    # Create a CSC sparse matrix from the triplet lists
    adjacency_matrix = sp.sparse.csc_matrix(
        (data, (rows, cols)), shape=(nv, len(T)), dtype=int
    )

    return adjacency_matrix
