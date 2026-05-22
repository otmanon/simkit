"""Sparse gather map from mesh vertices to stacked triangle corners.

For face list ``F`` with rows ``[v0, v1, v2]``, the map repeats each vertex
index so that ``(M @ x).reshape(-1, 3)`` lists corners in face order.
"""

import numpy as np
import scipy as sp


def triangle_map(F: np.ndarray, nv: int) -> sp.sparse.csc_matrix:
    """CSC matrix mapping vertex DOFs to flattened triangle corner positions.

    Parameters
    ----------
    F : np.ndarray (m, 3)
        Triangle vertex indices.
    nv : int
        Number of vertices (column dimension of the map).

    Returns
    -------
    M : scipy.sparse.csc_matrix (3*m, nv)
        Gather operator with one nonzero per corner entry. For vertex positions
        ``vertices`` of shape ``(nv, 3)``, ``(M @ vertices).reshape(-1, 3)``
        stacks the three corners of each triangle.
    """
    J = F.flatten()
    I = np.arange(J.shape[0])
    V = np.ones(I.shape[0])
    M = sp.sparse.csc_matrix((V, (I, J)), (3 * F.shape[0], nv))
    return M
