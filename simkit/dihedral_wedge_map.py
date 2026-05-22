"""Selection map from hinge-vertex DOFs to global vertex DOFs."""

import numpy as np
import scipy as sp


def dihedral_wedge_map(D: np.ndarray, nv: int) -> "sp.sparse.csc_matrix":
    """Sparse selector mapping stacked hinge vertices to global vertices.

    Each interior edge ("hinge") references four vertices ``(x0, x1, x2, x3)``.
    Row ``4*e + k`` of the returned matrix has a single ``1`` in the column of
    the global vertex that is the k-th vertex of hinge ``e``. Multiplying a
    global per-vertex array by this map gathers the four hinge vertices for
    every edge into one stacked vector.

    Parameters
    ----------
    D : np.ndarray (nd, 4)
        Hinge vertex indices. Columns are ``(x0, x1, x2, x3)``, where
        ``(x1, x2)`` are the shared edge and ``x0``, ``x3`` are the opposite
        apex vertices of the two incident triangles.
    nv : int
        Total number of vertices in the mesh.

    Returns
    -------
    M : scipy.sparse.csc_matrix (4*nd, nv)
        Gather matrix from global vertices to stacked hinge vertices.
    """
    J = D.flatten()                 # global vertex index for each stacked row
    I = np.arange(J.shape[0])       # one row per stacked hinge vertex
    V = np.ones(I.shape[0])         # a single unit entry per row
    M = sp.sparse.csc_matrix((V, (I, J)), (4 * D.shape[0], nv))
    return M