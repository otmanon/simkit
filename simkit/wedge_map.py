"""Selection map from per-hinge vertex DOFs to global vertex DOFs.

Dimension-agnostic generalization of :func:`simkit.dihedral_wedge_map`. A
discrete bending element ("hinge") references a fixed number of vertices: three
for a 2D vertex hinge (triple ``(A, B, C)``) and four for a 3D edge hinge
(quadruple ``(x0, x1, x2, x3)``). This builds the gather that stacks those
vertices for every hinge into one vector.
"""

import numpy as np
import scipy as sp


def wedge_map(C: np.ndarray, nv: int) -> "sp.sparse.csc_matrix":
    """Sparse selector mapping stacked hinge vertices to global vertices.

    Each hinge references ``k`` vertices (``k == C.shape[1]``; 3 for a 2D vertex
    hinge, 4 for a 3D edge hinge). Row ``k*e + j`` of the returned matrix has a
    single ``1`` in the column of the global vertex that is the ``j``-th vertex
    of hinge ``e``. Multiplying a global per-vertex array by this map gathers the
    ``k`` hinge vertices for every element into one stacked vector.

    Parameters
    ----------
    C : np.ndarray (E, k)
        Hinge connectivity (vertex triples for 2D, quadruples for 3D).
    nv : int
        Total number of vertices in the mesh.

    Returns
    -------
    M : scipy.sparse.csc_matrix (k*E, nv)
        Gather matrix from global vertices to stacked hinge vertices.
    """
    J = C.flatten()                 # global vertex index for each stacked row
    I = np.arange(J.shape[0])       # one row per stacked hinge vertex
    V = np.ones(I.shape[0])         # a single unit entry per row
    M = sp.sparse.csc_matrix((V, (I, J)), (C.shape[0] * C.shape[1], nv))
    return M
