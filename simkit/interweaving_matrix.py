"""Sparse permutation that interleaves per-vertex DOFs by component.

Maps a vector stacked vertex-by-vertex ``(v0_d0, v0_d1, ..., v1_d0, ...)`` to
the component-major layout ``(v0_d0, v1_d0, ..., v0_d1, ...)`` used by some
solvers and matrix assemblies.
"""

import numpy as np
import scipy as sp


def interweaving_matrix(t: int, d: int) -> sp.sparse.csc_matrix:
    """Permutation matrix between vertex-major and component-major stacking.

    Parameters
    ----------
    t : int
        Number of vertices (or entities).
    d : int
        Spatial dimension (components per vertex).

    Returns
    -------
    M : scipy.sparse.csc_matrix (t*d, t*d)
        Identity-on-nonzeros permutation; ``M @ x`` reorders ``x`` from
        vertex-major to component-major layout.
    """
    ii = np.arange(t * d)

    i = ii.reshape(t, d)
    j = ii.reshape(t, d, order='F')
    v = np.ones(ii.shape)
    M = sp.sparse.csc_matrix((v, (i.flatten(), j.flatten())), shape=(t * d, t * d))
    return M
