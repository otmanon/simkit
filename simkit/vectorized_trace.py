"""Sparse trace operator on a stack of flattened ``d x d`` matrices.

Given ``n`` matrices stored column-major as a length ``n * d * d`` vector,
the operator returns the ``n`` traces. Used when differentiating energies
that depend on ``tr(F)`` or similar reductions.
"""

import numpy as np
import scipy as sp


def vectorized_trace(n: int, d: int) -> sp.sparse.csc_matrix:
    """CSC matrix that extracts the trace of each ``d x d`` block.

    Assumes column-major flattening: block ``i`` occupies indices
    ``i, n + i, 2*n + i, ...`` along the diagonal in the stacked layout.

    Parameters
    ----------
    n : int
        Number of matrices.
    d : int
        Matrix side length.

    Returns
    -------
    T : scipy.sparse.csc_matrix (n, n * d * d)
        Trace operator; ``T @ vec`` yields length-``n`` traces.
    """
    # trace of matrix i will have elements i, then n+i + 1, 2n+i+2

    ii = np.arange(n * d * d)
    Mi = np.reshape(ii, (n * d, d))

    Mii = np.reshape(Mi, (n, d, d))
    Mii = np.reshape(Mii, (d * n, d))

    mask = np.identity(d, dtype=bool)
    maskii = np.tile(mask, (n, 1))
    j = Mii[maskii]
    i = np.repeat(np.arange(n), d)

    v = np.ones((i.shape[0]))
    T = sp.sparse.csc_matrix((v, (i, j)), (n, n * d * d))

    return T
