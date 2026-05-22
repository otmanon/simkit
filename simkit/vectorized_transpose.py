"""Sparse transpose operator on a stack of flattened ``d x d`` matrices.

Given ``n`` matrices stored column-major as a length ``n * d * d`` vector,
the operator permutes entries so each block is transposed in place.
"""

import numpy as np
import scipy as sp


def vectorized_transpose(n: int, d: int) -> sp.sparse.csc_matrix:
    """CSC permutation matrix that transposes each ``d x d`` block.

    Assumes column-major flattening of the input stack.

    Parameters
    ----------
    n : int
        Number of matrices.
    d : int
        Matrix side length.

    Returns
    -------
    T : scipy.sparse.csc_matrix (n * d * d, n * d * d)
        Transpose operator; ``T @ vec`` is the stacked transposed matrices in
        the same flattening convention.
    """
    ii = np.arange(n * d * d)
    Mi = np.reshape(ii, (n * d, d))

    Mii = np.reshape(Mi, (n, d, d))
    Mii = Mii.transpose((0, 2, 1))
    Mii = np.reshape(Mii, (d * n, d))
    Mj = Mii.flatten()
    # transpose each block

    i = Mj
    j = np.arange(n * d * d)
    v = np.ones(n * d * d)
    T = sp.sparse.csc_matrix((v, (i, j)), (n * d * d, n * d * d))
    return T
