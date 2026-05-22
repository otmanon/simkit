"""Sparse selection matrix for row indexing."""

import numpy as np
import scipy as sp


def selection_matrix(cI: np.ndarray, n: int) -> sp.sparse.csc_matrix:
    """Build a sparse matrix that picks rows indexed by ``cI``.

    Each row ``i`` of the result has a single ``1`` in column ``cI[i]``,
    so ``G @ x`` extracts ``x[cI]`` when ``x`` is a column vector.

    Parameters
    ----------
    cI : np.ndarray (m,)
        Column indices to select (one per output row).
    n : int
        Number of columns (length of the input vector).

    Returns
    -------
    G : scipy.sparse.csc_matrix (m, n)
        Selection matrix with one nonzero per row.
    """

    I = np.arange(cI.shape[0])
    J = cI

    v = np.ones(cI.shape[0])

    G = sp.sparse.csc_matrix(
        (v, (I, J)),
        shape=(cI.shape[0], n),
    )
    return G
