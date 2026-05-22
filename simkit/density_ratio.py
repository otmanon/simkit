"""Sparsity diagnostic: fraction of nonzero entries in a sparse matrix."""

import scipy as sp


def density_ratio(A: "sp.sparse.spmatrix") -> float:
    """Fraction of a sparse matrix's entries that are explicitly stored.

    A value near 0 means the matrix is very sparse; a value near 1 means it is
    nearly dense and the sparse representation is buying little.

    Parameters
    ----------
    A : scipy.sparse.spmatrix (m, n)
        Sparse matrix to inspect.

    Returns
    -------
    ratio : float
        ``nnz / (m * n)``, in ``[0, 1]``.
    """
    nnz = A.nnz                          # number of explicitly stored entries
    nelem = A.shape[0] * A.shape[1]      # total entries if it were dense
    return nnz / nelem