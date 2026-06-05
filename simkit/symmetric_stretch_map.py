"""Linear maps between full and symmetric independent stretch components.

Builds block-diagonal sparse matrices that embed the ``d(d+1)/2`` independent
entries of a symmetric ``d x d`` stretch tensor into its ``d^2`` stacked
components (and the inverse averaging map for the off-diagonal terms).
"""

from typing import Tuple

import numpy as np
import scipy as sp


def symmetric_stretch_map(t: int, dim: int) -> Tuple[sp.sparse.csc_matrix, sp.sparse.csc_matrix]:
    """Block-diagonal stretch embedding and extraction maps over ``t`` elements.

    For ``dim == 2``, six stacked entries map to three independent symmetric
    components; for ``dim == 3``, nine entries map to six. Off-diagonal
    coupling uses duplicate indexing with weight ``1/2`` in the inverse map.

    Parameters
    ----------
    t : int
        Number of elements (blocks along the diagonal).
    dim : int
        Spatial dimension (typically 2 or 3).

    Returns
    -------
    Se : scipy.sparse.csc_matrix (t * dim^2, t * n_sym)
        Embeds independent symmetric components into stacked stretch entries.
    Sei : scipy.sparse.csc_matrix (t * n_sym, t * dim^2)
        Extracts symmetric components from stacked entries, averaging
        off-diagonals with weight ``1/2``.
    """

    SI = np.arange(dim * dim, dtype=int).reshape(dim, dim)
    SJ = -np.ones((dim, dim), dtype=int)
    SV = np.zeros((dim, dim), dtype=float)
    SVi = np.zeros((dim, dim), dtype=float)
    counter = 0
    for i in range(dim):
        SJ[i, i] = counter
        counter += 1
        SV[i, i] = 1
        SVi[i, i] = 1

    for i in range(dim):
        for j in range(dim):
            if i == j:
                continue
            else:
                if j > i:  # upper triangular
                    SJ[i, j] = counter
                    counter += 1
                elif j < i:  # lower triangular
                    SJ[i, j] = SJ[j, i]
                SV[i, j] = 1
                SVi[i, j] = 1 / 2


    S = sp.sparse.csc_matrix((SV.flatten(), (SI.flatten(), SJ.flatten())), shape=(dim * dim, SJ.max() + 1))
    Si = sp.sparse.csc_matrix((SVi.flatten(), (SJ.flatten(), SI.flatten())), shape=(SJ.max() + 1, dim * dim))

    Se = sp.sparse.kron(sp.sparse.identity(t), S)
    Sei = sp.sparse.kron(sp.sparse.identity(t), Si)
    return Se, Sei
