"""Lumped diagonal mass matrix for simplicial meshes.

Each vertex receives an equal share of the mass of every incident simplex,
weighted by simplex volume and (optionally) per-element density.
"""

from typing import Optional, Union

import numpy as np
import scipy as sp

from .volume import volume
from .vertex_to_simplex_adjacency import vertex_to_simplex_adjacency


def massmatrix(
    X: np.ndarray,
    T: np.ndarray,
    rho: Optional[Union[float, np.ndarray]] = 1,
) -> sp.sparse.dia_matrix:
    """Build a lumped diagonal mass matrix for a simplicial mesh.

    Simplex volumes are scaled by ``rho`` and distributed equally to each
    corner vertex via the vertex-to-simplex adjacency.

    Parameters
    ----------
    X : np.ndarray (n, 3)
        Vertex positions.
    T : np.ndarray (t, s)
        Simplex indices (``s`` is the number of vertices per simplex).
    rho : float or np.ndarray (t, 1), optional
        Material density, scalar or per-simplex. Default 1.

    Returns
    -------
    M : scipy.sparse.dia_matrix (n, n)
        Lumped diagonal mass matrix.
    """
    v = volume(X, T)

    m = v * rho

    # deposit the mass into the global mass matrix
    Av_t = vertex_to_simplex_adjacency(T, X.shape[0])

    vv = (Av_t @ m) / T.shape[1]

    M = sp.sparse.diags(vv.flatten())
    return M
