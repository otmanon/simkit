"""Harmonic (barycentric-style) coordinates with Dirichlet boundary data."""

import numpy as np
import scipy as sp

from .dirichlet_laplacian import dirichlet_laplacian


def harmonic_coordinates(
    X: np.ndarray, T: np.ndarray, bI: np.ndarray
) -> np.ndarray:
    """Harmonic coordinates with identity data on boundary vertices ``bI``.

    Solves ``L x = 0`` on interior vertices with ``x[bI] = I``, where ``L`` is
    the mesh Dirichlet Laplacian. Each column is one coordinate function that
    is 1 on one boundary vertex and 0 on the others.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Vertex positions.
    T : np.ndarray (t, s)
        Simplex indices.
    bI : np.ndarray
        Boundary vertex indices (one identity row per index).

    Returns
    -------
    x : np.ndarray (n, |bI|)
        Harmonic coordinate values; ``x[bI]`` is the identity matrix.
    """
    L = dirichlet_laplacian(X, T)

    bI = np.unique(bI)
    aI = np.setdiff1d(np.arange(X.shape[0]), bI)

    Q = L

    bc = np.identity(bI.shape[0])

    Qii = Q[aI, :][:, aI]

    Qbi = Q[aI, :][:, bI]

    xii = sp.sparse.linalg.spsolve(Qii, -Qbi @ bc)

    if xii.ndim == 1:
        xii = xii.reshape(-1, 1)
    x = np.zeros((X.shape[0], bI.shape[0]))

    x[aI, :] = xii
    x[bI, :] = bc

    return x
