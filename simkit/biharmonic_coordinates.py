"""Biharmonic interpolation weights for handle-based deformation."""

import numpy as np
import scipy as sp

from .dirichlet_laplacian import dirichlet_laplacian
from .massmatrix import massmatrix


def biharmonic_coordinates(X: np.ndarray, T: np.ndarray, bI: np.ndarray) -> np.ndarray:
    """Biharmonic coordinates that interpolate values from handle vertices.

    Minimizes the discrete biharmonic energy ``x^T (L^T M^{-1} L) x`` subject to
    each handle vertex being held at its indicator value, giving one smooth
    weight field per handle. The fields sum to one (partition of unity) and
    reproduce the handle values exactly.

    Parameters
    ----------
    X : np.ndarray (n, d)
        Vertex positions.
    T : np.ndarray (t, s)
        Simplex connectivity.
    bI : np.ndarray (b,)
        Handle (boundary) vertex indices. Duplicates are ignored.

    Returns
    -------
    W : np.ndarray (n, b_unique)
        Biharmonic weights; column k is the field for the k-th unique handle.
    """
    L = dirichlet_laplacian(X, T)
    M = massmatrix(X, T)
    Mi = sp.sparse.diags(1 / M.diagonal())   # lumped inverse mass

    # Partition vertices into constrained handles (bI) and free interior (aI).
    bI = np.unique(bI)
    aI = np.setdiff1d(np.arange(X.shape[0]), bI)

    # Biharmonic operator Q = L^T M^{-1} L.
    Q = L.T @ Mi @ L

    # Handle constraint values: each handle gets its own indicator column.
    bc = np.identity(bI.shape[0])

    # Solve the constrained minimization for the free vertices:
    #   Qii x_free = -Qbi bc.
    Qii = Q[aI, :][:, aI]
    Qbi = Q[aI, :][:, bI]
    xii = sp.sparse.linalg.spsolve(Qii, -Qbi @ bc)
    if xii.ndim == 1:
        xii = xii.reshape(-1, 1)

    # Reassemble the full weight matrix: solved values on free vertices,
    # the indicator values on the handles.
    W = np.zeros((X.shape[0], bI.shape[0]))
    W[aI, :] = xii
    W[bI, :] = bc
    return W