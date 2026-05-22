"""Rewrite displacement constraints as constraints on LBS skinning weights.

Transforms a linear equality ``C @ u = 0`` on per-vertex displacements into an
equivalent (redundant-column-stripped) constraint ``A @ W = 0`` on per-vertex
skinning weights, using the vertex positions and mass matrix for rank reduction.
"""

import numpy as np
import scipy as sp

from simkit.remove_redundant_columns import remove_redundant_columns

from .massmatrix import massmatrix
from .orthonormalize import orthonormalize


def lbs_weight_space_constraint(
    V: np.ndarray, T: np.ndarray, C: np.ndarray
) -> np.ndarray:
    """Linear equality on skinning weights equivalent to ``C @ u = 0``.

    Parameters
    ----------
    V : np.ndarray (n, d)
        Mesh vertex positions.
    T : np.ndarray (t, d+1)
        Simplex indices (for the mass matrix).
    C : np.ndarray (c, d*n)
        Linear equality constraint on stacked per-vertex displacements.

    Returns
    -------
    A : np.ndarray (c', n)
        Constraint matrix acting on per-vertex skinning weights (redundant
        columns removed via :func:`remove_redundant_columns`).
    """
    C = C.T
    n = V.shape[0]
    d = V.shape[1]

    v = np.ones((n, 1))

    A = np.zeros((0, n))
    for i in range(0, d):
        Id = np.arange(0, n) * d + i
        Jd = np.arange(0, n)
        Pd = sp.sparse.coo_matrix((v.flatten(), (Id, Jd)), shape=(d * n, n))
        for j in range(0, d):
            Vj = V[:, j]
            Adj = C.T @ Pd @ sp.sparse.diags(Vj)
            A = np.vstack([A, Adj])
        Ad1 = C.T @ Pd
        A = np.vstack([A, Ad1])

    W = A

    M = massmatrix(V, T)

    W2 = remove_redundant_columns(W.T, M=M).T

    return W2
