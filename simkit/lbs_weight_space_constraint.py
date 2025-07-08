import scipy as sp
import numpy as np

from simkit.remove_redundant_columns import remove_redundant_columns
from .orthonormalize import orthonormalize
from .massmatrix import massmatrix
def lbs_weight_space_constraint(V, T, C):
    """ Rewrites a linear equality constraint that acts on per-vertex displacements (CU(W) = 0)
        to instead act on the per-vertex skinning weights  (AW = 0).

    Parameters
    ----------
    V : (n, d) float numpy array
        Mesh vertices

    C : (c, dn) float numpy array
        Linear equality constraint matrix that acts on per-vertex displacements

    Returns
    -------
    A : (n, c') float numpy array
        Linear equality constraint matrix that acts on per-vertex skinning weights
    """
    C = C.T
    n = V.shape[0]
    d = V.shape[1]

    v = np.ones((n, 1))

    A = np.zeros((0, n))
    for i in range(0, d):
        Id = np.arange(0, n) *d + i
        Jd = np.arange(0, n)
        Pd = sp.sparse.coo_matrix((v.flatten(), (Id, Jd)), shape=(d*n,n))
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