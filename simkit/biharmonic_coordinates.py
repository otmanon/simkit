import scipy as sp
import numpy as np

from .dirichlet_laplacian import dirichlet_laplacian
from .massmatrix import massmatrix


def biharmonic_coordinates(X, T, bI):

    """
    Compute the biharmonic coordinates of the mesh with handles at indices bI.
    
    
    Parameters
    ----------
    X (n, d) array of vertex positions
    T (t, s) array of simplex indices
    bI (b,) array of boundary vertex indices
    
    Returns
    -------
    W (n, b) array of biharmonic coordinates
    
    Example
    -------
    ```python
    X = np.random.rand(100, 3)
    """
    
    L = dirichlet_laplacian(X, T)
    M = massmatrix(X, T)
    Mi = sp.sparse.diags(1/M.diagonal())
    
    bI = np.unique(bI)
    aI = np.setdiff1d(np.arange(X.shape[0]), bI)

    Q = L.T @ Mi @ L

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