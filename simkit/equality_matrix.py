import numpy as np
import scipy as sp


def equality_matrix(bI, n):
    """
    Returns a constraint matrixn S that enforces equality over multiple vertices  of a mesh
    
    for example if bI = [1, 2, 3], then the first vertex is constrained to be equal to the second and third vertices
    
    the expression S @ V will return the difference between the first vertex and the second, as well as the difference between the first vertex and the third, and the difference between the second and the third
    
    Should be equivalent to edge_difference_matrix(E, n) where E is [[1, 2], [2, 3]]

    Parameters
    ----------
    bI : (m, 1) array
        Indices of the vertices to be constrained to be equal
    n : int
        Number of vertices in the mesh
    
    Returns
    -------
    S : (m-1, n) sparse matrix
        Equality matrix
        
    """
    I = np.repeat(np.arange(bI.shape[0] - 1)[:, None], 2, axis=1)

    J = np.zeros((bI.shape[0]-1, 2), dtype=int)
    J[:, 0] = bI[1:]
    J[:, 1] = bI[0]

    V = np.ones((bI.shape[0]-1, 2))
    V[:, 1] = -1

    S = sp.sparse.csc_matrix((V.flatten(), (I.flatten(), J.flatten())), shape=(bI.shape[0]-1, n))

    return S


