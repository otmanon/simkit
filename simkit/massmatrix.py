import igl
import numpy as np
import scipy as sp

from .volume import volume
from .vertex_to_simplex_adjacency import vertex_to_simplex_adjacency
def massmatrix(X, T, rho : None | float | np.ndarray = 1):
    """
    Lumped diagonal mass matrix for a mesh

    Parameters
    ----------
    X : (n, 3) numpy array
        The vertices of the mesh
    T : (t, s) simplex indeces array
        The triangles of the mesh
    rho : float or (t, 1) numpy array of per-simplex densities
        The density of the material
    """

    v = volume(X, T)


    m = v * rho


    # deposit the mass into the global mass matrix
    Av_t = vertex_to_simplex_adjacency(T, X.shape[0])

    vv = ( Av_t @ m )/ T.shape[1]


    M = sp.sparse.diags(vv.flatten()) 
    return M