import numpy as np
import igl
import scipy as sp


from .volume import volume


def cluster_grouping_matrices(l, V, T, return_mass=False):
    
    """
    Compute the grouping matrices for the cluster labels l, and the mesh V, T.
    
    Parameters
    ----------
    l (t,) array of cluster labels
    V (n, d) array of vertex positions
    T (t, s) array of simplex indices
    return_mass (bool, optional): whether to return the mass of each cluster
    
    Returns
    -------
    G (c, t) grouping matrix
    Gm (c, t) grouping matrix with mass normalization
    mc (c,) array of cluster masses (if return_mass is True)
    mt (t,) array of simplex masses (if return_mass is True)
    f (t,) array of cluster mass fractions (if return_mass is True)
    """
    t = T.shape[0]
    c = l.max() + 1
    assert(T.shape[1] == 4 or T.shape[1] == 3)
    I= l
    J = np.arange(t)
    mt = volume(V, T)
    if mt.ndim==0:
        mt = mt[None]
    mc = np.bincount(l, mt[:, 0]) #mass of each cluster
    Mci = sp.sparse.diags(1/mc, 0)
    Mt = sp.sparse.diags(mt[:, 0],  0)

    VV = np.ones(t)
    G = sp.sparse.csc_matrix((VV, (I, J)), shape=(c, t))

    Gm = Mci @ G @ Mt

    if return_mass:
         f = mt[:, 0] / mc[l]
         return G, Gm, mc, mt, f
    return G, Gm