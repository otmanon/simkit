
import numpy as np
import scipy as sp

from .deformation_jacobian import deformation_jacobian
from .cluster_grouping_matrices import cluster_grouping_matrices
from .polar_svd import polar_svd
from .volume import volume
from .cluster_grouping_matrices import cluster_grouping_matrices


def subspace_rotation(z, B, X, T, GAJB=None, return_GAJB=False, labels=None):
    """
    Compute the rotation matrix for each cluster described in labels in the current state in the subspace defined by B.

    Parameters
    ----------
    z : (m,1) np.ndarray 
        The reduced current positions.
    B : (m,n) np.ndarray 
        The basis of the subspace.
    X : (n,(2|3)) np.ndarray 
        The rest vertex positions
    T : (t, (3|4)) np.ndarray 
        mesh simplicies
    GAJB : (dim*dim * k, m) np.ndarray, optional
        The averaging deformation jacobian map
    return_GAJB : bool, optional
        whether to return the averaging deformation jacobian map
    labels : (t,) np.ndarray, optional
        The cluster labels. If None, assumes a single clusters enclosing the entire mesh.
      
    Returns
    -------
    R : (k, dim, dim) np.ndarray
        The rotation matrix for each cluster.
    GAJB : (dim*dim * k, m) np.ndarray, optional
        The averaging deformation jacobian map

    ```

    """
    # Get the rotation matrix for the current state
    dim = X.shape[1]
    
    
    if GAJB is None:
        if labels is None:
            labels = np.zeros(T.shape[0]).astype(int)
            
        J = deformation_jacobian(X, T)
        A = sp.sparse.diags(volume(X, T).flatten())
        [G, Gm] = cluster_grouping_matrices(labels, X, T)
        GAe = sp.sparse.kron(G @ A, sp.sparse.identity(dim*dim))
        GAJB = GAe @ J @ B

    c = GAJB @ z 
    C = c.reshape((dim, dim, -1)).transpose(2, 0, 1) # covariances / deformation gradients
    R, Sf = polar_svd(C)  # this is the best fit rotation of the current state
    
    if return_GAJB:
        return R, GAJB
    else:
        return R