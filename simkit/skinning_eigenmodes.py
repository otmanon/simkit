import scipy as sp
import numpy as np

from .lbs_jacobian import lbs_jacobian
from .dirichlet_laplacian import dirichlet_laplacian
from .massmatrix import massmatrix

from .eigs import eigs

import igl 
def skinning_eigenmodes(X, T, k, mu=1, bI=None, Aeq=None):

    M = massmatrix(X, T)
    L = dirichlet_laplacian(X, T, mu=mu)
    # L = igl.cotmatrix(X, T)
    if bI is not None:
        assert(isinstance(bI, np.ndarray))
        Ii = np.setdiff1d(np.arange(X.shape[0]), bI)
        L = L[Ii, :][:, Ii]
        M = sp.sparse.diags(M.diagonal()[Ii,], 0)
        
        [E, Wi] = eigs(L, k, M=M)
        
        # [E, Wi] = sp.sparse.linalg.eigs(L, k=k, M=M, which='LM', sigma=0)
        Wi = Wi.real
        E = E.real
        W = np.zeros((X.shape[0], k))
        W[Ii, :] = Wi
    elif Aeq is not None:
        Z = sp.sparse.csc_matrix((Aeq.shape[0], Aeq.shape[0]))
        H = sp.sparse.bmat([[L , Aeq.T], [Aeq, Z]]).tocsc()
        K = sp.sparse.block_diag([M, Z]).tocsc()

        # [E, W] = eigs(H, k, M=K)

        [E, W] = sp.sparse.linalg.eigs(H, k=k, M=K, which='LM', sigma=0)
        E = E.real
        W = W.real
        W = W[:X.shape[0], :]
    else:
        # [E, W] = sp.sparse.linalg.eigs(L, k=k, M=M, which='LM', sigma=0)
        
        [E, W] = eigs(L, k, M=M)

        E = E.real
        W = W.real
    B = lbs_jacobian(X, W)

    return W, E, B