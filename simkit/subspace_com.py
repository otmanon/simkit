import numpy as np
import scipy as sp

from .massmatrix import massmatrix

def subspace_com(z : np.ndarray, B: np.ndarray,  X : np.ndarray, T : np.ndarray, return_SB:bool=False, SB : np.ndarray=None):    
    dim = X.shape[1]

    if SB is None:
        M = massmatrix(X, T)
        m = M.diagonal()
        total_mass = m.sum()
        rho = m / total_mass
        S = sp.sparse.kron(rho, np.identity(dim))
        SB = S @ B
    
    com = (SB @ z).reshape(-1, dim)
    if return_SB:
        return com, SB
    else:
        return com
