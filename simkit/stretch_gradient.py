

import numpy as np

from .polar_svd import polar_svd
from .rotation_gradient import rotation_gradient_F
import scipy as sp

def stretch_gradient(F):
    return stretch_gradient_dF(F)

def stretch_gradient_dF(F):
    [R, S] = polar_svd(F)
    dim = F.shape[-1]
    dRdF = rotation_gradient_F(F).reshape(-1, dim, dim, dim, dim)
    dRdFF = np.einsum('...mnki, ...kj->...mnij', dRdF, F)
    I = np.zeros((dim, dim, dim, dim))
    for i in range(dim):
        for j in range(dim):
            I[i, j, i, j] = 1
    RI = np.einsum('...ki, ...mnkj->...mnij', R, I)
    dSdF = dRdFF + RI
    return dSdF


def stretch_gradient_dx(X, J, Ci=None, dim=None, Jq=None):
    if dim is None:
        dim = X.shape[1]
    x = X.reshape(-1, 1)
    
    if Jq is None:
        F = (J @ x).reshape(-1, dim, dim)
    else:
        F = (J @ x + Jq).reshape(-1, dim, dim)
        
    dSdF = stretch_gradient(F).reshape(-1, dim * dim, dim * dim)
    dSdFb = sp.sparse.block_diag(dSdF)
    dsdx = J.T @ dSdFb

    if Ci is not None:
        dsdx = dsdx @ Ci.T
    return dsdx

def stretch_gradient_dz(z, GJB, dim, Ci=None, GJq=None):
    
    dsdx = stretch_gradient_dx(z, GJB, Ci, dim, Jq=GJq)

    return dsdx