from types import NoneType
import igl
import scipy as sp
import numpy as np

from simkit import volume
from simkit import rotation_gradient_F
from simkit import deformation_jacobian

def arap_hessian(**kwargs):
    
    if 'mu' in kwargs:
        mu = kwargs['mu']
    else :
        mu = 1

    if 'X'  in kwargs and 'T' in kwargs:
        X = kwargs['X']
        T = kwargs['T']

        if 'U' in kwargs:
            U = kwargs['U']
        else:
            U = kwargs['X'].copy()
        
        if 'J' in kwargs:
            J = kwargs['J']
        else:        
            J = deformation_jacobian(X, T)

        if 'vol' in kwargs:
            vol = kwargs['vol']
        else:
            vol = volume(X, T)

        return arap_hessian_d2x(U, J, mu=mu, vol=volume(X, T))
    else:
        ValueError("X and T are required")

    


def arap_hessian_d2F(F, mu=1, vol=1):
    dim = F.shape[-1]
    F = F.reshape(-1, dim, dim)
    n = F.shape[0]
    I = np.tile(np.identity(dim * dim), (n, 1, 1))
    H = 2 * I - 2 *  rotation_gradient_F(F)
    d = mu * vol
    H *= d.reshape(-1, 1, 1)
    return H

# def arap_hessian(X, T, U = None, mu=1, J=None):
#     if U is None:
#         U = X.copy()
#     x = U.reshape(-1, 1)
#     H = arap_hessian_x(x, X, T, mu=mu, J=None)
#     return H

def arap_hessian_d2x(X, J, mu,  vol):
    dim = X.shape[1]
    f = J @ X.reshape(-1, 1)
    F = f.reshape(-1, dim, dim)
    d2psidF2 = arap_hessian_d2F(F, mu=mu, vol=vol)
    H = sp.sparse.block_diag(d2psidF2)  # block diagonal hessian matrix
    Q = J.transpose() @ H @ J
    return Q


def arap_hessian_d2S(s, mu, vol):
    """
    E_arap(s) = 0.5 * sum_i mu_i * vol_i * ||S_i - I_i||^2

    H_s = mu * vol * I

    """
    if s.ndim == 3:
        t = s.shape[0]
        dim = s.shape[-1]

        ones = np.ones((t, 1))
        h = (mu * vol * ones)
        h = np.repeat(h, dim*dim, axis=0)
        H_s = sp.sparse.diags(h.flatten())
    elif s.ndim == 2:
        k = s.shape[-1]
        if k == 3:
            dim = 2
            w = np.array([ 1, 1, 2])[None, :]
            i = np.array([1, 1, 0])[None, :]
        elif k == 6:
            dim = 3
            w = np.array([ 1, 1, 1, 2, 2, 2])[None, :]
            i = np.array([1, 1, 1, 0, 0, 0])[None, :]
        else:
            raise ValueError("Unknown dimension, k must be 3 (for 2D) or 6 (for 3D)")
        
        ones = np.ones((1, s.shape[-1]))
        h = (mu * vol * ones * w)
        H_s = sp.sparse.diags(h.flatten())
            
    return H_s 
