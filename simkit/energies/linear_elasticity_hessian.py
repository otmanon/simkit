import numpy as np
import scipy as sp

from ..deformation_jacobian import deformation_jacobian
from ..vectorized_trace import vectorized_trace
from ..vectorized_transpose import vectorized_transpose
from ..volume import volume
from ..polar_svd import polar_svd


def linear_elasticity_hessian(**kwargs):
    if 'mu' in kwargs:
        mu = kwargs['mu']
    else :
        mu = 1

    if 'lam' in kwargs:
        lam = kwargs['lam']
    else :
        lam = 1

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

        return linear_elasticity_hessian_d2x(U,mu, lam, vol, J=J)
    else:
        ValueError("X and T are required")


def linear_elasticity_hessian_d2F(F, mu, lam, vol):
    
    mu = np.array(mu)
    lam = np.array(lam)
    vol = np.array(vol)
    assert(F.ndim == 3)
    assert(F.shape[1] == F.shape[2])

    dim = F.shape[1]

    I = np.identity(dim*dim)
    T = np.asarray(vectorized_transpose(1, F.shape[1]).toarray())
    Tr = np.asarray(vectorized_trace(1, F.shape[1]).toarray())


    H1 = np.repeat((I + T)[None, :, :], F.shape[0], axis=0)
    H2 = np.repeat((Tr.T @ Tr)[None, :, :], F.shape[0], axis=0)

    H = (mu.reshape(-1, 1, 1) * H1 + lam.reshape(-1, 1, 1) * H2)*vol.flatten()[:, None, None]

    return H


def linear_elasticity_hessian_d2x(U, mu,  lam, vol, J):

    dim = U.shape[1]
    F = (J @ U.reshape(-1, 1)).reshape(-1, dim, dim)
    d2psidF2 = linear_elasticity_hessian_d2F(F, mu=mu, lam=lam, vol=vol)
    H = sp.sparse.block_diag(d2psidF2)  # block diagonal hessian matrix
    Q = J.transpose() @ H @ J
    return Q
