import numpy as np
import scipy as sp

from .arap_hessian import arap_hessian_d2F, arap_hessian_d2S

def elastic_hessian_d2F(F: np.ndarray, mu: np.ndarray, lam: np.ndarray, vol : np.ndarray, material):
        
    if material == 'arap':
        return arap_hessian_d2F(F, mu, vol)
    else:
        raise ValueError("Unknown material type: " + material)
    return


def elastic_hessian_d2x(X : np.ndarray, J : np.ndarray, mu : np.ndarray, lam : np.ndarray, vol : np.ndarray, material):
    dim = X.shape[1]
    x = X.reshape(-1, 1)
    f = J @ x
    F = f.reshape(-1, dim, dim)
    d2psidF2 =  elastic_hessian_d2F(F, mu, lam, vol, material)
    H = sp.sparse.block_diag(d2psidF2)  # block diagonal hessian matrix
    Q = J.transpose() @ H @ J
    return Q



from .elastic_energy import ElasticEnergyZPrecomp
def elastic_hessian_d2z(z : np.ndarray, mu : np.ndarray, lam : np.ndarray, vol : np.ndarray, material, precomp : ElasticEnergyZPrecomp, F=None):
    if F is None:
        dim = precomp.dim
        F = (precomp.JB @ z).reshape(-1, dim, dim)  # should give an option for passing in F so we don't recompute
    d2psidF2 = elastic_hessian_d2F(F, mu, lam, vol, material)
    H = sp.sparse.block_diag(d2psidF2)
    H = precomp.JB.transpose() @ H @ precomp.JB
    return H


def elastic_hessian_d2S(s : np.ndarray, mu : np.ndarray, lam : np.ndarray, vol : np.ndarray, material):
    if material == 'arap':
        return arap_hessian_d2S(s, mu, vol)
    else:
        raise ValueError("Unknown material type: " + material)
    return