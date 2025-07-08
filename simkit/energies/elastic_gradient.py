
import numpy as np
from .arap_gradient import arap_gradient_dF, arap_gradient_dS

def elastic_gradient_dF(F: np.ndarray, mu: np.ndarray, lam: np.ndarray, vol, material):

    if material == 'arap':
        return arap_gradient_dF(F, mu, vol)
    else:
        raise ValueError("Unknown material type: "  + material)
    return



def elastic_gradient_dx(X: np.ndarray, J: np.ndarray, mu: np.ndarray, lam: np.ndarray, vol, material):
    dim = X.shape[1]
    x = X.reshape(-1, 1)
    F = (J @ x).reshape(-1, dim, dim)
    
    PK1 = elastic_gradient_dF(F, mu, lam, vol, material)

    g = J.transpose() @ PK1.reshape(-1, 1)

    return g

from .elastic_energy import ElasticEnergyZPrecomp

def elastic_gradient_dz(z: np.ndarray, mu: np.ndarray, lam: np.ndarray, vol, material, precomp : ElasticEnergyZPrecomp, F=None):
    if F is None:
        dim = precomp.dim
        F = (precomp.JB @ z).reshape(-1, dim, dim)
    g = elastic_gradient_dF(F, mu, lam, vol, material)

    g = precomp.JB.transpose() @ g.reshape(-1, 1)
    return g

def elastic_gradient_dS(S : np.ndarray, mu: np.ndarray, lam : np.ndarray, vol, material):
    if material == 'arap':
        return arap_gradient_dS(S, mu, vol)
    else:
        raise ValueError("Unknown material type: "  + material)
    return
