
import numpy as np

from .linear_elasticity import linear_elasticity_gradient_dF
from .neo_hookean import neo_hookean_gradient_dF
from .fcr import fcr_gradient_dF
from .arap import arap_gradient_dF, arap_gradient_dS

def elastic_gradient_dF(F: np.ndarray, mu: np.ndarray, lam: np.ndarray, vol, material):
    
    if material == 'linear-elasticity':
        return linear_elasticity_gradient_dF(F, mu, lam, vol) 
    elif material == 'arap':
        return arap_gradient_dF(F, mu, vol)
    elif material == 'fcr':
        return fcr_gradient_dF(F, mu, lam, vol)
    elif material == 'neo-hookean':
        return neo_hookean_gradient_dF(F, mu, lam, vol)
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
        F = (precomp.JB @ z + precomp.Jx0).reshape(-1, dim, dim)
        
    g = elastic_gradient_dF(F, mu, lam, vol, material)

    g = precomp.JB.transpose() @ g.reshape(-1, 1) 
    return g

from .elastic_energy import ElasticEnergyZFilteredPrecomp

def elastic_gradient_filtered_dz(z: np.ndarray, mu: np.ndarray, lam: np.ndarray, vol, material, precomp : ElasticEnergyZFilteredPrecomp, F=None):
    if F is None:
        dim = precomp.dim
        F = (precomp.JB @ z + precomp.Jx0).reshape(-1, dim, dim)
        
    g = elastic_gradient_dF(F, mu, lam, vol, material)

    precomp_g = precomp.BJAMuJx0 + precomp.BJAMuJB @ z
    g = precomp.JB.transpose() @ g.reshape(-1, 1) + precomp_g
    return g

def elastic_gradient_dS(S : np.ndarray, mu: np.ndarray, lam : np.ndarray, vol, material):
    if material == 'arap':
        return arap_gradient_dS(S, mu, vol)
    else:
        raise ValueError("Unknown material type: "  + material)
    return
