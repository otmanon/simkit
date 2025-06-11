import numpy as np


from .arap_energy import arap_energy_F, arap_energy_S


def elastic_energy(F: np.ndarray,  mu: np.ndarray, lam: np.ndarray, vol : np.ndarray, material):

    if material == 'linear_elasticity':
        raise NotImplemented
    elif material == 'arap':
        e = arap_energy_F(F,  mu,  vol)
    elif material == 'corot':
        raise NotImplemented
    elif material == 'neo_hookean':
        raise NotImplemented
    else:
        raise ValueError("Unknown material type: " + material)
    return e


def elastic_energy_x(X: np.ndarray, J: np.ndarray, mu: np.ndarray, lam: np.ndarray, vol : np.ndarray, material):

    dim = X.shape[1]
    x = X.reshape(-1, 1)
    F = (J @ x).reshape(-1, dim, dim)  
    
    e = elastic_energy(F, mu, lam, vol, material)
    return e


class ElasticEnergyZPrecomp():
    def __init__(self, B, G, J, dim):
        self.JB = G @ J @ B
        self.dim = dim

def elastic_energy_z(z: np.ndarray, mu : np.ndarray, lam:np.ndarray, vol : np.ndarray, material, precomp : ElasticEnergyZPrecomp, F=None):
    if F is None:
        dim = precomp.dim
        F = (precomp.JB @ z).reshape(-1, dim, dim)
    e = elastic_energy(F, mu, lam, vol, material)
    return e


def elastic_energy_S(S : np.ndarray, mu: np.ndarray, lam : np.ndarray, vol : np.ndarray, material):

    if material == 'arap':
        e = arap_energy_S(S, mu, vol)
    else:
        raise ValueError("Unknown material type: " + material)
    return e

