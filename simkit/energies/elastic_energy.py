import numpy as np

from .fcr_energy import fcr_energy_F
from .linear_elasticity_energy import linear_elasticity_energy_F
from .arap_energy import arap_energy_F, arap_energy_S
from .neo_hookean_energy import neo_hookean_energy_F, neo_hookean_filtered_energy_F


def elastic_energy(F: np.ndarray,  mu: np.ndarray, lam: np.ndarray, vol : np.ndarray, material):

    if material == 'linear-elasticity':
        e = linear_elasticity_energy_F(F,  mu, lam,  vol)
    elif material == 'arap':
        e = arap_energy_F(F,  mu,  vol)
    elif material == 'fcr':
        e = fcr_energy_F(F, mu, lam, vol)
    elif material == 'neo-hookean':
        e = neo_hookean_energy_F(F, mu, lam, vol)
    elif material == 'neo-hookean-filtered':
        e = neo_hookean_filtered_energy_F(F, mu, lam, vol)
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
    def __init__(self, B, x0, G, J, dim):
        self.JB = G @ J @ B
        self.dim = dim

        if x0 is None:
            x0 = np.zeros((B.shape[0], 1))
        self.Jx0 = G @ J @ x0


import scipy as sp

class ElasticEnergyZFilteredPrecomp(ElasticEnergyZPrecomp):
    def __init__(self, B, x0, G, J, dim, mu, vol):
        super().__init(B, x0, G, J, dim)
        
        AMu = sp.sparse.diags(mu * vol)
        
        AMue = sp.sparse.kron(AMu, sp.sparse.eye(dim*dim))
        JAMuJ = J.T @ AMue @ J
        BJAMuJ = B.T @ JAMuJ
        BJAMuJB = BJAMuJ @ B
        
        self.BJAMuJB = BJAMuJB
        self.BJAMuJx0 = BJAMuJ @ x0
        
        return
        
        
        
def elastic_energy_z(z: np.ndarray, mu : np.ndarray, lam:np.ndarray, vol : np.ndarray, material, precomp : ElasticEnergyZPrecomp, F=None):
    if F is None:
        dim = precomp.dim
        F = (precomp.JB @ z + precomp.Jx0).reshape(-1, dim, dim)
    e = elastic_energy(F, mu, lam, vol, material)
    return e



        
def elastic_energy_filtered_z(z: np.ndarray, mu : np.ndarray, lam:np.ndarray, vol : np.ndarray, material, precomp : ElasticEnergyZFilteredPrecomp, F=None):
    if F is None:
        dim = precomp.dim
        F = (precomp.JB @ z + precomp.Jx0).reshape(-1, dim, dim)
    e = elastic_energy(F, mu, lam, vol, material)
    
    e += 0.5 * z.T @ precomp.BJAMuJB @ z + z.T @ precomp.BJAMuJx0
    return e




def elastic_energy_S(S : np.ndarray, mu: np.ndarray, lam : np.ndarray, vol : np.ndarray, material):

    if material == 'arap':
        e = arap_energy_S(S, mu, vol)
    # elif material == 'fcr':
    #     e = fcr_energy_S(S, mu, lam, vol)
    # elif material == 'neo-hookean':
    #     e = neo_hookean_energy_S(S, mu, lam, vol)
    else:
        raise ValueError("Unknown material type: " + material)
    return e

