import numpy as np
import scipy as sp

from .fcr_hessian import fcr_hessian_d2F
from .linear_elasticity_hessian import linear_elasticity_hessian_d2F
from .neo_hookean_hessian import neo_hookean_hessian_d2F, neo_hookean_filtered_hessian_d2F
from .arap_hessian import arap_hessian_d2F, arap_hessian_d2S
from ..deformation_jacobian import deformation_jacobian
from ..volume import volume
from .linear_elasticity_hessian import linear_elasticity_hessian_d2x
from .arap_hessian import arap_hessian_d2x
# from .fcr_hessian import fcr_hessian_d2x


from ..psd_project import psd_project

def elastic_hessian_d2F(F: np.ndarray, mu: np.ndarray, lam: np.ndarray, vol : np.ndarray, material, psd=True):
    
    if material == 'linear-elasticity':
        Q = linear_elasticity_hessian_d2F(F, mu, lam, vol)
    elif material == 'arap':
        Q =  arap_hessian_d2F(F, mu, vol)
    elif material == 'fcr':
        Q =  fcr_hessian_d2F(F, mu, lam, vol)
    elif material == 'neo-hookean':
        Q =  neo_hookean_hessian_d2F(F, mu, lam, vol)
    elif material == 'neo-hookean-filtered':
        Q =  neo_hookean_filtered_hessian_d2F(F, mu, lam, vol)
    else:
        raise ValueError("Unknown material type: " + material)
    
    if psd:
        Qp = psd_project(Q)
    else:
        Qp = Q
    return Qp



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


        
def elastic_hessian(**kwargs):
    """
    Compute the elastic hessian for a given material type.
    """

    if 'X' in kwargs and 'T' in kwargs:
        X = kwargs['X']
        T = kwargs['T']

        if 'U' in kwargs:
            U = kwargs['U']
        else:
            U = kwargs['X'].copy()

        if 'material' in kwargs:
            material = kwargs['material']
        else:
            raise ValueError("Material is required")

        if 'psd' in kwargs:
            psd = kwargs['psd']
        else:
            psd = True
            
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

        if material == 'linear-elasticity':
            Q = linear_elasticity_hessian_d2x(U, mu, lam, vol, J)
        elif material == 'arap':
            Q = arap_hessian_d2x(U, mu, vol, J)
        # elif material == 'fcr':
            # Q = fcr_hessian_d2x(U, mu, lam, vol, J)
        elif material == 'neo-hookean':
            Q = neo_hookean_hessian_d2x(U, mu, lam, vol, J)
        return Q
    else:
        raise ValueError("X and T are required")

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
        F = (precomp.JB @ z + precomp.Jx0).reshape(-1, dim, dim)  # should give an option for passing in F so we don't recompute
    d2psidF2 = elastic_hessian_d2F(F, mu, lam, vol, material)


    H = sp.sparse.block_diag(d2psidF2)
    H = precomp.JB.transpose() @ H @ precomp.JB
    return H



from .elastic_energy import ElasticEnergyZFilteredPrecomp
def elastic_hessian_filtered_d2z(z : np.ndarray, mu : np.ndarray, lam : np.ndarray, vol : np.ndarray, material, precomp : ElasticEnergyZFilteredPrecomp, F=None):
    if F is None:
        dim = precomp.dim
        F = (precomp.JB @ z + precomp.Jx0).reshape(-1, dim, dim)  # should give an option for passing in F so we don't recompute
    d2psidF2 = elastic_hessian_d2F(F, mu, lam, vol, material)


    H = sp.sparse.block_diag(d2psidF2)
    H = precomp.JB.transpose() @ H @ precomp.JB +  precomp.BJAMuJB 
    return H



def elastic_hessian_d2S(s : np.ndarray, mu : np.ndarray, lam : np.ndarray, vol : np.ndarray, material):
    if material == 'arap':
        return arap_hessian_d2S(s, mu, vol)
    else:
        raise ValueError("Unknown material type: " + material)
    return