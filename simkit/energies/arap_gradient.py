# import igl
import scipy as sp
import numpy as np

from ..volume import volume
from ..deformation_jacobian import deformation_jacobian
from ..polar_svd import polar_svd



def arap_gradient_dF(F, mu, vol):
    dim = F.shape[-1]
    F = F.reshape(-1, dim, dim)
    [R, S] = polar_svd(F)
    PK1 = F - R
    d = mu * vol
    PK1 *= d.reshape(-1, 1, 1)
    return PK1

# def arap_gradient( X, T, U=None, mu=1, J=None, vol=None):

#     if U is None:
#         U = X.copy()
#     x = U.reshape(-1, 1)
#     g = arap_gradient_x(x, X, T, mu=mu, J=None, vol=vol)

#     return g


def arap_gradient_dx(x, J, mu, vol):
    dim = V.shape[1]

    f = J @ x
    F = f.reshape(-1, dim, dim)

    dpsi_dF = arap_gradient_dF(F, mu=mu, vol=vol)
    pk1 = dpsi_dF.reshape(-1, 1)

    g = J.transpose() @ pk1

    return g


def arap_gradient_dS(s, mu, vol):
    """
    E_arap(s) = 0.5 * sum_i mu_i * vol_i * ||S_i - I_i||^2
    PK1_s = mu * vol * (S - I)
    """
    if s.ndim == 3:
        dim = s.shape[1]
        S = s.reshape(-1, dim, dim)
        
        PK1_S = (mu * vol)[:, None] * (S - np.eye(dim)[None, :, :])
        pk1_s = PK1_S.reshape(-1, 1)
    if s.ndim == 2:
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
        
        psi = (s - i)
        pk1_s = ((mu * vol) * (psi * w)).reshape(-1, 1)


    return pk1_s

# def arap_gradient(V, T, mu=None, U=None):
#     """
#
#         Computes the ARAP Gradient Vector at a given displacement U
#
#     """
#     dim = V.shape[1]
#     # B = deformation_jacobian(V, F);
#     if (mu is None):
#         mu = np.ones((T.shape[0]))
#     elif (np.isscalar(mu)):
#         mu = mu* np.ones((T.shape[0]))
#     else:
#         assert(mu.shape[0] == T.shape[0])
#
#     if U is None:
#         U = V.copy() # assume at rest
#
#     if dim == 2:
#         vol = igl.doublearea(V, T).reshape(-1)/2
#     elif dim == 3:
#         vol = igl.volume(V, T).reshape(-1)
#     else:
#         ValueError("Only V.shape[1] == 2 or 3 are supported")
#
#     J = deformation_jacobian(V, T)
#     f = J @ U.flatten()
#     F = np.reshape(f, (-1, dim, dim))
#     dpsi_dF = darap_dF(F, mu=mu, a=vol)
#     pk1 = dpsi_dF.reshape(-1, 1)
#     g = J.transpose() @ pk1
#     return g
#
#
