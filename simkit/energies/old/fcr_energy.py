import numpy as np


from ..volume import volume
from ..deformation_jacobian import deformation_jacobian
from ..polar_svd import polar_svd

# def arap_energy(X, T, U=None, mu=1, vol=None):
#     if U is None:
#         U = X.copy()
#     x = U.reshape(-1, 1)
#     e = arap_energy_x(x, X, T, mu=mu, vol=vol)
#     return e


# def arap_energy_S(s, mu, vol):
#     assert (s.ndim == 2 or s.ndim == 3)
#     if s.ndim == 3:
#         dim = s.shape[-1]
#         psi = s - np.eye(dim)[None, :, :]
#         E = 0.5 * np.sum((mu * vol) * np.sum(psi**2, axis=(1, 2))[:, None])
#     if s.ndim == 2:
#         k = s.shape[-1]
#         # relationship between k =  dim * (dim + 1)/2
#         if k == 3:
#             dim = 2
#             w = np.array([ 1, 1, 2])[None, :]
#             i = np.array([1, 1, 0])[None, :]
#         elif k == 6:
#             dim = 3
#             w = np.array([ 1, 1, 1, 2, 2, 2])[None, :]
#             i = np.array([1, 1, 1, 0, 0, 0])[None, :]
#         else:
#             raise ValueError("Unknown dimension, k must be 3 (for 2D) or 6 (for 3D)")
        
#         psi = (s - i)
#         E = 0.5 * np.sum((mu * vol) * np.sum(psi**2 * w, axis=1)[:, None])
#     return E

def fcr_energy_F(F, mu, lam, vol):

    mu = mu.reshape(-1, 1)
    vol = vol.reshape(-1, 1)
    lam = lam.reshape(-1, 1)
    arap_e = 2 * arap_energy_F(F, mu,  vol);

    dim = F.shape[1]
    if dim == 2:
        I3 = np.linalg.det(F).reshape(-1, 1)
        E_vol = 0.5 * lam * vol * ((I3 - 1.0 )**2);
    elif dim == 3:
        I3 = np.linalg.det(F).reshape(-1, 1)
        E_vol = 0.5 * lam * vol * ((I3  - 1.0 )**2);
    
    E = np.sum(E_vol)[None] + arap_e
    return E

# def arap_energy(V, F, mu=None, U=None):
#     """
#         Computes the ARAP energy evaluated at a given displacement U
#
#         Parameters
#         ----------
#         V : (n, 3) array
#             The input mesh vertices
#         F : (m, 4) array
#             Tet mesh indices
#         mu : float or (m, 1) array
#             first lame parameter, defaults to 1.0
#         U : (n, 3) array
#             The deformed  mesh vertices
#
#         Returns
#         -------
#         E : float
#             The ARAP energy
#     """
#     dim = V.shape[1]
#
#     sim = F.shape[1]
#     if (mu is None):
#         mu = np.ones((F.shape[0]))
#     elif (np.isscalar(mu)):
#         mu = mu* np.ones((F.shape[0]))
#     else:
#         assert(mu.shape[0] == F.shape[0])
#
#     if U is None:
#         U = V.copy() # assume at rest
#
#     assert(V.shape == U.shape)
#     assert(V.shape[0] >= F.max())
#     assert((sim == 4 or sim == 3) and "Only tet or triangle meshes supported")
#
#     if dim == 2:
#         assert(sim == 3)
#     if dim == 3 :
#         assert(sim == 4)
#
#     vol = volume(V, F)
#
#     J = deformation_jacobian(V, F)
#     f = J @ U.reshape(-1, 1)
#
#     F = np.reshape(f, (-1, dim, dim))
#     [R, S] = polar_svd(F)
#
#     E = 0.5 * np.sum((mu * vol) * np.sum((F - R)**2))
#
#     return E
#
#
