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

def neo_hookean_energy_F(F, mu, lam, vol):

    dim = F.shape[-1]

    F = F.reshape(-1, dim, dim)
    
    if dim == 2:
        # convert the cpp code to python
        # double mu = config_->mu;
        # double la = config_->la;
        # double F1_1 = F(0);
        # double F2_1 = F(1);
        # double F1_2 = F(2);
        # double F2_2 = F(3);
        # return mu*(-F1_1*F2_2+F1_2*F2_1+1.0)+(la*pow(-F1_1*F2_2+F1_2*F2_1+1.0,2.0))/2.0+(mu*(F1_1*F1_1+F1_2*F1_2+F2_1*F2_1+F2_2*F2_2-2.0))/2.0;
        F_00 = F[:, 0, 0].reshape(-1, 1)
        F_01 = F[:, 0, 1].reshape(-1, 1)
        F_10 = F[:, 1, 0].reshape(-1, 1)
        F_11 = F[:, 1, 1].reshape(-1, 1)

        e = mu * (-F_00 * F_11 + F_01 * F_10 + 1.0) + (lam * (-(F_00 * F_11) + F_01 * F_10 + 1.0)**2) / 2.0 + (mu * (F_00**2 + F_01**2 + F_10**2 + F_11**2 - 2.0)) / 2.0
    
    if dim == 3:

        # if dim == 3:

        # convert cpp code to python
    #       double mu = config_->mu;
    #   double la = config_->la;
    #   double F1_1 = F(0);
    #   double F2_1 = F(1);
    #   double F3_1 = F(2);
    #   double F1_2 = F(3);
    #   double F2_2 = F(4);
    #   double F3_2 = F(5);
    #   double F1_3 = F(6);
    #   double F2_3 = F(7);
    #   double F3_3 = F(8);
    #   return mu*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0)+(la*pow(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0,2.0))/2.0+(mu*(F1_1*F1_1+F1_2*F1_2+F1_3*F1_3+F2_1*F2_1+F2_2*F2_2+F2_3*F2_3+F3_1*F3_1+F3_2*F3_2+F3_3*F3_3-3.0))/2.0;
    
        F_00 = F[:, 0, 0].reshape(-1, 1)
        F_01 = F[:, 0, 1].reshape(-1, 1)
        F_02 = F[:, 0, 2].reshape(-1, 1)
        F_10 = F[:, 1, 0].reshape(-1, 1)
        F_11 = F[:, 1, 1].reshape(-1, 1)
        F_12 = F[:, 1, 2].reshape(-1, 1)
        F_20 = F[:, 2, 0].reshape(-1, 1)
        F_21 = F[:, 2, 1].reshape(-1, 1)
        F_22 = F[:, 2, 2].reshape(-1, 1)

        e = mu * (-F_00 * F_11 * F_22 + F_00 * F_12 * F_21 + F_01 * F_10 * F_22 - F_01 * F_12 * F_20 - F_02 * F_10 * F_21 + F_02 * F_11 * F_20 + 1.0) + (lam * (-(F_00 * F_11 * F_22) + F_00 * F_12 * F_21 + F_01 * F_10 * F_22 - F_01 * F_12 * F_20 - F_02 * F_10 * F_21 + F_02 * F_11 * F_20 + 1.0)**2) / 2.0 + (mu * (F_00**2 + F_01**2 + F_02**2 + F_10**2 + F_11**2 + F_12**2 + F_20**2 + F_21**2 + F_22**2 - 3.0)) / 2.0

    E = np.array([(e * vol.reshape(-1, 1)).sum()])
    return E



def neo_hookean_filtered_energy_F(F, mu, lam, vol):
    
    dim = F.shape[-1]
    
    lam = np.array(lam).reshape(-1)
    mu = np.array(mu).reshape(-1)
    vol = np.array(vol).reshape(-1)
    
    if dim == 2:
        F_00 = F[:, 0, 0]
        F_01 = F[:, 0, 1]
        F_10 = F[:, 1, 0]
        F_11 = F[:, 1, 1]

        
        e = 0.5*lam*(F_00*F_11 - F_01*F_10 - 1.0)**2 - mu*(F_00*F_11 - F_01*F_10 - 1.0)
    if dim == 3:
        raise ValueError("Not implemented")
    
    E = np.array([(e * vol.reshape(-1)).sum()])
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
