# import igl
import scipy as sp
import numpy as np

from ..volume import volume
from ..deformation_jacobian import deformation_jacobian
from ..polar_svd import polar_svd


from ..mat2py import _4vector_2D_ordering_, _9vector_3D_ordering_

def neo_hookean_gradient_dF(F, mu, lam, vol):

    dim = F.shape[-1]

    if dim ==2:

        F_00 = F[:, 0, 0].reshape(-1, 1)
        F_01 = F[:, 0, 1].reshape(-1, 1)
        F_10 = F[:, 1, 0].reshape(-1, 1)
        F_11 = F[:, 1, 1].reshape(-1, 1)

        g = np.hstack([
            F_00 * mu - F_11 * mu - F_11 * lam * (-F_00 * F_11 + F_01 * F_10 + 1.0),
            F_01 * mu + F_10 * mu + F_01 * lam * (-F_00 * F_11 + F_01 * F_10 + 1.0),
            F_01 * mu + F_10 * mu + F_10 * lam * (-F_00 * F_11 + F_01 * F_10 + 1.0),
            -F_00 * mu + F_11 * mu - F_00 * lam * (-F_00 * F_11 + F_01 * F_10 + 1.0)
            ])[:, _4vector_2D_ordering_].reshape(-1, 2, 2) # this is copying matlab code, so need to reshape with F
        
    elif dim == 3:
       
        F_00 = F[:, 0, 0].reshape(-1, 1)
        F_01 = F[:, 0, 1].reshape(-1, 1)
        F_02 = F[:, 0, 2].reshape(-1, 1)
        F_10 = F[:, 1, 0].reshape(-1, 1)
        F_11 = F[:, 1, 1].reshape(-1, 1)
        F_12 = F[:, 1, 2].reshape(-1, 1)
        F_20 = F[:, 2, 0].reshape(-1, 1)
        F_21 = F[:, 2, 1].reshape(-1, 1)
        F_22 = F[:, 2, 2].reshape(-1, 1)

        
        g = np.hstack([
        -mu * (F_11 * F_22 - F_12 * F_21) + F_00 * mu - lam * (F_11 * F_22 - F_12 * F_21) * (-F_00 * F_11 * F_22 + F_00 * F_12 * F_21 + F_01 * F_10 * F_22 - F_01 * F_12 * F_20 - F_02 * F_10 * F_21 + F_02 * F_11 * F_20 + 1.0),
        mu * (F_01 * F_22 - F_02 * F_21) + F_10 * mu + lam * (F_01 * F_22 - F_02 * F_21) * (-F_00 * F_11 * F_22 + F_00 * F_12 * F_21 + F_01 * F_10 * F_22 - F_01 * F_12 * F_20 - F_02 * F_10 * F_21 + F_02 * F_11 * F_20 + 1.0),
        -mu * (F_01 * F_12 - F_02 * F_11) + F_20 * mu - lam * (F_01 * F_12 - F_02 * F_11) * (-F_00 * F_11 * F_22 + F_00 * F_12 * F_21 + F_01 * F_10 * F_22 - F_01 * F_12 * F_20 - F_02 * F_10 * F_21 + F_02 * F_11 * F_20 + 1.0),
        mu * (F_10 * F_22 - F_12 * F_20) + F_01 * mu + lam * (F_10 * F_22 - F_12 * F_20) * (-F_00 * F_11 * F_22 + F_00 * F_12 * F_21 + F_01 * F_10 * F_22 - F_01 * F_12 * F_20 - F_02 * F_10 * F_21 + F_02 * F_11 * F_20 + 1.0),
        -mu * (F_00 * F_22 - F_02 * F_20) + F_11 * mu - lam * (F_00 * F_22 - F_02 * F_20) * (-F_00 * F_11 * F_22 + F_00 * F_12 * F_21 + F_01 * F_10 * F_22 - F_01 * F_12 * F_20 - F_02 * F_10 * F_21 + F_02 * F_11 * F_20 + 1.0),
        mu * (F_00 * F_12 - F_02 * F_10) + F_21 * mu + lam * (F_00 * F_12 - F_02 * F_10) * (-F_00 * F_11 * F_22 + F_00 * F_12 * F_21 + F_01 * F_10 * F_22 - F_01 * F_12 * F_20 - F_02 * F_10 * F_21 + F_02 * F_11 * F_20 + 1.0),
        -mu * (F_10 * F_21 - F_11 * F_20) + F_02 * mu - lam * (F_10 * F_21 - F_11 * F_20) * (-F_00 * F_11 * F_22 + F_00 * F_12 * F_21 + F_01 * F_10 * F_22 - F_01 * F_12 * F_20 - F_02 * F_10 * F_21 + F_02 * F_11 * F_20 + 1.0),
        mu * (F_00 * F_21 - F_01 * F_20) + F_12 * mu + lam * (F_00 * F_21 - F_01 * F_20) * (-F_00 * F_11 * F_22 + F_00 * F_12 * F_21 + F_01 * F_10 * F_22 - F_01 * F_12 * F_20 - F_02 * F_10 * F_21 + F_02 * F_11 * F_20 + 1.0),
        -mu * (F_00 * F_11 - F_01 * F_10) + F_22 * mu - lam * (F_00 * F_11 - F_01 * F_10) * (-F_00 * F_11 * F_22 + F_00 * F_12 * F_21 + F_01 * F_10 * F_22 - F_01 * F_12 * F_20 - F_02 * F_10 * F_21 + F_02 * F_11 * F_20 + 1.0)
        ])[:, _9vector_3D_ordering_].reshape(-1, 3, 3)

    PK1 = g * vol.reshape(-1, 1, 1)
    return PK1

def neo_hookean_filtered_gradient_dF(F, mu, lam, vol):
    
    dim = F.shape[-1]
    
    mu = np.array(mu).reshape(-1)
    lam = np.array(lam).reshape(-1)
    vol = np.array(vol).reshape(-1)
    
    if dim == 2:
        F_00 = F[:, 0, 0]
        F_01 = F[:, 0, 1]
        F_10 = F[:, 1, 0]
        F_11 = F[:, 1, 1]
        

        g = np.array(
            [
                [
                    1.0 * F_11 * lam * (F_00 * F_11 - F_01 * F_10 - 1.0) - F_11 * mu,
                    -1.0 * F_10 * lam * (F_00 * F_11 - F_01 * F_10 - 1.0) + F_10 * mu,
                ],
                [
                    -1.0 * F_01 * lam * (F_00 * F_11 - F_01 * F_10 - 1.0) + F_01 * mu,
                    1.0 * F_00 * lam * (F_00 * F_11 - F_01 * F_10 - 1.0) - F_00 * mu,
                ],
            ]
        )
    elif dim == 3:
        raise ValueError("Not implemented")
        
    PK1 = g.transpose(2, 0, 1) * vol.reshape(-1, 1, 1)
    return PK1