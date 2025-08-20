# import igl
import scipy as sp
import numpy as np

from .arap_gradient import arap_gradient_dF

from ..volume import volume
from ..deformation_jacobian import deformation_jacobian
from ..polar_svd import polar_svd



def fcr_gradient_dF(F, mu, lam, vol):
    dim = F.shape[-1]

    arap_g = 2 * arap_gradient_dF(F, mu, vol)

    mu = mu.reshape(-1, 1)
    vol = vol.reshape(-1, 1)
    lam = lam.reshape(-1, 1)
    if dim == 2:
        F_00 = F[:, 0, 0].reshape(-1, 1)
        F_01 = F[:, 0, 1].reshape(-1, 1)
        F_10 = F[:, 1, 0].reshape(-1, 1)
        F_11 = F[:, 1, 1].reshape(-1, 1)
        t2 = F_00 * F_11
        t3 = F_01 * F_10
        t4 = -t2
        t5 = t3 + t4 + 1.0

        PK1_volume = np.hstack([
            -F_11 * lam * t5, 
            F_01 * lam * t5,
            F_10 * lam * t5,
            -F_00 * lam * t5
            ]).reshape(-1, 2, 2, order='F') # this is copying matlab code, so need to reshape with F
        
    elif dim == 3:

        # opy all this stuff down there
                
        # F1_1 = F(1);
        # F1_2 = F(4);
        # F1_3 = F(7);
        # F2_1 = F(2);
        # F2_2 = F(5);
        # F2_3 = F(8);
        # F3_1 = F(3);
        # F3_2 = F(6);
        # F3_3 = F(9);
        # la = in2(2);
        # t2 = F1_1.*F2_2.*F3_3;
        # t3 = F1_1.*F2_3.*F3_2;
        # t4 = F1_2.*F2_1.*F3_3;
        # t5 = F1_2.*F2_3.*F3_1;
        # t6 = F1_3.*F2_1.*F3_2;
        # t7 = F1_3.*F2_2.*F3_1;
        # t8 = -t2;
        # t9 = -t5;
        # t10 = -t6;
        # t11 = t3+t4+t7+t8+t9+t10+1.0;

        # g = [-la.*t11.*(F2_2.*F3_3-F2_3.*F3_2);la.*t11.*(F1_2.*F3_3-F1_3.*F3_2);...
        #     -la.*t11.*(F1_2.*F2_3-F1_3.*F2_2);la.*t11.*(F2_1.*F3_3-F2_3.*F3_1);...
        #     -la.*t11.*(F1_1.*F3_3-F1_3.*F3_1);la.*t11.*(F1_1.*F2_3-F1_3.*F2_1);...
        #     -la.*t11.*(F2_1.*F3_2-F2_2.*F3_1);la.*t11.*(F1_1.*F3_2-F1_2.*F3_1);...
        #     -la.*t11.*(F1_1.*F2_2-F1_2.*F2_1)];

        F_00 = F[:, 0, 0].reshape(-1, 1)
        F_01 = F[:, 0, 1].reshape(-1, 1)
        F_02 = F[:, 0, 2].reshape(-1, 1)
        F_10 = F[:, 1, 0].reshape(-1, 1)
        F_11 = F[:, 1, 1].reshape(-1, 1)
        F_12 = F[:, 1, 2].reshape(-1, 1)
        F_20 = F[:, 2, 0].reshape(-1, 1)
        F_21 = F[:, 2, 1].reshape(-1, 1)
        F_22 = F[:, 2, 2].reshape(-1, 1)
        t2 = F_00 * F_11 * F_22
        t3 = F_00 * F_12 * F_21
        t4 = F_01 * F_10 * F_22
        t5 = F_01 * F_12 * F_20
        t6 = F_02 * F_10 * F_21
        t7 = F_02 * F_11 * F_20
        t8 = -t2
        t9 = -t5
        t10 = -t6
        t11 = t3 + t4 + t7 + t8 + t9 + t10 + 1.0

        PK1_volume = np.hstack([
            -lam * t11 * (F_11 * F_22 - F_12 * F_21),
            lam * t11 * (F_01 * F_22 - F_02 * F_21),
            -lam * t11 * (F_01 * F_12 - F_02 * F_11),
            lam * t11 * (F_10 * F_22 - F_12 * F_20),
            -lam * t11 * (F_00 * F_22 - F_02 * F_20),
            lam * t11 * (F_00 * F_12 - F_02 * F_10),
            -lam * t11 * (F_10 * F_21 - F_11 * F_20),
            lam * t11 * (F_00 * F_21 - F_01 * F_20),
            -lam * t11 * (F_00 * F_11 - F_01 * F_10)
            ]).reshape(-1, 3, 3, order='F')


    PK1 = PK1_volume * vol.reshape(-1, 1, 1) + arap_g
    return PK1
