# import numpy as np
# import scipy as sp
# from ..mat2py import _4vector_2D_ordering_, _9vector_3D_ordering_, _4x4matrix_2D_ordering_, _9x9matrix_3D_ordering_

# def neo_hookean_energy_F(F, mu, lam, vol):

#     dim = F.shape[-1]

#     F = F.reshape(-1, dim, dim)
    
#     if dim == 2:
#         # convert the cpp code to python
#         F_00 = F[:, 0, 0].reshape(-1, 1)
#         F_01 = F[:, 0, 1].reshape(-1, 1)
#         F_10 = F[:, 1, 0].reshape(-1, 1)
#         F_11 = F[:, 1, 1].reshape(-1, 1)

#         e = mu * (-F_00 * F_11 + F_01 * F_10 + 1.0) + (lam * (-(F_00 * F_11) + F_01 * F_10 + 1.0)**2) / 2.0 + (mu * (F_00**2 + F_01**2 + F_10**2 + F_11**2 - 2.0)) / 2.0
    
#     if dim == 3:
#         F_00 = F[:, 0, 0].reshape(-1, 1)
#         F_01 = F[:, 0, 1].reshape(-1, 1)
#         F_02 = F[:, 0, 2].reshape(-1, 1)
#         F_10 = F[:, 1, 0].reshape(-1, 1)
#         F_11 = F[:, 1, 1].reshape(-1, 1)
#         F_12 = F[:, 1, 2].reshape(-1, 1)
#         F_20 = F[:, 2, 0].reshape(-1, 1)
#         F_21 = F[:, 2, 1].reshape(-1, 1)
#         F_22 = F[:, 2, 2].reshape(-1, 1)

#         e = mu * (-F_00 * F_11 * F_22 + F_00 * F_12 * F_21 + F_01 * F_10 * F_22 - F_01 * F_12 * F_20 - F_02 * F_10 * F_21 + F_02 * F_11 * F_20 + 1.0) + (lam * (-(F_00 * F_11 * F_22) + F_00 * F_12 * F_21 + F_01 * F_10 * F_22 - F_01 * F_12 * F_20 - F_02 * F_10 * F_21 + F_02 * F_11 * F_20 + 1.0)**2) / 2.0 + (mu * (F_00**2 + F_01**2 + F_02**2 + F_10**2 + F_11**2 + F_12**2 + F_20**2 + F_21**2 + F_22**2 - 3.0)) / 2.0

#     E = np.array([(e * vol.reshape(-1, 1)).sum()])
#     return E




# import torch
# def neo_hookean_energy_F_torch(F, mu, lam, vol):

#     dim = F.shape[-1]

#     F = F.reshape(-1, dim, dim)
    
#     if dim == 2:
#         # convert the cpp code to python
#         F_00 = F[:, 0, 0].reshape(-1, 1)
#         F_01 = F[:, 0, 1].reshape(-1, 1)
#         F_10 = F[:, 1, 0].reshape(-1, 1)
#         F_11 = F[:, 1, 1].reshape(-1, 1)

#         e = mu * (-F_00 * F_11 + F_01 * F_10 + 1.0) + (lam * (-(F_00 * F_11) + F_01 * F_10 + 1.0)**2) / 2.0 + (mu * (F_00**2 + F_01**2 + F_10**2 + F_11**2 - 2.0)) / 2.0
    
#     if dim == 3:
#         F_00 = F[:, 0, 0].reshape(-1, 1)
#         F_01 = F[:, 0, 1].reshape(-1, 1)
#         F_02 = F[:, 0, 2].reshape(-1, 1)
#         F_10 = F[:, 1, 0].reshape(-1, 1)
#         F_11 = F[:, 1, 1].reshape(-1, 1)
#         F_12 = F[:, 1, 2].reshape(-1, 1)
#         F_20 = F[:, 2, 0].reshape(-1, 1)
#         F_21 = F[:, 2, 1].reshape(-1, 1)
#         F_22 = F[:, 2, 2].reshape(-1, 1)

#         e = mu * (-F_00 * F_11 * F_22 + F_00 * F_12 * F_21 + F_01 * F_10 * F_22 - F_01 * F_12 * F_20 - F_02 * F_10 * F_21 + F_02 * F_11 * F_20 + 1.0) + (lam * (-(F_00 * F_11 * F_22) + F_00 * F_12 * F_21 + F_01 * F_10 * F_22 - F_01 * F_12 * F_20 - F_02 * F_10 * F_21 + F_02 * F_11 * F_20 + 1.0)**2) / 2.0 + (mu * (F_00**2 + F_01**2 + F_02**2 + F_10**2 + F_11**2 + F_12**2 + F_20**2 + F_21**2 + F_22**2 - 3.0)) / 2.0


#     E = (e * vol.reshape(-1, 1))
#     return E.sum()

# def neo_hookean_gradient_dF(F, mu, lam, vol):

#     dim = F.shape[-1]

#     if dim ==2:

#         F_00 = F[:, 0, 0].reshape(-1, 1)
#         F_01 = F[:, 0, 1].reshape(-1, 1)
#         F_10 = F[:, 1, 0].reshape(-1, 1)
#         F_11 = F[:, 1, 1].reshape(-1, 1)

#         g = np.hstack([
#             F_00 * mu - F_11 * mu - F_11 * lam * (-F_00 * F_11 + F_01 * F_10 + 1.0),
#             F_01 * mu + F_10 * mu + F_01 * lam * (-F_00 * F_11 + F_01 * F_10 + 1.0),
#             F_01 * mu + F_10 * mu + F_10 * lam * (-F_00 * F_11 + F_01 * F_10 + 1.0),
#             -F_00 * mu + F_11 * mu - F_00 * lam * (-F_00 * F_11 + F_01 * F_10 + 1.0)
#             ])[:, _4vector_2D_ordering_].reshape(-1, 2, 2) # this is copying matlab code, so need to reshape with F
        
#     elif dim == 3:
       
#         F_00 = F[:, 0, 0].reshape(-1, 1)
#         F_01 = F[:, 0, 1].reshape(-1, 1)
#         F_02 = F[:, 0, 2].reshape(-1, 1)
#         F_10 = F[:, 1, 0].reshape(-1, 1)
#         F_11 = F[:, 1, 1].reshape(-1, 1)
#         F_12 = F[:, 1, 2].reshape(-1, 1)
#         F_20 = F[:, 2, 0].reshape(-1, 1)
#         F_21 = F[:, 2, 1].reshape(-1, 1)
#         F_22 = F[:, 2, 2].reshape(-1, 1)

        
#         g = np.hstack([
#         -mu * (F_11 * F_22 - F_12 * F_21) + F_00 * mu - lam * (F_11 * F_22 - F_12 * F_21) * (-F_00 * F_11 * F_22 + F_00 * F_12 * F_21 + F_01 * F_10 * F_22 - F_01 * F_12 * F_20 - F_02 * F_10 * F_21 + F_02 * F_11 * F_20 + 1.0),
#         mu * (F_01 * F_22 - F_02 * F_21) + F_10 * mu + lam * (F_01 * F_22 - F_02 * F_21) * (-F_00 * F_11 * F_22 + F_00 * F_12 * F_21 + F_01 * F_10 * F_22 - F_01 * F_12 * F_20 - F_02 * F_10 * F_21 + F_02 * F_11 * F_20 + 1.0),
#         -mu * (F_01 * F_12 - F_02 * F_11) + F_20 * mu - lam * (F_01 * F_12 - F_02 * F_11) * (-F_00 * F_11 * F_22 + F_00 * F_12 * F_21 + F_01 * F_10 * F_22 - F_01 * F_12 * F_20 - F_02 * F_10 * F_21 + F_02 * F_11 * F_20 + 1.0),
#         mu * (F_10 * F_22 - F_12 * F_20) + F_01 * mu + lam * (F_10 * F_22 - F_12 * F_20) * (-F_00 * F_11 * F_22 + F_00 * F_12 * F_21 + F_01 * F_10 * F_22 - F_01 * F_12 * F_20 - F_02 * F_10 * F_21 + F_02 * F_11 * F_20 + 1.0),
#         -mu * (F_00 * F_22 - F_02 * F_20) + F_11 * mu - lam * (F_00 * F_22 - F_02 * F_20) * (-F_00 * F_11 * F_22 + F_00 * F_12 * F_21 + F_01 * F_10 * F_22 - F_01 * F_12 * F_20 - F_02 * F_10 * F_21 + F_02 * F_11 * F_20 + 1.0),
#         mu * (F_00 * F_12 - F_02 * F_10) + F_21 * mu + lam * (F_00 * F_12 - F_02 * F_10) * (-F_00 * F_11 * F_22 + F_00 * F_12 * F_21 + F_01 * F_10 * F_22 - F_01 * F_12 * F_20 - F_02 * F_10 * F_21 + F_02 * F_11 * F_20 + 1.0),
#         -mu * (F_10 * F_21 - F_11 * F_20) + F_02 * mu - lam * (F_10 * F_21 - F_11 * F_20) * (-F_00 * F_11 * F_22 + F_00 * F_12 * F_21 + F_01 * F_10 * F_22 - F_01 * F_12 * F_20 - F_02 * F_10 * F_21 + F_02 * F_11 * F_20 + 1.0),
#         mu * (F_00 * F_21 - F_01 * F_20) + F_12 * mu + lam * (F_00 * F_21 - F_01 * F_20) * (-F_00 * F_11 * F_22 + F_00 * F_12 * F_21 + F_01 * F_10 * F_22 - F_01 * F_12 * F_20 - F_02 * F_10 * F_21 + F_02 * F_11 * F_20 + 1.0),
#         -mu * (F_00 * F_11 - F_01 * F_10) + F_22 * mu - lam * (F_00 * F_11 - F_01 * F_10) * (-F_00 * F_11 * F_22 + F_00 * F_12 * F_21 + F_01 * F_10 * F_22 - F_01 * F_12 * F_20 - F_02 * F_10 * F_21 + F_02 * F_11 * F_20 + 1.0)
#         ])[:, _9vector_3D_ordering_].reshape(-1, 3, 3)

#     PK1 = g * vol.reshape(-1, 1, 1)
#     return PK1

# def neo_hookean_filtered_gradient_dF(F, mu, lam, vol):
    
#     dim = F.shape[-1]
    
#     mu = np.array(mu).reshape(-1)
#     lam = np.array(lam).reshape(-1)
#     vol = np.array(vol).reshape(-1)
    
#     if dim == 2:
#         F_00 = F[:, 0, 0]
#         F_01 = F[:, 0, 1]
#         F_10 = F[:, 1, 0]
#         F_11 = F[:, 1, 1]
        

#         g = np.array(
#             [
#                 [
#                     1.0 * F_11 * lam * (F_00 * F_11 - F_01 * F_10 - 1.0) - F_11 * mu,
#                     -1.0 * F_10 * lam * (F_00 * F_11 - F_01 * F_10 - 1.0) + F_10 * mu,
#                 ],
#                 [
#                     -1.0 * F_01 * lam * (F_00 * F_11 - F_01 * F_10 - 1.0) + F_01 * mu,
#                     1.0 * F_00 * lam * (F_00 * F_11 - F_01 * F_10 - 1.0) - F_00 * mu,
#                 ],
#             ]
#         )
#     elif dim == 3:
#         raise ValueError("Not implemented")
        
#     PK1 = g.transpose(2, 0, 1) * vol.reshape(-1, 1, 1)
#     return PK1



# def neo_hookean_hessian_d2F(F, mu=1, lam=1, vol=1):

#     dim = F.shape[-1]

#     if dim == 2:
#         # convert cpp code to python below
#         # double mu = config_->mu;
#         # double la = config_->la;
#         # double F1_1 = F(0);
#         # double F2_1 = F(1);
#         # double F1_2 = F(2);
#         # double F2_2 = F(3);
#         # Matrix4d H;

    

#         F_00 = F[:, 0, 0].reshape(-1, 1)
#         F_01 = F[:, 0, 1].reshape(-1, 1)
#         F_10 = F[:, 1, 0].reshape(-1, 1)
#         F_11 = F[:, 1, 1].reshape(-1, 1)

#         # H(0,0) = mu+(F2_2*F2_2)*la;
#         # H(0,1) = -F1_2*F2_2*la;
#         # H(0,2) = -F2_1*F2_2*la;
#         # H(0,3) = -la-mu+F1_1*F2_2*la*2.0-F1_2*F2_1*la;
#         t0 = np.zeros( (F.shape[0],  4))
#         t0[:, [0]] = mu + (F_11**2) * lam
#         t0[:, [1]] = -F_01 * F_11 * lam
#         t0[:, [2]] = -F_10 * F_11 * lam
#         t0[:, [3]] = -lam - mu + F_00 * F_11 * lam * 2.0 - F_01 * F_10 * lam


#         # H(1,0) = -F1_2*F2_2*la;
#         # H(1,1) = mu+(F1_2*F1_2)*la;
#         # H(1,2) = la+mu-F1_1*F2_2*la+F1_2*F2_1*la*2.0;
#         # H(1,3) = -F1_1*F1_2*la;
#         t1 = np.zeros( (F.shape[0],  4))
#         t1[:, [0]] = -F_01 * F_11 * lam
#         t1[:, [1]] = mu + (F_01**2) * lam
#         t1[:, [2]] = lam + mu - F_00 * F_11 * lam + F_01 * F_10 * lam * 2.0
#         t1[:, [3]] = -F_00 * F_01 * lam


        
#         # H(2,0) = -F2_1*F2_2*la;
#         # H(2,1) = la+mu-F1_1*F2_2*la+F1_2*F2_1*la*2.0;
#         # H(2,2) = mu+(F2_1*F2_1)*la;
#         # H(2,3) = -F1_1*F2_1*la;
#         t2 = np.zeros( (F.shape[0],  4))
#         t2[:, [0]] = -F_10 * F_11 * lam
#         t2[:, [1]] = lam + mu - F_00 * F_11 * lam + F_01 * F_10 * lam * 2.0
#         t2[:, [2]] = mu + (F_10**2) * lam
#         t2[:, [3]] = -F_00 * F_10 * lam


#         # H(3,0) = -la-mu+F1_1*F2_2*la*2.0-F1_2*F2_1*la;
#         # H(3,1) = -F1_1*F1_2*la;
#         # H(3,2) = -F1_1*F2_1*la;
#         # H(3,3) = mu+(F1_1*F1_1)*la;
#         t3 = np.zeros( (F.shape[0],  4))
#         t3[:, [0]] = -lam - mu + F_00 * F_11 * lam * 2.0 - F_01 * F_10 * lam
#         t3[:, [1]] = -F_00 * F_01 * lam
#         t3[:, [2]] = -F_00 * F_10 * lam
#         t3[:, [3]] = mu + (F_00**2) * lam
        
#         H = np.hstack([
#             t0, t1, t2, t3
#             ])[:, _4x4matrix_2D_ordering_].reshape(-1, 4, 4)
#     elif dim == 3:
#         # convert cpp code to python below

        
#         # double mu = config_->mu;
#         # double la = config_->la;
#         # double F1_1 = F(0);
#         # double F2_1 = F(1);
#         # double F3_1 = F(2);
#         # double F1_2 = F(3);
#         # double F2_2 = F(4);
#         # double F3_2 = F(5);
#         # double F1_3 = F(6);
#         # double F2_3 = F(7);
#         # double F3_3 = F(8);
#         # Matrix9d H;

#         la = lam
      
#         F1_1 = F[:, 0, 0].reshape(-1, 1)
#         F1_2 = F[:, 0, 1].reshape(-1, 1)
#         F1_3 = F[:, 0, 2].reshape(-1, 1)
#         F2_1 = F[:, 1, 0].reshape(-1, 1)
#         F2_2 = F[:, 1, 1].reshape(-1, 1)
#         F2_3= F[:, 1, 2].reshape(-1, 1)
#         F3_1 = F[:, 2, 0].reshape(-1, 1)
#         F3_2 = F[:, 2, 1].reshape(-1, 1)
#         F3_3 = F[:, 2, 2].reshape(-1, 1)

#         # copy first row from cpp to python
#         # H(0,0) = mu+la*pow(F2_2*F3_3-F2_3*F3_2,2.0);
#         # H(0,1) = -la*(F1_2*F3_3-F1_3*F3_2)*(F2_2*F3_3-F2_3*F3_2);
#         # H(0,2) = la*(F1_2*F2_3-F1_3*F2_2)*(F2_2*F3_3-F2_3*F3_2);
#         # H(0,3) = -la*(F2_1*F3_3-F2_3*F3_1)*(F2_2*F3_3-F2_3*F3_2);
#         # H(0,4) = -F3_3*mu+la*(F1_1*F3_3-F1_3*F3_1)*(F2_2*F3_3-F2_3*F3_2)-F3_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         # H(0,5) = F2_3*mu-la*(F1_1*F2_3-F1_3*F2_1)*(F2_2*F3_3-F2_3*F3_2)+F2_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         # H(0,6) = la*(F2_1*F3_2-F2_2*F3_1)*(F2_2*F3_3-F2_3*F3_2);
#         # H(0,7) = F3_2*mu-la*(F1_1*F3_2-F1_2*F3_1)*(F2_2*F3_3-F2_3*F3_2)+F3_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         # H(0,8) = -F2_2*mu+la*(F1_1*F2_2-F1_2*F2_1)*(F2_2*F3_3-F2_3*F3_2)-F2_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t1 = np.zeros( (F.shape[0],  9))
#         t1[:,[0]] = mu+la*pow(F2_2*F3_3-F2_3*F3_2,2.0);
#         t1[:,[1]] = -la*(F1_2*F3_3-F1_3*F3_2)*(F2_2*F3_3-F2_3*F3_2);
#         t1[:,[2]] = la*(F1_2*F2_3-F1_3*F2_2)*(F2_2*F3_3-F2_3*F3_2);
#         t1[:,[3]] = -la*(F2_1*F3_3-F2_3*F3_1)*(F2_2*F3_3-F2_3*F3_2);
#         t1[:,[4]] = -F3_3*mu+la*(F1_1*F3_3-F1_3*F3_1)*(F2_2*F3_3-F2_3*F3_2)-F3_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t1[:,[5]] = F2_3*mu-la*(F1_1*F2_3-F1_3*F2_1)*(F2_2*F3_3-F2_3*F3_2)+F2_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t1[:,[6]] = la*(F2_1*F3_2-F2_2*F3_1)*(F2_2*F3_3-F2_3*F3_2);
#         t1[:,[7]] = F3_2*mu-la*(F1_1*F3_2-F1_2*F3_1)*(F2_2*F3_3-F2_3*F3_2)+F3_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t1[:,[8]] = -F2_2*mu+la*(F1_1*F2_2-F1_2*F2_1)*(F2_2*F3_3-F2_3*F3_2)-F2_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
        

    
#         # H(1,0) = -la*(F1_2*F3_3-F1_3*F3_2)*(F2_2*F3_3-F2_3*F3_2);
#         # H(1,1) = mu+la*pow(F1_2*F3_3-F1_3*F3_2,2.0);
#         # H(1,2) = -la*(F1_2*F2_3-F1_3*F2_2)*(F1_2*F3_3-F1_3*F3_2);
#         # H(1,3) = F3_3*mu+la*(F1_2*F3_3-F1_3*F3_2)*(F2_1*F3_3-F2_3*F3_1)+F3_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         # H(1,4) = -la*(F1_1*F3_3-F1_3*F3_1)*(F1_2*F3_3-F1_3*F3_2);
#         # H(1,5) = -F1_3*mu+la*(F1_1*F2_3-F1_3*F2_1)*(F1_2*F3_3-F1_3*F3_2)-F1_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         # H(1,6) = -F3_2*mu-la*(F1_2*F3_3-F1_3*F3_2)*(F2_1*F3_2-F2_2*F3_1)-F3_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         # H(1,7) = la*(F1_1*F3_2-F1_2*F3_1)*(F1_2*F3_3-F1_3*F3_2);
#         # H(1,8) = F1_2*mu-la*(F1_1*F2_2-F1_2*F2_1)*(F1_2*F3_3-F1_3*F3_2)+F1_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t2 = np.zeros( (F.shape[0],  9))
#         t2[:, [0]] = -la*(F1_2*F3_3-F1_3*F3_2)*(F2_2*F3_3-F2_3*F3_2);
#         t2[:, [1]] = mu+la*pow(F1_2*F3_3-F1_3*F3_2,2.0);
#         t2[:, [2]] = -la*(F1_2*F2_3-F1_3*F2_2)*(F1_2*F3_3-F1_3*F3_2);
#         t2[:, [3]] = F3_3*mu+la*(F1_2*F3_3-F1_3*F3_2)*(F2_1*F3_3-F2_3*F3_1)+F3_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t2[:, [4]] = -la*(F1_1*F3_3-F1_3*F3_1)*(F1_2*F3_3-F1_3*F3_2);
#         t2[:, [5]] = -F1_3*mu+la*(F1_1*F2_3-F1_3*F2_1)*(F1_2*F3_3-F1_3*F3_2)-F1_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t2[:, [6]] = -F3_2*mu-la*(F1_2*F3_3-F1_3*F3_2)*(F2_1*F3_2-F2_2*F3_1)-F3_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t2[:, [7]] = la*(F1_1*F3_2-F1_2*F3_1)*(F1_2*F3_3-F1_3*F3_2);
#         t2[:, [8]] = F1_2*mu-la*(F1_1*F2_2-F1_2*F2_1)*(F1_2*F3_3-F1_3*F3_2)+F1_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
    
#         # H(2,0) = la*(F1_2*F2_3-F1_3*F2_2)*(F2_2*F3_3-F2_3*F3_2);
#         # H(2,1) = -la*(F1_2*F2_3-F1_3*F2_2)*(F1_2*F3_3-F1_3*F3_2);
#         # H(2,2) = mu+la*pow(F1_2*F2_3-F1_3*F2_2,2.0);
#         # H(2,3) = -F2_3*mu-la*(F1_2*F2_3-F1_3*F2_2)*(F2_1*F3_3-F2_3*F3_1)-F2_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         # H(2,4) = F1_3*mu+la*(F1_2*F2_3-F1_3*F2_2)*(F1_1*F3_3-F1_3*F3_1)+F1_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         # H(2,5) = -la*(F1_1*F2_3-F1_3*F2_1)*(F1_2*F2_3-F1_3*F2_2);
#         # H(2,6) = F2_2*mu+la*(F1_2*F2_3-F1_3*F2_2)*(F2_1*F3_2-F2_2*F3_1)+F2_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         # H(2,7) = -F1_2*mu-la*(F1_2*F2_3-F1_3*F2_2)*(F1_1*F3_2-F1_2*F3_1)-F1_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         # H(2,8) = la*(F1_1*F2_2-F1_2*F2_1)*(F1_2*F2_3-F1_3*F2_2);
#         t3 = np.zeros( (F.shape[0],  9))
#         t3[:,[0]] = la*(F1_2*F2_3-F1_3*F2_2)*(F2_2*F3_3-F2_3*F3_2);
#         t3[:,[1]] = -la*(F1_2*F2_3-F1_3*F2_2)*(F1_2*F3_3-F1_3*F3_2);
#         t3[:,[2]] = mu+la*pow(F1_2*F2_3-F1_3*F2_2,2.0);
#         t3[:,[3]] = -F2_3*mu-la*(F1_2*F2_3-F1_3*F2_2)*(F2_1*F3_3-F2_3*F3_1)-F2_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t3[:,[4]] = F1_3*mu+la*(F1_2*F2_3-F1_3*F2_2)*(F1_1*F3_3-F1_3*F3_1)+F1_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t3[:,[5]] = -la*(F1_1*F2_3-F1_3*F2_1)*(F1_2*F2_3-F1_3*F2_2);
#         t3[:,[6]] = F2_2*mu+la*(F1_2*F2_3-F1_3*F2_2)*(F2_1*F3_2-F2_2*F3_1)+F2_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t3[:,[7]] = -F1_2*mu-la*(F1_2*F2_3-F1_3*F2_2)*(F1_1*F3_2-F1_2*F3_1)-F1_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t3[:,[8]] = la*(F1_1*F2_2-F1_2*F2_1)*(F1_2*F2_3-F1_3*F2_2);
        
#         # H(3,0) = -la*(F2_1*F3_3-F2_3*F3_1)*(F2_2*F3_3-F2_3*F3_2);
#         # H(3,1) = F3_3*mu+la*(F1_2*F3_3-F1_3*F3_2)*(F2_1*F3_3-F2_3*F3_1)+F3_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         # H(3,2) = -F2_3*mu-la*(F1_2*F2_3-F1_3*F2_2)*(F2_1*F3_3-F2_3*F3_1)-F2_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         # H(3,3) = mu+la*pow(F2_1*F3_3-F2_3*F3_1,2.0);
#         # H(3,4) = -la*(F1_1*F3_3-F1_3*F3_1)*(F2_1*F3_3-F2_3*F3_1);
#         # H(3,5) = la*(F1_1*F2_3-F1_3*F2_1)*(F2_1*F3_3-F2_3*F3_1);
#         # H(3,6) = -la*(F2_1*F3_2-F2_2*F3_1)*(F2_1*F3_3-F2_3*F3_1);
#         # H(3,7) = -F3_1*mu+la*(F1_1*F3_2-F1_2*F3_1)*(F2_1*F3_3-F2_3*F3_1)-F3_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         # H(3,8) = F2_1*mu-la*(F1_1*F2_2-F1_2*F2_1)*(F2_1*F3_3-F2_3*F3_1)+F2_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t4 = np.zeros( (F.shape[0],  9))
#         t4[:, [0]] = -la*(F2_1*F3_3-F2_3*F3_1)*(F2_2*F3_3-F2_3*F3_2);
#         t4[:, [1]] = F3_3*mu+la*(F1_2*F3_3-F1_3*F3_2)*(F2_1*F3_3-F2_3*F3_1)+F3_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t4[:, [2]] = -F2_3*mu-la*(F1_2*F2_3-F1_3*F2_2)*(F2_1*F3_3-F2_3*F3_1)-F2_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t4[:, [3]] = mu+la*pow(F2_1*F3_3-F2_3*F3_1,2.0);
#         t4[:, [4]] = -la*(F1_1*F3_3-F1_3*F3_1)*(F2_1*F3_3-F2_3*F3_1);
#         t4[:, [5]] = la*(F1_1*F2_3-F1_3*F2_1)*(F2_1*F3_3-F2_3*F3_1);
#         t4[:, [6]] = -la*(F2_1*F3_2-F2_2*F3_1)*(F2_1*F3_3-F2_3*F3_1);
#         t4[:, [7]] = -F3_1*mu+la*(F1_1*F3_2-F1_2*F3_1)*(F2_1*F3_3-F2_3*F3_1)-F3_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t4[:, [8]] = F2_1*mu-la*(F1_1*F2_2-F1_2*F2_1)*(F2_1*F3_3-F2_3*F3_1)+F2_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
        
        
#         # H(4,0) = -F3_3*mu+la*(F1_1*F3_3-F1_3*F3_1)*(F2_2*F3_3-F2_3*F3_2)-F3_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         # H(4,1) = -la*(F1_1*F3_3-F1_3*F3_1)*(F1_2*F3_3-F1_3*F3_2);
#         # H(4,2) = F1_3*mu+la*(F1_2*F2_3-F1_3*F2_2)*(F1_1*F3_3-F1_3*F3_1)+F1_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         # H(4,3) = -la*(F1_1*F3_3-F1_3*F3_1)*(F2_1*F3_3-F2_3*F3_1);
#         # H(4,4) = mu+la*pow(F1_1*F3_3-F1_3*F3_1,2.0);
#         # H(4,5) = -la*(F1_1*F2_3-F1_3*F2_1)*(F1_1*F3_3-F1_3*F3_1);
#         # H(4,6) = F3_1*mu+la*(F1_1*F3_3-F1_3*F3_1)*(F2_1*F3_2-F2_2*F3_1)+F3_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         # H(4,7) = -la*(F1_1*F3_2-F1_2*F3_1)*(F1_1*F3_3-F1_3*F3_1);
#         # H(4,8) = -F1_1*mu+la*(F1_1*F2_2-F1_2*F2_1)*(F1_1*F3_3-F1_3*F3_1)-F1_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t5 = np.zeros( (F.shape[0],  9))
#         t5[:,[0]] = -F3_3*mu+la*(F1_1*F3_3-F1_3*F3_1)*(F2_2*F3_3-F2_3*F3_2)-F3_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t5[:,[1]] = -la*(F1_1*F3_3-F1_3*F3_1)*(F1_2*F3_3-F1_3*F3_2);
#         t5[:,[2]] = F1_3*mu+la*(F1_2*F2_3-F1_3*F2_2)*(F1_1*F3_3-F1_3*F3_1)+F1_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t5[:,[3]] = -la*(F1_1*F3_3-F1_3*F3_1)*(F2_1*F3_3-F2_3*F3_1);
#         t5[:,[4]] = mu+la*pow(F1_1*F3_3-F1_3*F3_1,2.0);
#         t5[:,[5]] = -la*(F1_1*F2_3-F1_3*F2_1)*(F1_1*F3_3-F1_3*F3_1);
#         t5[:,[6]] = F3_1*mu+la*(F1_1*F3_3-F1_3*F3_1)*(F2_1*F3_2-F2_2*F3_1)+F3_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t5[:,[7]] = -la*(F1_1*F3_2-F1_2*F3_1)*(F1_1*F3_3-F1_3*F3_1);
#         t5[:,[8]] = -F1_1*mu+la*(F1_1*F2_2-F1_2*F2_1)*(F1_1*F3_3-F1_3*F3_1)-F1_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
        
        
        
#         # H(5,0) = F2_3*mu-la*(F1_1*F2_3-F1_3*F2_1)*(F2_2*F3_3-F2_3*F3_2)+F2_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         # H(5,1) = -F1_3*mu+la*(F1_1*F2_3-F1_3*F2_1)*(F1_2*F3_3-F1_3*F3_2)-F1_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         # H(5,2) = -la*(F1_1*F2_3-F1_3*F2_1)*(F1_2*F2_3-F1_3*F2_2);
#         # H(5,3) = la*(F1_1*F2_3-F1_3*F2_1)*(F2_1*F3_3-F2_3*F3_1);
#         # H(5,4) = -la*(F1_1*F2_3-F1_3*F2_1)*(F1_1*F3_3-F1_3*F3_1);
#         # H(5,5) = mu+la*pow(F1_1*F2_3-F1_3*F2_1,2.0);
#         # H(5,6) = -F2_1*mu-la*(F1_1*F2_3-F1_3*F2_1)*(F2_1*F3_2-F2_2*F3_1)-F2_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         # H(5,7) = F1_1*mu+la*(F1_1*F2_3-F1_3*F2_1)*(F1_1*F3_2-F1_2*F3_1)+F1_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         # H(5,8) = -la*(F1_1*F2_2-F1_2*F2_1)*(F1_1*F2_3-F1_3*F2_1);
#         t6 = np.zeros( (F.shape[0],  9))
#         t6[:, [0]] = F2_3*mu-la*(F1_1*F2_3-F1_3*F2_1)*(F2_2*F3_3-F2_3*F3_2)+F2_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t6[:, [1]] = -F1_3*mu+la*(F1_1*F2_3-F1_3*F2_1)*(F1_2*F3_3-F1_3*F3_2)-F1_3*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t6[:, [2]] = -la*(F1_1*F2_3-F1_3*F2_1)*(F1_2*F2_3-F1_3*F2_2);
#         t6[:, [3]] = la*(F1_1*F2_3-F1_3*F2_1)*(F2_1*F3_3-F2_3*F3_1);
#         t6[:, [4]] = -la*(F1_1*F2_3-F1_3*F2_1)*(F1_1*F3_3-F1_3*F3_1);
#         t6[:, [5]] = mu+la*pow(F1_1*F2_3-F1_3*F2_1,2.0);
#         t6[:, [6]] = -F2_1*mu-la*(F1_1*F2_3-F1_3*F2_1)*(F2_1*F3_2-F2_2*F3_1)-F2_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t6[:, [7]] = F1_1*mu+la*(F1_1*F2_3-F1_3*F2_1)*(F1_1*F3_2-F1_2*F3_1)+F1_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t6[:, [8]] = -la*(F1_1*F2_2-F1_2*F2_1)*(F1_1*F2_3-F1_3*F2_1);
        
        
        
#         # H(6,0) = la*(F2_1*F3_2-F2_2*F3_1)*(F2_2*F3_3-F2_3*F3_2);
#         # H(6,1) = -F3_2*mu-la*(F1_2*F3_3-F1_3*F3_2)*(F2_1*F3_2-F2_2*F3_1)-F3_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         # H(6,2) = F2_2*mu+la*(F1_2*F2_3-F1_3*F2_2)*(F2_1*F3_2-F2_2*F3_1)+F2_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         # H(6,3) = -la*(F2_1*F3_2-F2_2*F3_1)*(F2_1*F3_3-F2_3*F3_1);
#         # H(6,4) = F3_1*mu+la*(F1_1*F3_3-F1_3*F3_1)*(F2_1*F3_2-F2_2*F3_1)+F3_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         # H(6,5) = -F2_1*mu-la*(F1_1*F2_3-F1_3*F2_1)*(F2_1*F3_2-F2_2*F3_1)-F2_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         # H(6,6) = mu+la*pow(F2_1*F3_2-F2_2*F3_1,2.0);
#         # H(6,7) = -la*(F1_1*F3_2-F1_2*F3_1)*(F2_1*F3_2-F2_2*F3_1);
#         # H(6,8) = la*(F1_1*F2_2-F1_2*F2_1)*(F2_1*F3_2-F2_2*F3_1);
#         t7 = np.zeros( (F.shape[0],  9))
#         t7[:, [0]] = la*(F2_1*F3_2-F2_2*F3_1)*(F2_2*F3_3-F2_3*F3_2);
#         t7[:, [1]] = -F3_2*mu-la*(F1_2*F3_3-F1_3*F3_2)*(F2_1*F3_2-F2_2*F3_1)-F3_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t7[:, [2]] = F2_2*mu+la*(F1_2*F2_3-F1_3*F2_2)*(F2_1*F3_2-F2_2*F3_1)+F2_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t7[:, [3]] = -la*(F2_1*F3_2-F2_2*F3_1)*(F2_1*F3_3-F2_3*F3_1);
#         t7[:, [4]] = F3_1*mu+la*(F1_1*F3_3-F1_3*F3_1)*(F2_1*F3_2-F2_2*F3_1)+F3_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t7[:, [5]] = -F2_1*mu-la*(F1_1*F2_3-F1_3*F2_1)*(F2_1*F3_2-F2_2*F3_1)-F2_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t7[:, [6]] = mu+la*pow(F2_1*F3_2-F2_2*F3_1,2.0);
#         t7[:, [7]] = -la*(F1_1*F3_2-F1_2*F3_1)*(F2_1*F3_2-F2_2*F3_1);
#         t7[:, [8]] = la*(F1_1*F2_2-F1_2*F2_1)*(F2_1*F3_2-F2_2*F3_1);
        
#         # H(7,0) = F3_2*mu-la*(F1_1*F3_2-F1_2*F3_1)*(F2_2*F3_3-F2_3*F3_2)+F3_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         # H(7,1) = la*(F1_1*F3_2-F1_2*F3_1)*(F1_2*F3_3-F1_3*F3_2);
#         # H(7,2) = -F1_2*mu-la*(F1_2*F2_3-F1_3*F2_2)*(F1_1*F3_2-F1_2*F3_1)-F1_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         # H(7,3) = -F3_1*mu+la*(F1_1*F3_2-F1_2*F3_1)*(F2_1*F3_3-F2_3*F3_1)-F3_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         # H(7,4) = -la*(F1_1*F3_2-F1_2*F3_1)*(F1_1*F3_3-F1_3*F3_1);
#         # H(7,5) = F1_1*mu+la*(F1_1*F2_3-F1_3*F2_1)*(F1_1*F3_2-F1_2*F3_1)+F1_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         # H(7,6) = -la*(F1_1*F3_2-F1_2*F3_1)*(F2_1*F3_2-F2_2*F3_1);
#         # H(7,7) = mu+la*pow(F1_1*F3_2-F1_2*F3_1,2.0);
#         # H(7,8) = -la*(F1_1*F2_2-F1_2*F2_1)*(F1_1*F3_2-F1_2*F3_1);
        
#         t8 = np.zeros( (F.shape[0],  9))
#         t8[:, [0]] = F3_2*mu-la*(F1_1*F3_2-F1_2*F3_1)*(F2_2*F3_3-F2_3*F3_2)+F3_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t8[:, [1]] = la*(F1_1*F3_2-F1_2*F3_1)*(F1_2*F3_3-F1_3*F3_2);
#         t8[:, [2]] = -F1_2*mu-la*(F1_2*F2_3-F1_3*F2_2)*(F1_1*F3_2-F1_2*F3_1)-F1_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t8[:, [3]] = -F3_1*mu+la*(F1_1*F3_2-F1_2*F3_1)*(F2_1*F3_3-F2_3*F3_1)-F3_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t8[:, [4]] = -la*(F1_1*F3_2-F1_2*F3_1)*(F1_1*F3_3-F1_3*F3_1);
#         t8[:, [5]] = F1_1*mu+la*(F1_1*F2_3-F1_3*F2_1)*(F1_1*F3_2-F1_2*F3_1)+F1_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t8[:, [6]] = -la*(F1_1*F3_2-F1_2*F3_1)*(F2_1*F3_2-F2_2*F3_1);
#         t8[:, [7]] = mu+la*pow(F1_1*F3_2-F1_2*F3_1,2.0);
#         t8[:, [8]] = -la*(F1_1*F2_2-F1_2*F2_1)*(F1_1*F3_2-F1_2*F3_1);
        
#         # H(8,0) = -F2_2*mu+la*(F1_1*F2_2-F1_2*F2_1)*(F2_2*F3_3-F2_3*F3_2)-F2_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         # H(8,1) = F1_2*mu-la*(F1_1*F2_2-F1_2*F2_1)*(F1_2*F3_3-F1_3*F3_2)+F1_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         # H(8,2) = la*(F1_1*F2_2-F1_2*F2_1)*(F1_2*F2_3-F1_3*F2_2);
#         # H(8,3) = F2_1*mu-la*(F1_1*F2_2-F1_2*F2_1)*(F2_1*F3_3-F2_3*F3_1)+F2_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         # H(8,4) = -F1_1*mu+la*(F1_1*F2_2-F1_2*F2_1)*(F1_1*F3_3-F1_3*F3_1)-F1_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         # H(8,5) = -la*(F1_1*F2_2-F1_2*F2_1)*(F1_1*F2_3-F1_3*F2_1);
#         # H(8,6) = la*(F1_1*F2_2-F1_2*F2_1)*(F2_1*F3_2-F2_2*F3_1);
#         # H(8,7) = -la*(F1_1*F2_2-F1_2*F2_1)*(F1_1*F3_2-F1_2*F3_1);
#         # H(8,8) = mu+la*pow(F1_1*F2_2-F1_2*F2_1,2.0);

#         t9 = np.zeros( (F.shape[0],  9))
#         t9[:, [0]] = -F2_2*mu+la*(F1_1*F2_2-F1_2*F2_1)*(F2_2*F3_3-F2_3*F3_2)-F2_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t9[:, [1]] = F1_2*mu-la*(F1_1*F2_2-F1_2*F2_1)*(F1_2*F3_3-F1_3*F3_2)+F1_2*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t9[:, [2]] = la*(F1_1*F2_2-F1_2*F2_1)*(F1_2*F2_3-F1_3*F2_2);
#         t9[:, [3]] = F2_1*mu-la*(F1_1*F2_2-F1_2*F2_1)*(F2_1*F3_3-F2_3*F3_1)+F2_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t9[:, [4]] = -F1_1*mu+la*(F1_1*F2_2-F1_2*F2_1)*(F1_1*F3_3-F1_3*F3_1)-F1_1*la*(-F1_1*F2_2*F3_3+F1_1*F2_3*F3_2+F1_2*F2_1*F3_3-F1_2*F2_3*F3_1-F1_3*F2_1*F3_2+F1_3*F2_2*F3_1+1.0);
#         t9[:, [5]] = -la*(F1_1*F2_2-F1_2*F2_1)*(F1_1*F2_3-F1_3*F2_1);
#         t9[:, [6]] = la*(F1_1*F2_2-F1_2*F2_1)*(F2_1*F3_2-F2_2*F3_1);
#         t9[:, [7]] = -la*(F1_1*F2_2-F1_2*F2_1)*(F1_1*F3_2-F1_2*F3_1);
#         t9[:, [8]] = mu+la*pow(F1_1*F2_2-F1_2*F2_1,2.0);


#         H = np.hstack([t1, t2, t3, t4, t5, t6, t7, t8, t9])[:, _9x9matrix_3D_ordering_].reshape(-1, 9, 9)

    
#     H = H * vol.reshape(-1, 1, 1)

#     return H



"""Neo-Hookean elastic energy.

Follows the standardized three-tier layout (see :mod:`simkit.energies.arap`
for the reference). Neo-Hookean has only the deformation gradient (``F``)
representation, so there is no ``_S`` tier:

Element tier (``*_element_F``)
    Per-element density and derivative blocks. Material parameters ``mu`` and
    ``lam`` only: no quadrature weight ``vol``, no summation, no operator.

Global explicit tier (``*_x``)
    Takes a prebuilt deformation Jacobian ``J`` and weights ``vol``, calls the
    element tier, weights, and assembles.

Self-contained tier (no suffix)
    Builds ``J`` and ``vol`` from rest geometry ``(X, T)``.

Notes
-----
Closed-form gradient and Hessian transcribed from generated C++. The
``mat2py`` ordering helpers reindex from the column-major (MATLAB/C++) flat
layout to the row-major ``F`` layout used throughout the library.
"""

from typing import Optional

import numpy as np
import scipy as sp

from ..mat2py import (
    _4vector_2D_ordering_,
    _9vector_3D_ordering_,
    _4x4matrix_2D_ordering_,
    _9x9matrix_3D_ordering_,
)
from ..deformation_jacobian import deformation_jacobian
from ..volume import volume
from ..psd_project import psd_project


# --------------------------------------------------------------------------- #
# Element tier: deformation gradient (F) representation                       #
# --------------------------------------------------------------------------- #
def neo_hookean_energy_element_F(F: np.ndarray, mu: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Per-element Neo-Hookean energy density.

    Parameters
    ----------
    F : np.ndarray (t, dim, dim)
        Per-element deformation gradients.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.

    Returns
    -------
    psi : np.ndarray (t, 1)
        Per-element energy densities. No quadrature weighting applied.
    """
    dim = F.shape[-1]
    F = F.reshape(-1, dim, dim)
    mu = np.asarray(mu).reshape(-1, 1)
    lam = np.asarray(lam).reshape(-1, 1)

    if dim == 2:
        F_00 = F[:, 0, 0].reshape(-1, 1)
        F_01 = F[:, 0, 1].reshape(-1, 1)
        F_10 = F[:, 1, 0].reshape(-1, 1)
        F_11 = F[:, 1, 1].reshape(-1, 1)
        psi = (
            mu * (-F_00 * F_11 + F_01 * F_10 + 1.0)
            + (lam * (-(F_00 * F_11) + F_01 * F_10 + 1.0) ** 2) / 2.0
            + (mu * (F_00 ** 2 + F_01 ** 2 + F_10 ** 2 + F_11 ** 2 - 2.0)) / 2.0
        )
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
        det = (
            -F_00 * F_11 * F_22 + F_00 * F_12 * F_21 + F_01 * F_10 * F_22
            - F_01 * F_12 * F_20 - F_02 * F_10 * F_21 + F_02 * F_11 * F_20 + 1.0
        )
        psi = (
            mu * det
            + (lam * det ** 2) / 2.0
            + (mu * (
                F_00 ** 2 + F_01 ** 2 + F_02 ** 2
                + F_10 ** 2 + F_11 ** 2 + F_12 ** 2
                + F_20 ** 2 + F_21 ** 2 + F_22 ** 2 - 3.0)) / 2.0
        )
    else:
        raise ValueError("Neo-Hookean supports dim=2 or dim=3")
    return psi


def neo_hookean_gradient_element_F(F: np.ndarray, mu: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Per-element first Piola-Kirchhoff stress (gradient w.r.t. ``F``).

    Parameters
    ----------
    F : np.ndarray (t, dim, dim)
        Per-element deformation gradients.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.

    Returns
    -------
    P : np.ndarray (t, dim, dim)
        Per-element PK1 stress blocks. No quadrature weighting applied.
    """
    dim = F.shape[-1]
    mu = np.asarray(mu).reshape(-1, 1)
    lam = np.asarray(lam).reshape(-1, 1)

    if dim == 2:
        F_00 = F[:, 0, 0].reshape(-1, 1)
        F_01 = F[:, 0, 1].reshape(-1, 1)
        F_10 = F[:, 1, 0].reshape(-1, 1)
        F_11 = F[:, 1, 1].reshape(-1, 1)
        det = -F_00 * F_11 + F_01 * F_10 + 1.0
        P = np.hstack([
            F_00 * mu - F_11 * mu - F_11 * lam * det,
            F_01 * mu + F_10 * mu + F_01 * lam * det,
            F_01 * mu + F_10 * mu + F_10 * lam * det,
            -F_00 * mu + F_11 * mu - F_00 * lam * det,
        ])[:, _4vector_2D_ordering_].reshape(-1, 2, 2)
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
        det = (
            -F_00 * F_11 * F_22 + F_00 * F_12 * F_21 + F_01 * F_10 * F_22
            - F_01 * F_12 * F_20 - F_02 * F_10 * F_21 + F_02 * F_11 * F_20 + 1.0
        )
        P = np.hstack([
            -mu * (F_11 * F_22 - F_12 * F_21) + F_00 * mu - lam * (F_11 * F_22 - F_12 * F_21) * det,
            mu * (F_01 * F_22 - F_02 * F_21) + F_10 * mu + lam * (F_01 * F_22 - F_02 * F_21) * det,
            -mu * (F_01 * F_12 - F_02 * F_11) + F_20 * mu - lam * (F_01 * F_12 - F_02 * F_11) * det,
            mu * (F_10 * F_22 - F_12 * F_20) + F_01 * mu + lam * (F_10 * F_22 - F_12 * F_20) * det,
            -mu * (F_00 * F_22 - F_02 * F_20) + F_11 * mu - lam * (F_00 * F_22 - F_02 * F_20) * det,
            mu * (F_00 * F_12 - F_02 * F_10) + F_21 * mu + lam * (F_00 * F_12 - F_02 * F_10) * det,
            -mu * (F_10 * F_21 - F_11 * F_20) + F_02 * mu - lam * (F_10 * F_21 - F_11 * F_20) * det,
            mu * (F_00 * F_21 - F_01 * F_20) + F_12 * mu + lam * (F_00 * F_21 - F_01 * F_20) * det,
            -mu * (F_00 * F_11 - F_01 * F_10) + F_22 * mu - lam * (F_00 * F_11 - F_01 * F_10) * det,
        ])[:, _9vector_3D_ordering_].reshape(-1, 3, 3)
    else:
        raise ValueError("Neo-Hookean supports dim=2 or dim=3")
    return P


def neo_hookean_hessian_element_F(F: np.ndarray, mu: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Per-element Hessian of the density w.r.t. ``F`` (vectorized blocks).

    Parameters
    ----------
    F : np.ndarray (t, dim, dim)
        Per-element deformation gradients.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.

    Returns
    -------
    H : np.ndarray (t, dim*dim, dim*dim)
        Per-element Hessian blocks in vectorized ``F`` layout. No quadrature
        weighting applied. Not PSD-projected; projection happens in the global
        tier.
    """
    dim = F.shape[-1]
    mu = np.asarray(mu).reshape(-1, 1)
    lam = np.asarray(lam).reshape(-1, 1)

    if dim == 2:
        F_00 = F[:, 0, 0].reshape(-1, 1)
        F_01 = F[:, 0, 1].reshape(-1, 1)
        F_10 = F[:, 1, 0].reshape(-1, 1)
        F_11 = F[:, 1, 1].reshape(-1, 1)

        t0 = np.zeros((F.shape[0], 4))
        t0[:, [0]] = mu + (F_11 ** 2) * lam
        t0[:, [1]] = -F_01 * F_11 * lam
        t0[:, [2]] = -F_10 * F_11 * lam
        t0[:, [3]] = -lam - mu + F_00 * F_11 * lam * 2.0 - F_01 * F_10 * lam

        t1 = np.zeros((F.shape[0], 4))
        t1[:, [0]] = -F_01 * F_11 * lam
        t1[:, [1]] = mu + (F_01 ** 2) * lam
        t1[:, [2]] = lam + mu - F_00 * F_11 * lam + F_01 * F_10 * lam * 2.0
        t1[:, [3]] = -F_00 * F_01 * lam

        t2 = np.zeros((F.shape[0], 4))
        t2[:, [0]] = -F_10 * F_11 * lam
        t2[:, [1]] = lam + mu - F_00 * F_11 * lam + F_01 * F_10 * lam * 2.0
        t2[:, [2]] = mu + (F_10 ** 2) * lam
        t2[:, [3]] = -F_00 * F_10 * lam

        t3 = np.zeros((F.shape[0], 4))
        t3[:, [0]] = -lam - mu + F_00 * F_11 * lam * 2.0 - F_01 * F_10 * lam
        t3[:, [1]] = -F_00 * F_01 * lam
        t3[:, [2]] = -F_00 * F_10 * lam
        t3[:, [3]] = mu + (F_00 ** 2) * lam

        H = np.hstack([t0, t1, t2, t3])[:, _4x4matrix_2D_ordering_].reshape(-1, 4, 4)
    elif dim == 3:
        la = lam
        F1_1 = F[:, 0, 0].reshape(-1, 1)
        F1_2 = F[:, 0, 1].reshape(-1, 1)
        F1_3 = F[:, 0, 2].reshape(-1, 1)
        F2_1 = F[:, 1, 0].reshape(-1, 1)
        F2_2 = F[:, 1, 1].reshape(-1, 1)
        F2_3 = F[:, 1, 2].reshape(-1, 1)
        F3_1 = F[:, 2, 0].reshape(-1, 1)
        F3_2 = F[:, 2, 1].reshape(-1, 1)
        F3_3 = F[:, 2, 2].reshape(-1, 1)

        det = (
            -F1_1 * F2_2 * F3_3 + F1_1 * F2_3 * F3_2 + F1_2 * F2_1 * F3_3
            - F1_2 * F2_3 * F3_1 - F1_3 * F2_1 * F3_2 + F1_3 * F2_2 * F3_1 + 1.0
        )

        t1 = np.zeros((F.shape[0], 9))
        t1[:, [0]] = mu + la * (F2_2 * F3_3 - F2_3 * F3_2) ** 2.0
        t1[:, [1]] = -la * (F1_2 * F3_3 - F1_3 * F3_2) * (F2_2 * F3_3 - F2_3 * F3_2)
        t1[:, [2]] = la * (F1_2 * F2_3 - F1_3 * F2_2) * (F2_2 * F3_3 - F2_3 * F3_2)
        t1[:, [3]] = -la * (F2_1 * F3_3 - F2_3 * F3_1) * (F2_2 * F3_3 - F2_3 * F3_2)
        t1[:, [4]] = -F3_3 * mu + la * (F1_1 * F3_3 - F1_3 * F3_1) * (F2_2 * F3_3 - F2_3 * F3_2) - F3_3 * la * det
        t1[:, [5]] = F2_3 * mu - la * (F1_1 * F2_3 - F1_3 * F2_1) * (F2_2 * F3_3 - F2_3 * F3_2) + F2_3 * la * det
        t1[:, [6]] = la * (F2_1 * F3_2 - F2_2 * F3_1) * (F2_2 * F3_3 - F2_3 * F3_2)
        t1[:, [7]] = F3_2 * mu - la * (F1_1 * F3_2 - F1_2 * F3_1) * (F2_2 * F3_3 - F2_3 * F3_2) + F3_2 * la * det
        t1[:, [8]] = -F2_2 * mu + la * (F1_1 * F2_2 - F1_2 * F2_1) * (F2_2 * F3_3 - F2_3 * F3_2) - F2_2 * la * det

        t2 = np.zeros((F.shape[0], 9))
        t2[:, [0]] = -la * (F1_2 * F3_3 - F1_3 * F3_2) * (F2_2 * F3_3 - F2_3 * F3_2)
        t2[:, [1]] = mu + la * (F1_2 * F3_3 - F1_3 * F3_2) ** 2.0
        t2[:, [2]] = -la * (F1_2 * F2_3 - F1_3 * F2_2) * (F1_2 * F3_3 - F1_3 * F3_2)
        t2[:, [3]] = F3_3 * mu + la * (F1_2 * F3_3 - F1_3 * F3_2) * (F2_1 * F3_3 - F2_3 * F3_1) + F3_3 * la * det
        t2[:, [4]] = -la * (F1_1 * F3_3 - F1_3 * F3_1) * (F1_2 * F3_3 - F1_3 * F3_2)
        t2[:, [5]] = -F1_3 * mu + la * (F1_1 * F2_3 - F1_3 * F2_1) * (F1_2 * F3_3 - F1_3 * F3_2) - F1_3 * la * det
        t2[:, [6]] = -F3_2 * mu - la * (F1_2 * F3_3 - F1_3 * F3_2) * (F2_1 * F3_2 - F2_2 * F3_1) - F3_2 * la * det
        t2[:, [7]] = la * (F1_1 * F3_2 - F1_2 * F3_1) * (F1_2 * F3_3 - F1_3 * F3_2)
        t2[:, [8]] = F1_2 * mu - la * (F1_1 * F2_2 - F1_2 * F2_1) * (F1_2 * F3_3 - F1_3 * F3_2) + F1_2 * la * det

        t3 = np.zeros((F.shape[0], 9))
        t3[:, [0]] = la * (F1_2 * F2_3 - F1_3 * F2_2) * (F2_2 * F3_3 - F2_3 * F3_2)
        t3[:, [1]] = -la * (F1_2 * F2_3 - F1_3 * F2_2) * (F1_2 * F3_3 - F1_3 * F3_2)
        t3[:, [2]] = mu + la * (F1_2 * F2_3 - F1_3 * F2_2) ** 2.0
        t3[:, [3]] = -F2_3 * mu - la * (F1_2 * F2_3 - F1_3 * F2_2) * (F2_1 * F3_3 - F2_3 * F3_1) - F2_3 * la * det
        t3[:, [4]] = F1_3 * mu + la * (F1_2 * F2_3 - F1_3 * F2_2) * (F1_1 * F3_3 - F1_3 * F3_1) + F1_3 * la * det
        t3[:, [5]] = -la * (F1_1 * F2_3 - F1_3 * F2_1) * (F1_2 * F2_3 - F1_3 * F2_2)
        t3[:, [6]] = F2_2 * mu + la * (F1_2 * F2_3 - F1_3 * F2_2) * (F2_1 * F3_2 - F2_2 * F3_1) + F2_2 * la * det
        t3[:, [7]] = -F1_2 * mu - la * (F1_2 * F2_3 - F1_3 * F2_2) * (F1_1 * F3_2 - F1_2 * F3_1) - F1_2 * la * det
        t3[:, [8]] = la * (F1_1 * F2_2 - F1_2 * F2_1) * (F1_2 * F2_3 - F1_3 * F2_2)

        t4 = np.zeros((F.shape[0], 9))
        t4[:, [0]] = -la * (F2_1 * F3_3 - F2_3 * F3_1) * (F2_2 * F3_3 - F2_3 * F3_2)
        t4[:, [1]] = F3_3 * mu + la * (F1_2 * F3_3 - F1_3 * F3_2) * (F2_1 * F3_3 - F2_3 * F3_1) + F3_3 * la * det
        t4[:, [2]] = -F2_3 * mu - la * (F1_2 * F2_3 - F1_3 * F2_2) * (F2_1 * F3_3 - F2_3 * F3_1) - F2_3 * la * det
        t4[:, [3]] = mu + la * (F2_1 * F3_3 - F2_3 * F3_1) ** 2.0
        t4[:, [4]] = -la * (F1_1 * F3_3 - F1_3 * F3_1) * (F2_1 * F3_3 - F2_3 * F3_1)
        t4[:, [5]] = la * (F1_1 * F2_3 - F1_3 * F2_1) * (F2_1 * F3_3 - F2_3 * F3_1)
        t4[:, [6]] = -la * (F2_1 * F3_2 - F2_2 * F3_1) * (F2_1 * F3_3 - F2_3 * F3_1)
        t4[:, [7]] = -F3_1 * mu + la * (F1_1 * F3_2 - F1_2 * F3_1) * (F2_1 * F3_3 - F2_3 * F3_1) - F3_1 * la * det
        t4[:, [8]] = F2_1 * mu - la * (F1_1 * F2_2 - F1_2 * F2_1) * (F2_1 * F3_3 - F2_3 * F3_1) + F2_1 * la * det

        t5 = np.zeros((F.shape[0], 9))
        t5[:, [0]] = -F3_3 * mu + la * (F1_1 * F3_3 - F1_3 * F3_1) * (F2_2 * F3_3 - F2_3 * F3_2) - F3_3 * la * det
        t5[:, [1]] = -la * (F1_1 * F3_3 - F1_3 * F3_1) * (F1_2 * F3_3 - F1_3 * F3_2)
        t5[:, [2]] = F1_3 * mu + la * (F1_2 * F2_3 - F1_3 * F2_2) * (F1_1 * F3_3 - F1_3 * F3_1) + F1_3 * la * det
        t5[:, [3]] = -la * (F1_1 * F3_3 - F1_3 * F3_1) * (F2_1 * F3_3 - F2_3 * F3_1)
        t5[:, [4]] = mu + la * (F1_1 * F3_3 - F1_3 * F3_1) ** 2.0
        t5[:, [5]] = -la * (F1_1 * F2_3 - F1_3 * F2_1) * (F1_1 * F3_3 - F1_3 * F3_1)
        t5[:, [6]] = F3_1 * mu + la * (F1_1 * F3_3 - F1_3 * F3_1) * (F2_1 * F3_2 - F2_2 * F3_1) + F3_1 * la * det
        t5[:, [7]] = -la * (F1_1 * F3_2 - F1_2 * F3_1) * (F1_1 * F3_3 - F1_3 * F3_1)
        t5[:, [8]] = -F1_1 * mu + la * (F1_1 * F2_2 - F1_2 * F2_1) * (F1_1 * F3_3 - F1_3 * F3_1) - F1_1 * la * det

        t6 = np.zeros((F.shape[0], 9))
        t6[:, [0]] = F2_3 * mu - la * (F1_1 * F2_3 - F1_3 * F2_1) * (F2_2 * F3_3 - F2_3 * F3_2) + F2_3 * la * det
        t6[:, [1]] = -F1_3 * mu + la * (F1_1 * F2_3 - F1_3 * F2_1) * (F1_2 * F3_3 - F1_3 * F3_2) - F1_3 * la * det
        t6[:, [2]] = -la * (F1_1 * F2_3 - F1_3 * F2_1) * (F1_2 * F2_3 - F1_3 * F2_2)
        t6[:, [3]] = la * (F1_1 * F2_3 - F1_3 * F2_1) * (F2_1 * F3_3 - F2_3 * F3_1)
        t6[:, [4]] = -la * (F1_1 * F2_3 - F1_3 * F2_1) * (F1_1 * F3_3 - F1_3 * F3_1)
        t6[:, [5]] = mu + la * (F1_1 * F2_3 - F1_3 * F2_1) ** 2.0
        t6[:, [6]] = -F2_1 * mu - la * (F1_1 * F2_3 - F1_3 * F2_1) * (F2_1 * F3_2 - F2_2 * F3_1) - F2_1 * la * det
        t6[:, [7]] = F1_1 * mu + la * (F1_1 * F2_3 - F1_3 * F2_1) * (F1_1 * F3_2 - F1_2 * F3_1) + F1_1 * la * det
        t6[:, [8]] = -la * (F1_1 * F2_2 - F1_2 * F2_1) * (F1_1 * F2_3 - F1_3 * F2_1)

        t7 = np.zeros((F.shape[0], 9))
        t7[:, [0]] = la * (F2_1 * F3_2 - F2_2 * F3_1) * (F2_2 * F3_3 - F2_3 * F3_2)
        t7[:, [1]] = -F3_2 * mu - la * (F1_2 * F3_3 - F1_3 * F3_2) * (F2_1 * F3_2 - F2_2 * F3_1) - F3_2 * la * det
        t7[:, [2]] = F2_2 * mu + la * (F1_2 * F2_3 - F1_3 * F2_2) * (F2_1 * F3_2 - F2_2 * F3_1) + F2_2 * la * det
        t7[:, [3]] = -la * (F2_1 * F3_2 - F2_2 * F3_1) * (F2_1 * F3_3 - F2_3 * F3_1)
        t7[:, [4]] = F3_1 * mu + la * (F1_1 * F3_3 - F1_3 * F3_1) * (F2_1 * F3_2 - F2_2 * F3_1) + F3_1 * la * det
        t7[:, [5]] = -F2_1 * mu - la * (F1_1 * F2_3 - F1_3 * F2_1) * (F2_1 * F3_2 - F2_2 * F3_1) - F2_1 * la * det
        t7[:, [6]] = mu + la * (F2_1 * F3_2 - F2_2 * F3_1) ** 2.0
        t7[:, [7]] = -la * (F1_1 * F3_2 - F1_2 * F3_1) * (F2_1 * F3_2 - F2_2 * F3_1)
        t7[:, [8]] = la * (F1_1 * F2_2 - F1_2 * F2_1) * (F2_1 * F3_2 - F2_2 * F3_1)

        t8 = np.zeros((F.shape[0], 9))
        t8[:, [0]] = F3_2 * mu - la * (F1_1 * F3_2 - F1_2 * F3_1) * (F2_2 * F3_3 - F2_3 * F3_2) + F3_2 * la * det
        t8[:, [1]] = la * (F1_1 * F3_2 - F1_2 * F3_1) * (F1_2 * F3_3 - F1_3 * F3_2)
        t8[:, [2]] = -F1_2 * mu - la * (F1_2 * F2_3 - F1_3 * F2_2) * (F1_1 * F3_2 - F1_2 * F3_1) - F1_2 * la * det
        t8[:, [3]] = -F3_1 * mu + la * (F1_1 * F3_2 - F1_2 * F3_1) * (F2_1 * F3_3 - F2_3 * F3_1) - F3_1 * la * det
        t8[:, [4]] = -la * (F1_1 * F3_2 - F1_2 * F3_1) * (F1_1 * F3_3 - F1_3 * F3_1)
        t8[:, [5]] = F1_1 * mu + la * (F1_1 * F2_3 - F1_3 * F2_1) * (F1_1 * F3_2 - F1_2 * F3_1) + F1_1 * la * det
        t8[:, [6]] = -la * (F1_1 * F3_2 - F1_2 * F3_1) * (F2_1 * F3_2 - F2_2 * F3_1)
        t8[:, [7]] = mu + la * (F1_1 * F3_2 - F1_2 * F3_1) ** 2.0
        t8[:, [8]] = -la * (F1_1 * F2_2 - F1_2 * F2_1) * (F1_1 * F3_2 - F1_2 * F3_1)

        t9 = np.zeros((F.shape[0], 9))
        t9[:, [0]] = -F2_2 * mu + la * (F1_1 * F2_2 - F1_2 * F2_1) * (F2_2 * F3_3 - F2_3 * F3_2) - F2_2 * la * det
        t9[:, [1]] = F1_2 * mu - la * (F1_1 * F2_2 - F1_2 * F2_1) * (F1_2 * F3_3 - F1_3 * F3_2) + F1_2 * la * det
        t9[:, [2]] = la * (F1_1 * F2_2 - F1_2 * F2_1) * (F1_2 * F2_3 - F1_3 * F2_2)
        t9[:, [3]] = F2_1 * mu - la * (F1_1 * F2_2 - F1_2 * F2_1) * (F2_1 * F3_3 - F2_3 * F3_1) + F2_1 * la * det
        t9[:, [4]] = -F1_1 * mu + la * (F1_1 * F2_2 - F1_2 * F2_1) * (F1_1 * F3_3 - F1_3 * F3_1) - F1_1 * la * det
        t9[:, [5]] = -la * (F1_1 * F2_2 - F1_2 * F2_1) * (F1_1 * F2_3 - F1_3 * F2_1)
        t9[:, [6]] = la * (F1_1 * F2_2 - F1_2 * F2_1) * (F2_1 * F3_2 - F2_2 * F3_1)
        t9[:, [7]] = -la * (F1_1 * F2_2 - F1_2 * F2_1) * (F1_1 * F3_2 - F1_2 * F3_1)
        t9[:, [8]] = mu + la * (F1_1 * F2_2 - F1_2 * F2_1) ** 2.0

        H = np.hstack([t1, t2, t3, t4, t5, t6, t7, t8, t9])[:, _9x9matrix_3D_ordering_].reshape(-1, 9, 9)
    else:
        raise ValueError("Neo-Hookean supports dim=2 or dim=3")
    return H


# --------------------------------------------------------------------------- #
# Global explicit tier: position (x) variable                                 #
# --------------------------------------------------------------------------- #
def neo_hookean_energy_x(X: np.ndarray, J: sp.sparse.spmatrix, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray) -> float:
    """Assembled Neo-Hookean energy at positions ``X``.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Current vertex positions.
    J : scipy.sparse matrix (t*dim*dim, n*dim)
        Deformation Jacobian.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.

    Returns
    -------
    E : float
        Total Neo-Hookean energy.
    """
    dim = X.shape[1]
    F = (J @ X.reshape(-1, 1)).reshape(-1, dim, dim)
    psi = neo_hookean_energy_element_F(F, mu, lam)
    E = float((np.asarray(vol).reshape(-1, 1) * psi).sum())
    return E


def neo_hookean_gradient_x(X: np.ndarray, J: sp.sparse.spmatrix, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Assembled Neo-Hookean gradient w.r.t. positions ``X``.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Current vertex positions.
    J : scipy.sparse matrix (t*dim*dim, n*dim)
        Deformation Jacobian.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.

    Returns
    -------
    g : np.ndarray (n*dim, 1)
        Assembled energy gradient.
    """
    dim = X.shape[1]
    F = (J @ X.reshape(-1, 1)).reshape(-1, dim, dim)
    P = neo_hookean_gradient_element_F(F, mu, lam)
    P = P * np.asarray(vol).reshape(-1, 1, 1)
    g = J.transpose() @ P.reshape(-1, 1)
    return g


def neo_hookean_hessian_x(X: np.ndarray, J: sp.sparse.spmatrix, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray, psd: bool = True) -> sp.sparse.spmatrix:
    """Assembled Neo-Hookean Hessian w.r.t. positions ``X``.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Current vertex positions.
    J : scipy.sparse matrix (t*dim*dim, n*dim)
        Deformation Jacobian.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.
    psd : bool, optional
        If ``True`` (default), project each per-element block to the nearest
        positive semi-definite matrix before assembly.

    Returns
    -------
    Q : scipy.sparse.csc_matrix (n*dim, n*dim)
        Assembled energy Hessian.
    """
    dim = X.shape[1]
    F = (J @ X.reshape(-1, 1)).reshape(-1, dim, dim)
    He = neo_hookean_hessian_element_F(F, mu, lam)
    He = He * np.asarray(vol).reshape(-1, 1, 1)
    if psd:
        He = psd_project(He)
    H = sp.sparse.block_diag(He)
    Q = J.transpose() @ H @ J
    return Q


# --------------------------------------------------------------------------- #
# Self-contained tier: builds J and vol from rest geometry                    #
# --------------------------------------------------------------------------- #
def neo_hookean_energy(X: np.ndarray, T: np.ndarray, mu: np.ndarray, lam: np.ndarray, U: Optional[np.ndarray] = None) -> float:
    """Neo-Hookean energy, building the operator and weights from rest geometry.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Rest vertex positions. Used to build ``J`` and ``vol``.
    T : np.ndarray (t, dim+1)
        Element connectivity.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.
    U : np.ndarray (n, dim), optional
        Current vertex positions. Defaults to ``X``.

    Returns
    -------
    E : float
        Total Neo-Hookean energy.
    """
    if U is None:
        U = X
    J = deformation_jacobian(X, T)
    vol = volume(X, T)
    return neo_hookean_energy_x(U, J, mu, lam, vol)


def neo_hookean_gradient(X: np.ndarray, T: np.ndarray, mu: np.ndarray, lam: np.ndarray, U: Optional[np.ndarray] = None) -> np.ndarray:
    """Neo-Hookean gradient, building the operator and weights from rest geometry.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Rest vertex positions.
    T : np.ndarray (t, dim+1)
        Element connectivity.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.
    U : np.ndarray (n, dim), optional
        Current vertex positions. Defaults to ``X``.

    Returns
    -------
    g : np.ndarray (n*dim, 1)
        Assembled energy gradient.
    """
    if U is None:
        U = X
    J = deformation_jacobian(X, T)
    vol = volume(X, T)
    return neo_hookean_gradient_x(U, J, mu, lam, vol)


def neo_hookean_hessian(X: np.ndarray, T: np.ndarray, mu: np.ndarray, lam: np.ndarray, U: Optional[np.ndarray] = None, psd: bool = True) -> sp.sparse.spmatrix:
    """Neo-Hookean Hessian, building the operator and weights from rest geometry.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Rest vertex positions.
    T : np.ndarray (t, dim+1)
        Element connectivity.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.
    U : np.ndarray (n, dim), optional
        Current vertex positions. Defaults to ``X``.
    psd : bool, optional
        Project per-element blocks to PSD before assembly. Default ``True``.

    Returns
    -------
    Q : scipy.sparse.csc_matrix (n*dim, n*dim)
        Assembled energy Hessian.
    """
    if U is None:
        U = X
    J = deformation_jacobian(X, T)
    vol = volume(X, T)
    return neo_hookean_hessian_x(U, J, mu, lam, vol, psd=psd)