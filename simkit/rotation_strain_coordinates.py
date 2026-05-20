import numpy as np
import scipy as sp

from .deformation_jacobian import deformation_jacobian
from .dirichlet_penalty import dirichlet_penalty

from .volume import volume

from .membrane_deformation_jacobian import membrane_deformation_jacobian
class RSPrecompute():

    def __init__(self, X, T, pinned=None):
        self.X = X
        self.T = T
        dim = X.shape[1]
        
        
        if X.shape[1] == 3 and T.shape[1] == 3:
            self.J = membrane_deformation_jacobian(X, T)
            dd = 6
        else:
            self.J = deformation_jacobian(X, T)
            dd = dim * dim
        if pinned is None:
            mean_X = X.mean(axis=0).reshape(-1, dim)
            pinned = np.where(np.linalg.norm(X - mean_X, axis=1) < 0.01)[0]
        
        H_pin, _b = dirichlet_penalty(pinned, X[pinned], X.shape[0], 1e8)
      
        J = self.J
        vol = volume(X, T)
        Vol = sp.sparse.diags(vol.flatten())
        Vol = sp.sparse.kron(Vol, sp.sparse.identity(dd))
        L = J.T @ Vol @ J
        self.K = J.T @ Vol
        A = L + H_pin 
        self.factorization = sp.sparse.linalg.factorized(A.tocsc())
        
    def fit_displacements_to_jacobian(self, Y):
        return self.factorization(self.K @ Y.reshape(-1, 1))
        
    

# def rotation_strain_coordinates(X, T, u, 
#                                 pinned=None, pre=None, return_pre=True,
#                                 project_stretch_psd=True,
#                                 projection_threshold=1e-1):
    
#     dim = X.shape[1]
#     u = u.reshape(-1, 1)
#     x0 = X.reshape(-1, 1)
    
#     if pre is None:
#         pre = RSPrecompute(X, T, pinned)
    
#     if dim == 3 and T.shape[1] == 3:
#         grad_u= (pre.J @ u).reshape(-1, 2, 3)        

#         # add constant along normal direction of triangle
        
#     else:
#         grad_u= (pre.J @ u).reshape(-1, dim, dim)
        
               
#     I = np.identity(dim)[None, ...]

#     symmetric = (grad_u + grad_u.transpose(0, 2, 1))/2.0 + I
#     if project_stretch_psd:
#         eval, evec = np.linalg.eig(symmetric)
#         eval = np.maximum(eval, projection_threshold)
#         symmetric = evec.transpose(0, 2, 1) @ (eval[:, :, None] * evec)
#     # U, Sig, V = np.linalg.svd(symmetric + I)
#     # USV = U @ Sig[:, :, None] * V# - (symmetric + I)
#     antisymmetric = (grad_u - grad_u.transpose(0, 2, 1))/2.0
#     if dim == 2:
#         # sin_theta = - antisymmetric[:, 0, 1]    
#         w = -antisymmetric[:, 0, 1]
#         R = np.array([[np.cos(w), -np.sin(w)],
#                         [np.sin(w), np.cos(w)]]).transpose(2, 0, 1)
#     elif dim ==3:
#         w = np.concatenate( [-antisymmetric[:, 1, 2], antisymmetric[:, 0, 2], -antisymmetric[:, 0, 1]], axis=0)
#         theta = np.linalg.norm(w, axis=1) # angle by which we are rotating
#         direction = w / theta[:, None] # unit vector in the direction of rotation
#         R = antisymmetric 
    

#     Y = R @ (symmetric  ) - I     
    
#     u_rs = pre.fit_displacements_to_jacobian(Y).reshape(-1, dim)
#     # fit positions to deformation gradient.   
    
#     if return_pre:
#         return u_rs, pre
#     else:
#         return u_rs


def rotation_strain_coordinates(X, T, u, 
                                pinned=None, pre=None, return_pre=True,
                                project_stretch_psd=True,
                                projection_threshold=1e-8):
    
    dim = X.shape[1]
    u = u.reshape(-1, 1)
    
    if pre is None:
        pre = RSPrecompute(X, T, pinned)
    
    # -------------------------------------------------
    # MEMBRANE CASE (3D embedding, 2D triangle)
    # -------------------------------------------------
    if dim == 3 and T.shape[1] == 3:
        
        # F is 3x2 per triangle
        F = (pre.J @ u).reshape(-1, 3, 2)

        # Add identity in tangent directions
        I2 = np.eye(2)[None, :, :]
        I2 = np.repeat(I2, F.shape[0], axis=0)
        F = F + np.concatenate(
            [I2, np.zeros((F.shape[0], 1, 2))], axis=1
        )

        # --- Polar decomposition for rectangular F ---
        C = np.matmul(F.transpose(0,2,1), F)   # 2x2
        
        if project_stretch_psd:
            evals, evecs = np.linalg.eigh(C)
            evals = np.maximum(evals, projection_threshold)
            C = evecs @ (evals[..., None] * evecs.transpose(0,2,1))

        C_inv_sqrt = np.linalg.inv(
            np.array([sp.linalg.sqrtm(C[i]) for i in range(C.shape[0])])
        )

        R = np.matmul(F, C_inv_sqrt)  # 3x2 pure rotation
        
        # Embedded identity (3x2)
        I_emb = np.zeros_like(R)
        I_emb[:,0,0] = 1.0
        I_emb[:,1,1] = 1.0

        Y = R - I_emb

    # -------------------------------------------------
    # ORIGINAL SOLID / 2D CASE (unchanged)
    # -------------------------------------------------
    else:
        grad_u = (pre.J @ u).reshape(-1, dim, dim)
        I = np.identity(dim)[None, ...]

        F = grad_u + I

        symmetric = (F + F.transpose(0,2,1)) / 2.0
        antisymmetric = (F - F.transpose(0,2,1)) / 2.0

        if dim == 2:
            w = -antisymmetric[:, 0, 1]
            R = np.array([[np.cos(w), -np.sin(w)],
                          [np.sin(w),  np.cos(w)]]
                        ).transpose(2,0,1)
        else:
            # safer: use polar instead of axis-angle approx
            U, _, Vt = np.linalg.svd(F)
            R = U @ Vt

        Y = R - I

    # -------------------------------------------------
    # Fit back to displacement coordinates
    # -------------------------------------------------
    u_rs = pre.fit_displacements_to_jacobian(Y).reshape(-1, dim)

    if return_pre:
        return u_rs, pre
    else:
        return u_rs