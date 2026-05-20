from re import A
import numpy as np
import scipy as sp

from simkit.eigs import eigs
from simkit.fold_vector_hessian import fold_vector_hessian
from simkit.lbs_jacobian import lbs_jacobian
from simkit.orthonormalize import orthonormalize
from umfpack_lu_solve import umfpack_lu_solve




def force_dual_modes(H, Sigma_F_sqrt, m, M=None, HiSigma_F_sqrt=None, Sigma_F_inv=None):
    
    if Sigma_F_inv is not None:
        Sigma_U_inv = H @ Sigma_F_inv @ H

        D, U = eigs(Sigma_U_inv, m, M=M )
    else:
        if HiSigma_F_sqrt is None:
            # get rid of zero columns of DA
            # DA = remove_zero_cols(DA)
            HiSigma_F_sqrt = umfpack_lu_solve(H, Sigma_F_sqrt)
        if M is None:
            U, D, V = np.linalg.svd(HiSigma_F_sqrt, full_matrices=False)
            U = U[:, :m]
            D = D**2
        else:
            L = sp.sparse.diags(np.sqrt(M.diagonal()))
            Li = sp.sparse.diags(1 / L.diagonal())
            Y, D, V = np.linalg.svd(L @ HiSigma_F_sqrt, full_matrices=False)
            Y = Y[:, :m]
            
            U = Li @ Y
            D = D**2
        

    return D, U


def force_dual_modes_sqrt(H, sigma_F_sqrt, m, M_sqrt=None, use_cvxopt=False):
    """
    Computes Force Dual Modes Assuming a square root of the force covariance matrix sigma_F
    """
    if use_cvxopt:
        HiC = umfpack_lu_solve(H, sigma_F_sqrt)
    else:
        HiC = sp.sparse.linalg.spsolve(H, sigma_F_sqrt)
        if HiC.ndim == 1:
            HiC = HiC.reshape(-1, 1)
    if M_sqrt is not None:
        M_sqrt_inv = sp.sparse.diags(1.0/M_sqrt.diagonal())
    else:
        M_sqrt = sp.sparse.identity(H.shape[0])
        M_sqrt_inv = sp.sparse.identity(H.shape[0])

    U, D, V = np.linalg.svd( M_sqrt @ HiC, full_matrices=False)
    U = U[:, :m]
    D = D**2
    Y = M_sqrt_inv @ U
    return D, Y

def force_dual_modes_diagonal(H, sigma_F, m, M=None, return_components=False):
    """
    Computes Force Dual Modes Assuming a diagonal force covariance matrix sigma_F
    """
    
    if not isinstance(sigma_F, list):
        sigma_F_list = [sigma_F]
    else:
        sigma_F_list = sigma_F
    
    B_full = np.zeros((H.shape[0], m*len(sigma_F_list)))
    D_full = np.zeros((m*len(sigma_F_list)))
    
    D_list = []
    B_list = []
    
    for i, sigma_F in enumerate(sigma_F_list):
        sigma_F_inv = sp.sparse.diags(1.0/sigma_F.diagonal())
        sigma_U = H @ sigma_F_inv @ H.T
        Di, B = eigs(sigma_U, m, M=M)   
        D = 1.0/Di
        
        D_list.append(D)
        B_list.append(B)
        
        B_full[:, i*m:(i+1)*m] = B
        D_full[i*m:(i+1)*m] = D
    
    if return_components:
        return D_full, B_full, D_list, B_list
    else:
        return D_full, B_full

def force_dual_skinning_eigenmodes_diagonal(X, H, sigma_F, m, M=None):
    """
    Computes Force Dual SKinning Eigenmodes assuming a diagonal force covariance matrix sigma_F

    """
    sigma_F_inv = sp.sparse.diags(1.0/sigma_F.diagonal())
    sigma_U = H  @ sigma_F_inv  @  H.T
    dim = X.shape[1]
    Hs = fold_vector_hessian(sigma_U, dim)
    if M is not None:
        if M.shape[0] == H.shape[0]:
            Ms = fold_vector_hessian(sigma_U, dim, M=M)
        else:
            Ms = M

    [D, W] = eigs(Hs, m, M=Ms)
    B = lbs_jacobian(X, W)

    return W, D, B