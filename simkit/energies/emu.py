import numpy as np
import scipy as sp

def emu_energy_F(F, d, a, vol):
    """
    F - t x d x d
    d - t x d
    a - t x 1
    """
    FF = F.transpose(0, 2, 1) @ F
    de = d[:, :, None].copy()
    dFFd = de.transpose(0, 2, 1) @ FF @ de
    energy = np.sum(vol.reshape(-1, 1,1) * a.reshape(-1, 1, 1) * dFFd) * 0.5
    return energy

def emu_gradient_dF(F, d, a, vol):
    de = d[:, :, None].copy()
    ddT =  de @ de.transpose(0, 2, 1)
    P =  F @  ddT * (vol.reshape(-1, 1, 1) * a.reshape(-1, 1, 1))
    return P

def emu_gradient_dx(X, d, a, vol, J):
    F = (J @ X.reshape(-1, 1)).reshape(-1, 2, 2)
    P = emu_gradient_dF(F, d, a, vol)
    g = J.T @ P.reshape(-1, 1)
    return g


def emu_force_matrix(X, d, vol, J):
    F = (J @ X.reshape(-1, 1)).reshape(-1, 2, 2)
    a= np.ones((d.shape[0], 1))
    P = emu_gradient_dF(F, d, a, vol)
    P_mat = sp.sparse.diags(P.reshape(-1)) @ J
    N = sp.sparse.kron(sp.sparse.eye(d.shape[0]), np.ones((1, 4)))
    K = N @ P_mat
    return K

def emu_hessian_d2F(F, d, a, vol):

    dim = F.shape[-1]

    de = d[:, :, None].copy()
    ddT =  de @ de.transpose(0, 2, 1)
    Q = np.kron( np.identity(dim), ddT)

    Q2 = Q * vol.reshape(-1, 1, 1) * a.reshape(-1, 1, 1)
    return Q2

def emu_hessian_d2x(X, d, a, vol, J):
    dim = F.shape[-1]

    F = (J @ X.reshape(-1, 1)).reshape(-1, dim, dim)
    Q = emu_hessian_d2F(F, d, a, vol)
    H = J.T @  sp.sparse.block_diag(Q) @ J
    return H
