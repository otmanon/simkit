
import scipy as sp
import numpy as np

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