
import scipy as sp
import numpy as np

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
