import scipy as sp
import numpy as np

from simkit.energies import arap_hessian
from .massmatrix import massmatrix
from .eigs import eigs

def modal_analysis(X, T, k, bI=None):
    n = X.shape[0]
    dim = X.shape[1]
    H = arap_hessian(X=X, T=T)
    M = massmatrix(X, T)
    M = sp.sparse.kron(M, sp.sparse.identity(dim))

    if bI is not None:
        bI = np.array(bI)
        assert(bI.ndim==1)
        bIr = np.repeat(bI[:, None], X.shape[1], axis=1) 

        bIe = bIr * X.shape[1] + np.arange(dim)

        Ii = np.setdiff1d(np.arange(n*dim), bIe)
        H = H[Ii, :][:, Ii]
        M = sp.sparse.diags(M.diagonal()[Ii,], 0)

        [Ei, Bi] = eigs(H, k=k, M=M)
        B = np.zeros((n*dim, k))
        B[Ii, :] = Bi
        E = Ei
    else:
        [E, B] = eigs(H, k=k, M=M)

    E = E.real
    B = B.real

    return E, B
