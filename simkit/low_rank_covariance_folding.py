import numpy as np
import scipy as sp


from .umfpack_lu_solve import umfpack_lu_solve
from .selection_matrix import selection_matrix

def low_rank_covariance_folding(H, C, dim):

    n = H.shape[0] // dim
    HiC = umfpack_lu_solve(H, C)
    # HiC = sp.sparse.linalg.spsolve(H, C)
    inds = np.arange(HiC.shape[0])
    S_list = []
    for i in range(dim):
        inds_i = inds[i::dim]
        S = selection_matrix(inds_i, n*dim)
        S_list.append(S)
    Se = sp.sparse.hstack(S_list)

    Ge = np.kron(np.identity(dim), HiC)
    SGe = Se @ Ge


    return SGe