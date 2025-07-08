

import scipy as sp
import numpy as np


def remove_redundant_columns(B, M=None, threshold=1e-16):



    if M is None:
        M = sp.sparse.identity(B.shape[0])
    # M = sp.sparse.identity(B.shape[0])

    msqrt = np.sqrt(M.diagonal())
    msqrti = 1 / msqrt
    Msqrt = sp.sparse.diags(msqrt, 0)
    Msqrti = sp.sparse.diags(msqrti, 0)
    Bm = Msqrt @ B


    [Q, R] = np.linalg.qr(Bm, mode='reduced')

    [U, s, V] = np.linalg.svd(Bm, full_matrices=False)

    sI = np.where(s > 1e-14)[0]
    S = np.diag(s)
    B2 = U @ S[:, sI] @ V[sI, :][:, sI]

    B3 = Msqrti @ B2

    return B3