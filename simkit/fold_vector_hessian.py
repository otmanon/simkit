
import scipy as sp
import numpy as np


def fold_vector_hessian(H, dim, operation='sum', full=False):

    n = H.shape[0]//dim
    L = sp.sparse.csc_matrix((n, n))
    for i in range(dim):
        Ii = np.arange(n)*dim + i
        if not full:
            if operation == 'sum':
                L = L + H[Ii, :][:, Ii]
            if operation == 'abssum':
                L = L + np.abs(H[Ii, :][:, Ii])
            if operation == 'squaresum':
                L = L + H[Ii, :][:, Ii] ** 2
        if full:
            for j in range(dim):
                Ji = np.arange(n)*dim + j
                if operation == 'sum':
                    L = L + H[Ii, :][:, Ji]
                if operation == 'abssum':
                    L = L + np.abs(H[Ii, :][:, Ji])
                if operation == 'squaresum':
                    L = L + H[Ii, :][:, Ji] ** 2
    L = L / dim
    return L
