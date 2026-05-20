import numpy as np
import scipy as sp
import cvxopt
import cvxopt.umfpack

def umfpack_lu_solve(A, b):
    """
    Solves Ax = b using LU factorization with umfpack.
    Parameters
    ----------
    A : (n, n) float numpy array
        Matrix to solve
    b : (n, ) float numpy array
        Right hand side

    Returns
    -------
    x : (n, ) float numpy array
        Solution to Ax = b
    """
    # A += sp.sparse.eye(A.shape[0])*1
    [I, J] = A.nonzero()
    v = A.data
    Ac = cvxopt.spmatrix(v, I, J, A.shape)
    bc = cvxopt.matrix(b)
    info = cvxopt.umfpack.linsolve(Ac, bc)
    cnp = np.array(bc)
    return cnp

