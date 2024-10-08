import scipy as sp
from scipy.sparse import hstack, vstack
from scipy.sparse.linalg import LinearOperator
import numpy as np
import cvxopt
import cvxopt.umfpack


#Overrides scipy's defualt LU factorization,
# which uses https://portal.nersc.gov/project/sparse/superlu/
# and instead use LU decopmosition from UMFPACK
class umfpack_LU_LinearOperator(LinearOperator):
    def __init__(self, A):
        self.A = A
        self.shape = A.shape
        self.dtype = A.dtype
        self.A = A

        [I, J] = A.nonzero()
        v = A[I, J]
        Ac = cvxopt.spmatrix(v, I, J, A.shape)
        F = cvxopt.umfpack.symbolic(Ac)
        self.numeric = cvxopt.umfpack.numeric(Ac, F)
        self.A = Ac
        # bc = cvxopt.matrix(b)
        # cvxopt.umfpack.linsolve(Ac, bc)
        # cnp = np.array(bc)
        super(umfpack_LU_LinearOperator, self).__init__(A.dtype, A.shape)

    def _matvec(self, v):
        b = cvxopt.matrix(v)
        x = b
        cvxopt.umfpack.solve(self.A, self.numeric, b)
        return x

'''
Computes Generalized Eigenvalues and Eigenvectors of sparse non-definite matrix A, with massmatrix M

Inputs:
A - n x n indefinite sparse matrix

Optional
k - int number of eigenvectors/values tos olve for (default=5)
M - n x n indefinite mass matrix

Returns
D - k x 1 eigenvalues
B - n x k eigenvectors
'''
def eigs(A, k=5, M=None):
    """
    Computes Generalized Eigenvalues and Eigenvectors of sparse non-definite matrix A, with massmatrix M

    Parameters
    ----------
    A : (n, n) float sparse matrix
        Indefinite sparse matrix
    k : int
        Number of eigenvectors/values to solve for (default=5)
    M : (n, n) float sparse matrix
        Indefinite mass matrix

    Returns
    --------
    D : (k, 1) float numpy array
        Eigenvalues
    B : (n, k) float numpy array
        Eigenvectors

    """
    if k == 0:
        return np.zeros((0)), np.zeros((A.shape[0], 0))
    if M is None:
        M = sp.sparse.identity(A.shape[0])


    OpInv = umfpack_LU_LinearOperator(A)
    # MInv = umfpack_LU_LinearOperator(M)
    n0 = np.ones(A.shape[0])

    if k < A.shape[0] - 1:
        [D, B] = sp.sparse.linalg.eigs(A, M=M, k=k, sigma=0,
                              which='LM', OPinv=OpInv, v0 = n0)
    else:
        [D, B] = sp.linalg.eigh(A.toarray(), b=M.toarray())
        B = B[:, :k]
        D = D[:k]

    return D, B