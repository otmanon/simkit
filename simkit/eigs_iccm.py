import numpy as np
import scipy as sp
import cvxopt

def sp2cvxopt(A):
    if sp.sparse.issparse(A):
        i, j, v = sp.sparse.find(A)
        A_cvxopt = cvxopt.spmatrix(v, i, j)
    else:
        A_cvxopt = cvxopt.matrix(A)
    return A_cvxopt

def cvxopt2np(A):
    if isinstance(A, cvxopt.spmatrix):
        return np.array(A)
    else:
        return np.array(A)

def eigs_iccm(H, l, k, M=None, tolerance=1e-6, max_iters=100, verbose=False):
    """
    Solves for the "sparse" vibration modes of a ystem H by iteratively solving the following problem:
    
    min_x 0.5 * x.T @ H @ x +  l * || x ||_1
    
    s.t. x.T @ M @ x = 1
    
    s.t. x.T @ U = 0
    
    by following "Compressed Vibration Modes of Elastic Bodies" https://www.sciencedirect.com/science/article/abs/pii/S0167839617300377
    
    
    where x is the displacement vector, H is the stiffness matrix, l is the regularization parameter, M is the mass matrix, and U is the running set of previous eigenvectors.
    
    Parameters
    ----------
    H : (n, n) float sparse matrix
        Stiffness matrix
    l : float
        l1 Regularization parameter
    k : int
        Number of modes to solve for
    M : (n, n) float sparse matrix
        Mass matrix
    tolerance : float
        Tolerance for the optimization
    max_iters : int
        Maximum number of iterations
    verbose : bool
        Whether to print verbose output
    """
    dim_x = H.shape[0]    
    U = np.zeros((dim_x, 0))

            
    cvxopt.solvers.options['show_progress'] = False
    
    # energy function to measure convergence
    def energy_func(x):
        e = 0.5 *  x.T @ H @ x +  l * np.abs(x).sum()
        return e.flatten()
    
    for mode in range(k):
        ck = np.random.randn(dim_x, 1)
        ck = ck / np.sqrt(ck.T  @ M @ ck)
        
        for i in range(max_iters):
            e_prev = energy_func(ck)
            H_exp = sp.sparse.bmat([[H, -H], [-H, H]]).tocsc()
            
            rhs_exp = l * np.concatenate((M.sum(axis=1),  M.sum(axis=1)), axis=0)
    
            un_eq_row = np.concatenate((M @ ck , - M @ ck), axis=0)
            un_eq_rhs = np.ones((1, 1))
            
            oc_eq_row = np.concatenate(( M @ U, - M @ U), axis=0)
            oc_eq_rhs = np.zeros((U.shape[1], 1))
            
            A = np.concatenate((un_eq_row, oc_eq_row), axis=1).T
            b = np.concatenate((un_eq_rhs, oc_eq_rhs), axis=0)
            
            in_eq_mat = sp.sparse.identity(dim_x*2).tocsc()
            in_eq_rhs = np.zeros((dim_x*2, 1))
            
            # confert from scipy csc matrix to cvxopt sparse matrix
            P = sp2cvxopt(H_exp)
            q = sp2cvxopt(rhs_exp)
            A = sp2cvxopt(A)
            b = sp2cvxopt(b)
            G = sp2cvxopt(-in_eq_mat)
            h = sp2cvxopt(in_eq_rhs)
            res = cvxopt.solvers.qp(P, q,  G, h,A, b)
           
            u = res['x'][:dim_x] - res['x'][dim_x:]
            u = cvxopt2np(u)
            
            ck = u / np.sqrt(u.T @ M @ u)
            
            e_curr = energy_func(ck)
            decrement = np.abs(e_curr - e_prev)
            
            if verbose:
                print(f"mode : {mode} , iter : {i} , decrement : {decrement}")
            if np.linalg.norm(decrement) < tolerance:
                break
    
        U = np.concatenate((U, ck.reshape(-1, 1)), axis=1)
    return U