import scipy as sp
import numpy as np


from ..backtracking_line_search import backtracking_line_search


from .Solver import Solver, SolverParams

class SQPMFEMSolverParams(SolverParams):

    def __init__(self, do_line_search=True, max_iter=100, tol=1e-6, verbose=False):
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.do_line_search = do_line_search
        return
    
class SQPMFEMSolver(Solver):

    def __init__(self, energy_func, hess_blocks_func, grad_blocks_func, params : SQPMFEMSolverParams = None):
        """
         SQP Solver for the MFEM System from https://www.dgp.toronto.edu/projects/subspace-mfem/ ,  Section 4.

         The full system looks like:
            [Hu    0    Gu] [du]   - [fu]
            [0     Hz   Gz] [dz] = - [fz]
            [Gz.T  0     0] [mu]   - [fmu]

        Where Gz is diagonal and easily invertible. Using this fact, we can rewrite the system into a small solve and a matrix mult

        (Hu + Gu Gz^-1 Hz Gz^-1 Gu.T) du = -fu + Gu Gz^-1 fz - Gu Gz^-1 Hz Gz^-1 fmu
        dz = -Gz^-1 (fz + Hz du)

        Parameters
        ----------
        energy_func : function
            Energy function to minimize
        hess_blocks_func : function
            Function that returns the important blocks of the hessian: Hu, Hz, Gu, Gzi
        grad_blocks_func : function
            Function that returns the blocks of the gradient: fu, fz, fmu
        params : SQPMFEMSolverParams
            Parameters for the solver

        """

        if params is None:
            params = SQPMFEMSolverParams()
        self.params = params
        self.energy_func = energy_func

        self.hess_blocks_func = hess_blocks_func
        self.grad_blocks_func = grad_blocks_func
        return

    
    def solve(self, p0):
        
        p = p0.copy()
        for i in range(self.params.max_iter):

            [H_u, H_z, G_u, G_z, G_zi] = self.hess_blocks_func(p)            
            
            [f_u, f_z, f_mu] = self.grad_blocks_func(p)

            #form K
            K = G_u @ G_zi @ H_z @ G_zi @ G_u.T 
            Q = H_u + K

            # form g_u
            g_u = -f_u + G_u @ G_zi @ (f_z - H_z @ G_zi @ f_mu)
            
            if sp.sparse.issparse(Q):
                du = sp.sparse.linalg.spsolve(Q, g_u)
            else:
                du = sp.linalg.solve(Q, g_u)
                
            if du.ndim == 1:
                du = du.reshape(-1, 1)
            

            g_z = - (f_mu + G_u.T @ du)
            dz = G_zi @ g_z

            g = np.vstack([f_u, f_z])
            dp = np.vstack([du, dz])
            if self.params.do_line_search:
                energy_func = lambda z: self.energy_func(z)
                alpha, lx, ex = backtracking_line_search(energy_func, p, g, dp)
            else:
                alpha = 1.0

            p += alpha * dp
            if np.linalg.norm(g) < 1e-4:
                break

        return p
    
