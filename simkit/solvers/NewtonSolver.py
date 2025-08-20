import scipy as sp
import numpy as np

from .Solver import Solver, SolverParams
from ..backtracking_line_search import backtracking_line_search


class NewtonSolverParams(SolverParams):
    def __init__(self, tolerance= 1e-6, max_iter = 1, do_line_search = True):
        self.tolerance = tolerance
        self.max_iter = max_iter

        self.do_line_search = do_line_search
        return

class NewtonSolver(Solver):

    def __init__(self, energy_func, gradient_func, hessian_func, p : NewtonSolverParams = NewtonSolverParams()):
        self.p = p
        self.energy_func = energy_func
        self.gradient_func = gradient_func
        self.hessian_func = hessian_func
        pass


    def solve(self, x0, return_info=False):
        
        x = x0.copy()
        if return_info:
            info = {'g' : [], 'dx' : [], 'alphas' : [], 'iters' : -1}
        
        for i in range(self.p.max_iter):
            g = self.gradient_func(x)
            H = self.hessian_func(x)

            # if sparse matrix
            if sp.sparse.issparse(H):
                dx = sp.sparse.linalg.spsolve(H.tocsc(), -g).reshape(-1, 1)
            else:
                dx = sp.linalg.solve(H, -g).reshape(-1, 1)

            if self.p.do_line_search:
                energy_func = lambda z: self.energy_func(z)
                alpha, lx, ex = backtracking_line_search(energy_func, x, g, dx)
            else:
                alpha = 1.0

            x += alpha * dx

            if return_info:
                info['g'].append(g)
                info['dx'].append(dx)
                info['alphas'].append(alpha)
                info['iters'] = i

            if np.linalg.norm(g) < 1e-4:
                break
        if return_info:
            return x, info
        else:   
            return x
    

