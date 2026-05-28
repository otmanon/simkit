import numpy as np

from .Solver import Solver, SolverParams
from ..backtracking_line_search import backtracking_line_search


class GradientDescentSolverParams(SolverParams):
    def __init__(self, tolerance=1e-6, max_iter=1, do_line_search=True, step_size=1.0):
        self.tolerance = tolerance
        self.max_iter = max_iter

        self.do_line_search = do_line_search
        self.step_size = step_size
        return


class GradientDescentSolver(Solver):

    def __init__(self, energy_func, gradient_func, p: GradientDescentSolverParams = GradientDescentSolverParams()):
        self.p = p
        self.energy_func = energy_func
        self.gradient_func = gradient_func
        pass


    def solve(self, x0, return_info=False):

        x = x0.copy()
        if return_info:
            info = {'g': [], 'dx': [], 'alphas': [], 'iters': -1}

        for i in range(self.p.max_iter):
            g = self.gradient_func(x)

            dx = -g.reshape(-1, 1)

            if self.p.do_line_search:
                energy_func = lambda z: self.energy_func(z)
                alpha, lx, ex = backtracking_line_search(energy_func, x, g, dx)
            else:
                alpha = self.p.step_size


            x += alpha * dx

            if return_info:
                info['g'].append(g)
                info['dx'].append(dx)
                info['alphas'].append(alpha)
                info['iters'] = i

            if np.linalg.norm(alpha * dx) < 1e-6:
                break

        if return_info:
            return x, info
        else:
            return x
