import scipy as sp
import numpy as np

from ..backtracking_line_search import backtracking_line_search


def newton_solver(x0, energy_func, gradient_func, hessian_func,
                  tolerance=1e-6, max_iter=1, do_line_search=True, return_info=False):
    """
    Newton's method for minimizing a scalar energy.

    At each iteration we solve ``H dx = -g`` for the search direction, optionally
    backtrack along it, and step ``x <- x + alpha dx`` until the step is smaller
    than ``tolerance`` or ``max_iter`` is reached.

    Parameters
    ----------
    x0 : (n, 1) np.ndarray
        Initial guess.
    energy_func : callable
        Maps ``x`` to a scalar energy. Only used when ``do_line_search`` is True.
    gradient_func : callable
        Maps ``x`` to the gradient ``g``.
    hessian_func : callable
        Maps ``x`` to the hessian ``H`` (dense or sparse).
    tolerance : float
        Convergence tolerance on the norm of the step ``alpha * dx``.
    max_iter : int
        Maximum number of Newton iterations.
    do_line_search : bool
        If True, run a backtracking line search to pick the step size.
    return_info : bool
        If True, also return a dict with per-iteration diagnostics.

    Returns
    -------
    x : (n, 1) np.ndarray
        Minimizer estimate.
    info : dict, optional
        Returned only when ``return_info`` is True.
    """
    x = x0.copy()
    if return_info:
        info = {'g': [], 'dx': [], 'alphas': [], 'iters': -1}

    for i in range(max_iter):
        g = gradient_func(x)
        H = hessian_func(x)

        # if sparse matrix
        if sp.sparse.issparse(H):
            dx = sp.sparse.linalg.spsolve(H.tocsc(), -g).reshape(-1, 1)
        else:
            dx = sp.linalg.solve(H, -g).reshape(-1, 1)

        if do_line_search:
            alpha, lx, ex = backtracking_line_search(energy_func, x, g, dx)
        else:
            alpha = 1.0

        x += alpha * dx

        if return_info:
            info['g'].append(g)
            info['dx'].append(dx)
            info['alphas'].append(alpha)
            info['iters'] = i

        if np.linalg.norm(alpha * dx) < tolerance:
            break

    if return_info:
        return x, info
    else:
        return x
