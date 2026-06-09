import scipy as sp
import numpy as np

from ..backtracking_line_search import backtracking_line_search


def sqp_mfem(p0, energy_func, hess_blocks_func, grad_blocks_func,
                    tolerance=1e-4, max_iter=100, do_line_search=True, verbose=False):
    """
    SQP solver for the MFEM system from https://www.dgp.toronto.edu/projects/subspace-mfem/ , Section 4.

    The full system looks like:
        [Hu    0    Gu] [du]   - [fu]
        [0     Hz   Gz] [dz] = - [fz]
        [Gz.T  0     0] [mu]   - [fmu]

    Where Gz is diagonal and easily invertible. Using this fact, we can rewrite the
    system into a small solve and a matrix mult

    (Hu + Gu Gz^-1 Hz Gz^-1 Gu.T) du = -fu + Gu Gz^-1 fz - Gu Gz^-1 Hz Gz^-1 fmu
    dz = -Gz^-1 (fz + Hz du)

    Parameters
    ----------
    p0 : (n, 1) np.ndarray
        Initial guess for the stacked state.
    energy_func : callable
        Energy function to minimize. Only used when ``do_line_search`` is True.
    hess_blocks_func : callable
        Maps ``p`` to the hessian blocks ``Hu, Hz, Gu, Gz, Gzi``.
    grad_blocks_func : callable
        Maps ``p`` to the gradient blocks ``fu, fz, fmu``.
    tolerance : float
        Convergence tolerance on the norm of the gradient.
    max_iter : int
        Maximum number of iterations.
    do_line_search : bool
        If True, run a backtracking line search to pick the step size.
    verbose : bool
        Unused placeholder for diagnostics.

    Returns
    -------
    p : (n, 1) np.ndarray
        Minimizer estimate.
    """
    p = p0.copy()
    for i in range(max_iter):

        [H_u, H_z, G_u, G_z, G_zi] = hess_blocks_func(p)

        [f_u, f_z, f_mu] = grad_blocks_func(p)

        # form K
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

        mu = -G_zi @ (f_z + H_z @ dz)

        g = np.vstack([f_u + G_u @ mu, f_z + G_z @ mu])
        dp = np.vstack([du, dz])
        if do_line_search:
            energy_lambda = lambda p : energy_func(np.vstack([p, mu]))
            alpha, lx, ex = backtracking_line_search(energy_lambda, p[:-mu.shape[0]], g, dp)
        else:
            alpha = 1.0

        p[:-mu.shape[0]] += alpha * dp
        p[-mu.shape[0]:] =  mu

        nd = float((g_u.T @ du).item())
        if nd < tolerance:
            break

    return p
