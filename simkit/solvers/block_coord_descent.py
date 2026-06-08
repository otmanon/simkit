import numpy as np


def block_coord(x0, global_step_func, local_step_func,
                       tolerance=1e-6, max_iter=1):
    """
    Block coordinate descent alternating a local and a global step.

    Each iteration computes auxiliary variables with ``local_step_func`` and then
    updates the state with ``global_step_func`` until the change in state is
    smaller than ``tolerance`` or ``max_iter`` is reached.

    Parameters
    ----------
    x0 : (n, 1) np.ndarray
        Initial guess.
    global_step_func : callable
        Maps ``(x, r)`` to the updated state, where ``r`` is the local result.
    local_step_func : callable
        Maps ``x`` to the local auxiliary result ``r``.
    tolerance : float
        Convergence tolerance on the norm of the state change.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    x : (n, 1) np.ndarray
        Minimizer estimate.
    """
    x = x0.copy()
    for _i in range(max_iter):

        x_prev = x.copy()
        r = local_step_func(x)
        x = global_step_func(x, r)

        delta = (x - x_prev)
        if np.linalg.norm(delta) < tolerance:
            break

    return x
