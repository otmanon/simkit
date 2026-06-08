"""Forward (explicit) Euler time integrator.

The cheapest scheme: no implicit solve, no hessian. Given the current position
and velocity it reads the force off the potential gradient and takes one
explicit step of ``M x'' = -grad V(x)``:

    a      = M^{-1} (-grad V(x_curr))
    x_next = x_curr + h v_curr
    v_next = v_curr + h a

The mass matrix is **lumped** (row-summed to a diagonal) so the acceleration is
a single element-wise divide rather than a linear solve -- keeping the step
genuinely cheap, which is the whole point of an explicit method. As an explicit
integrator it is only stable below a (small) critical timestep; above it the
solution blows up. Because the state of an explicit scheme is the
``(position, velocity)`` pair, this integrator both takes and returns velocity.
"""

import numpy as np
import scipy as sp


def forward_euler(x_curr, v_curr, gradient_func, M, h):
    """Advance one explicit forward-Euler step.

    Parameters
    ----------
    x_curr : np.ndarray (n, 1)
        Current state (position).
    v_curr : np.ndarray (n, 1)
        Current velocity.
    gradient_func : callable
        Potential gradient ``grad V(x) -> (n, 1) np.ndarray``. The force is
        ``-grad V(x_curr)``.
    M : (n, n) scipy.sparse matrix or np.ndarray
        Mass matrix. Lumped (row-summed) internally for the acceleration solve.
    h : float
        Timestep.

    Returns
    -------
    x_next : np.ndarray (n, 1)
        State after one timestep.
    v_next : np.ndarray (n, 1)
        Velocity after one timestep.
    """
    x_curr = x_curr.reshape(-1, 1)
    v_curr = v_curr.reshape(-1, 1)

    f = -gradient_func(x_curr).reshape(-1, 1)

    # Lumped (row-summed) mass -> diagonal inverse, no linear solve.
    if sp.sparse.issparse(M):
        m = np.asarray(M.sum(axis=1)).reshape(-1, 1)
    else:
        m = np.asarray(M).sum(axis=1).reshape(-1, 1)
    a = f / m

    x_next = x_curr + h * v_curr
    v_next = v_curr + h * a
    return x_next, v_next
