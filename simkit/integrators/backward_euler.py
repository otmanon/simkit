"""Backward Euler implicit time integrator.

One implicit step advances a second-order system ``M x'' = -grad V(x)`` by
*minimizing* the incremental potential

    E(x) = V(x) + 0.5 / h^2 * (x - x_tilde)^T M (x - x_tilde)

over the next-step position ``x``, where the inertial target is
``x_tilde = x_curr + h v_curr`` and ``v_curr = (x_curr - x_prev) / h``. The
minimization is handed to :func:`~simkit.solvers.newton_solver`, built
automatically from the supplied potential energy / gradient / hessian and the
given Newton solver parameters.

Backward Euler is first-order accurate and unconditionally stable; it adds
numerical damping that grows with the timestep.
"""

from ..solvers import newton_solver
from ..energies.kinetic import (
    be_target,
    kinetic_energy_be,
    kinetic_gradient_be,
    kinetic_hessian_be,
)


def backward_euler(
    x_curr,
    x_prev,
    energy_func,
    gradient_func,
    hessian_func,
    M,
    h,
    tolerance: float = 1e-6,
    max_iter: int = 1,
    do_line_search: bool = True,
    return_info: bool = False,
):
    """Advance one backward-Euler step by an implicit Newton solve.

    Parameters
    ----------
    x_curr : np.ndarray (n, 1)
        Most recent known state (position).
    x_prev : np.ndarray (n, 1)
        State one step earlier. Together with ``x_curr`` it fixes the velocity
        ``(x_curr - x_prev) / h`` and hence the inertial target.
    energy_func : callable
        Potential energy ``V(x) -> float``. The kinetic (inertial) term is
        added internally; do **not** include it here.
    gradient_func : callable
        Potential gradient ``grad V(x) -> (n, 1) np.ndarray``.
    hessian_func : callable
        Potential hessian ``H V(x) -> (n, n) (sparse) matrix``.
    M : (n, n) scipy.sparse matrix
        Mass matrix.
    h : float
        Timestep.
    tolerance : float
        Convergence tolerance forwarded to :func:`newton_solver`.
    max_iter : int
        Maximum Newton iterations forwarded to :func:`newton_solver`.
    do_line_search : bool
        Whether the Newton solve runs a backtracking line search.
    return_info : bool
        If True, also return the solver's per-iteration info dict.

    Returns
    -------
    x_next : np.ndarray (n, 1)
        State after one timestep.
    info : dict
        Solver info, only when ``return_info`` is True.
    """
    def energy(x):
        return energy_func(x) + kinetic_energy_be(x, x_curr, x_prev, M, h)

    def gradient(x):
        return gradient_func(x) + kinetic_gradient_be(x, x_curr, x_prev, M, h)

    def hessian(x):
        return hessian_func(x) + kinetic_hessian_be(M, h)

    # Inertial guess x_tilde = x_curr + h v_curr is a good warm start.
    x0 = be_target(x_curr, x_prev, h)
    return newton_solver(
        x0, energy, gradient, hessian,
        tolerance=tolerance, max_iter=max_iter,
        do_line_search=do_line_search, return_info=return_info,
    )
