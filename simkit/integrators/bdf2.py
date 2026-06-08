"""BDF2 implicit time integrator.

A second-order constant-step backward differentiation formula. Like backward
Euler it advances ``M x'' = -grad V(x)`` by minimizing an incremental potential

    E(x) = V(x) + 0.5 * c / h^2 * (x - x_tilde)^T M (x - x_tilde),    c = (3/2)^2

but with the BDF2 inertial target

    x_tilde = (4/3) x_curr - (1/3) x_prev + (8h/9) v_curr - (2h/9) v_prev

whose two velocities are reconstructed from the position history by the
second-order backward difference. It therefore consumes two more position
levels than backward Euler (``x_prev2``, ``x_prev3``). The minimization is
handed to :func:`~simkit.solvers.newton_solver`, built automatically.

BDF2 is second-order accurate and stable, with far less numerical damping than
backward Euler. Bootstrap the first step with :func:`backward_euler` (there is
no position history yet for the backward difference).
"""

from ..solvers import newton_solver
from ..energies.kinetic import (
    bdf2_target,
    kinetic_energy_bdf2,
    kinetic_gradient_bdf2,
    kinetic_hessian_bdf2,
)


def bdf2(
    x_curr,
    x_prev,
    x_prev2,
    x_prev3,
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
    """Advance one BDF2 step by an implicit Newton solve.

    Parameters
    ----------
    x_curr : np.ndarray (n, 1)
        Most recent known state (position).
    x_prev : np.ndarray (n, 1)
        State one step earlier.
    x_prev2 : np.ndarray (n, 1)
        State two steps earlier.
    x_prev3 : np.ndarray (n, 1)
        State three steps earlier. The four levels reconstruct the two
        velocities the BDF2 inertial target needs.
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
        return energy_func(x) + kinetic_energy_bdf2(x, x_curr, x_prev, x_prev2, x_prev3, M, h)

    def gradient(x):
        return gradient_func(x) + kinetic_gradient_bdf2(x, x_curr, x_prev, x_prev2, x_prev3, M, h)

    def hessian(x):
        return hessian_func(x) + kinetic_hessian_bdf2(M, h)

    # The BDF2 inertial target is a good warm start.
    x0 = bdf2_target(x_curr, x_prev, x_prev2, x_prev3, h)
    return newton_solver(
        x0, energy, gradient, hessian,
        tolerance=tolerance, max_iter=max_iter,
        do_line_search=do_line_search, return_info=return_info,
    )
