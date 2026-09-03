"""Local/global (alternating projection) solver -- not yet implemented.

The name is exported so that ``simkit.solvers.local_global`` resolves, but
calling it raises :class:`NotImplementedError`. Use
:func:`~simkit.solvers.block_coord` instead: block coordinate descent
alternating a local and a global step is the same algorithm shape, and it is
implemented and tested.
"""


def local_global(x, local_func, global_func, tolerance=1e-6, max_iter=100):
    """Local/global solver. **Not implemented** -- always raises.

    Parameters
    ----------
    x : (n, 1) np.ndarray
        Initial guess.
    local_func : callable
        Local step.
    global_func : callable
        Global step.
    tolerance : float
        Convergence tolerance.
    max_iter : int
        Maximum number of iterations.

    Raises
    ------
    NotImplementedError
        Always. This function has never had an implementation; it previously
        returned ``x + 1``, which is not a solution to anything.

    See Also
    --------
    simkit.solvers.block_coord : Implemented alternating local/global descent.
    """
    raise NotImplementedError(
        "simkit.solvers.local_global is not implemented. Use "
        "simkit.solvers.block_coord, which performs the same local/global "
        "alternation and is tested."
    )
