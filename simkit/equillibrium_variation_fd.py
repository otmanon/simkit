"""Variation of an equilibrium response via finite differences."""

import numpy as np
import scipy as sp

from .variation_fd import variation_fd


def equillibrium_variation_fd(U0, phi_j, H_func, b_func, epsilon=1e-5):
    r"""Directional derivative of an equilibrium response.

    Consider an equilibrium relation parameterised by a point ``x``,

    .. math::

        H(x)\,\phi_i = b(x),

    where ``H(x)`` is a (symmetric) system matrix and ``b(x)`` a right-hand
    side, so the response is ``phi_i = H(x)^{-1} b(x)``. Differentiating both
    sides along a direction ``phi_j`` gives

    .. math::

        \frac{\partial H}{\partial x}\!:\!\phi_j \; \phi_i
        + H \,\frac{\partial \phi_i}{\partial \phi_j}
        = \frac{\partial b}{\partial x}\!:\!\phi_j,

    so the variation of the response solves

    .. math::

        \frac{\partial \phi_i}{\partial \phi_j}
        = H^{-1}\!\left(
            \frac{\partial b}{\partial x}\!:\!\phi_j
            - \frac{\partial H}{\partial x}\!:\!\phi_j \; \phi_i
        \right).

    The directional derivatives ``dH/dx : phi_j`` and ``db/dx : phi_j`` are
    estimated with :func:`simkit.variation_fd.variation_fd` (a central
    difference), while ``H^{-1}`` is applied with a direct solve --
    :func:`scipy.sparse.linalg.spsolve` when ``H(U0)`` is sparse, otherwise
    :func:`numpy.linalg.solve`. The equilibrium response is solved directly as
    ``phi_i = H(U0)^{-1} b(U0)``.

    A constant right-hand side needs no special handling: ``db/dx : phi_j`` is
    then estimated as (approximately) zero and the corresponding term drops out.

    Parameters
    ----------
    U0 : np.ndarray
        Point at which to evaluate the equilibrium and its variation.
    phi_j : np.ndarray
        Direction along which to differentiate, same shape as ``U0``.
    H_func : callable
        Maps a point ``x`` to the system matrix ``H_func(x)``. Must return a
        square, invertible matrix (dense ``np.ndarray`` or a scipy sparse
        matrix) and support evaluation at ``U0 +/- epsilon * phi_j``.
    b_func : callable
        Maps a point ``x`` to the right-hand side ``b_func(x)``, used both to
        solve for the response ``phi_i`` and (via finite differences) for the
        ``db/dx`` term.
    epsilon : float, optional
        Finite-difference step size passed to
        :func:`~simkit.variation_fd.variation_fd`. Default ``1e-5``.

    Returns
    -------
    dphi : np.ndarray
        Estimate of ``dphi_i/dphi_j``, the variation of the equilibrium
        response along ``phi_j``.

    See Also
    --------
    simkit.variation_fd.variation_fd : Central-difference Hessian variation.
    """
    H0 = H_func(U0)
    b0 = b_func(U0)

    solve = sp.sparse.linalg.spsolve if sp.sparse.issparse(H0) else np.linalg.solve

    phi_i = solve(H0, b0)

    hessian_variation = variation_fd(U0, phi_j, H_func, epsilon=epsilon)
    db = variation_fd(U0, phi_j, b_func, epsilon=epsilon)

    dphi = solve(H0, db - hessian_variation @ phi_i)

    return dphi
