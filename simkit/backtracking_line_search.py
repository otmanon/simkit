"""Backtracking line search for descent-direction optimization."""

from typing import Callable, Tuple

import numpy as np


def backtracking_line_search(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    g: np.ndarray,
    dx: np.ndarray,
    alpha: float = 0.01,
    beta: float = 0.5,
    max_iter: int = 100,
    threshold: float = 1e-12,
) -> Tuple[float, np.ndarray, float]:
    """Armijo backtracking line search along a descent direction.

    Implements Algorithm 9.2 from Boyd & Vandenberghe, "Convex Optimization"
    (2004), Chapter 9. Starting from a full step ``t = 1``, shrink ``t`` by a
    factor ``beta`` until the Armijo sufficient-decrease condition holds.

    Parameters
    ----------
    f : callable
        Objective function mapping a point to a scalar.
    x0 : np.ndarray
        Current point.
    g : np.ndarray
        Gradient of ``f`` at ``x0``.
    dx : np.ndarray
        Search direction (assumed to be a descent direction).
    alpha : float, optional
        Armijo sufficient-decrease parameter in ``(0, 0.5]``. Default 0.01.
    beta : float, optional
        Step reduction factor in ``(0, 1)``. Default 0.5.
    max_iter : int, optional
        Maximum number of backtracking steps. Default 100.
    threshold : float, optional
        Slack added to the Armijo test for numerical tolerance. Default 1e-12.

    Returns
    -------
    t : float
        Accepted step size, or 0.0 if no step satisfied the condition.
    x : np.ndarray
        Point ``x0 + t * dx`` (equal to ``x0`` if ``t`` is 0.0).
    fx : float
        Objective value at the returned point.
    """
    assert alpha > 0 and alpha <= 0.5
    assert beta > 0 and beta < 1
    assert np.ndim(x0) == np.ndim(dx)

    t = 1.0
    fx0 = f(x0)
    for _ in range(max_iter):
        x = x0 + t * dx
        fx = f(x)
        # Armijo condition: actual decrease beats predicted linear decrease.
        if fx <= fx0 + alpha * t * (g.T @ dx) + threshold:
            return t, x, fx
        t = beta * t                     # step too large; shrink and retry

    # No acceptable step found: report a zero step at the original point.
    return 0.0, x0, fx0