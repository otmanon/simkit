"""Central-difference Hessian of a callable by finite differences."""

from typing import Callable

import numpy as np

from .gradient_cfd import gradient_cfd


def hessian_cfd(
    phi: Callable[[np.ndarray], np.ndarray], y: np.ndarray, h: float
) -> np.ndarray:
    """Central-difference Hessian of ``phi`` with respect to ``y``.

    Applies :func:`gradient_cfd` twice: first to ``phi``, then to the resulting
    gradient field. If ``phi(y)`` has shape ``(n1, n2, ..., nd)``, the Hessian
    has shape ``(n1, n2, ..., nd, dim(y), dim(y))``.

    Parameters
    ----------
    phi : callable
        Function mapping ``y`` to an array-valued output.
    y : np.ndarray (n,)
        Point at which to evaluate the Hessian.
    h : float
        Central-difference step size.

    Returns
    -------
    H : np.ndarray
        Hessian tensor with two trailing axes of length ``n``.
    """
    def grad_func(p: np.ndarray) -> np.ndarray:
        return gradient_cfd(phi, p, h)

    phi_hess_fd = gradient_cfd(grad_func, y, h)
    return phi_hess_fd
