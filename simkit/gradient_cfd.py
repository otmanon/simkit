"""Central-difference gradient of a callable by finite differences."""

from typing import Callable

import numpy as np


def gradient_cfd(
    phi: Callable[[np.ndarray], np.ndarray], y: np.ndarray, h: float
) -> np.ndarray:
    """Central-difference gradient of ``phi`` with respect to ``y``.

    If ``phi(y)`` has shape ``(dim_1, ..., dim_d)``, the returned gradient has
    shape ``(dim_1, ..., dim_d, dim(y))``.

    Parameters
    ----------
    phi : callable
        Function mapping ``y`` to an array-valued output.
    y : np.ndarray (n,)
        Point at which to evaluate the gradient.
    h : float
        Central-difference step size.

    Returns
    -------
    g : np.ndarray
        Gradient tensor with an extra trailing axis of length ``n``.
    """
    y0 = y.copy()

    phi0 = phi(y)
    g = np.zeros(phi0.shape + (y.shape[0],))
    for i in range(y.shape[0]):
        yib = y0.copy()
        yif = y0.copy()
        yib[i] -= h
        yif[i] += h
        phi_b = phi(yib)
        phi_f = phi(yif)

        g[..., i] = (phi_f - phi_b) / (2 * h)  # the dots mean only inde

    return g
