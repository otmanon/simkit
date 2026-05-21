"""Generic quadratic energy ``0.5 x^T Q x + b^T x``.

A single-tier global energy with no per-element decomposition. Useful as a
building block (regularizers, springs to targets, linear constraints folded into
a penalty).
"""

import numpy as np
import scipy as sp
from typing import Union

Matrix = Union[np.ndarray, sp.sparse.spmatrix]


def quadratic_energy(x: np.ndarray, Q: Matrix, b: np.ndarray) -> float:
    """Generic quadratic energy.

    Parameters
    ----------
    x : np.ndarray (n, 1)
        System positions.
    Q : np.ndarray or scipy.sparse matrix (n, n)
        Quadratic form matrix.
    b : np.ndarray (n, 1)
        Linear coefficient vector.

    Returns
    -------
    e : float
        Quadratic energy.
    """
    e = 0.5 * x.T @ Q @ x + b.T @ x
    return float(np.asarray(e).item())


def quadratic_gradient(x: np.ndarray, Q: Matrix, b: np.ndarray) -> np.ndarray:
    """Gradient of the quadratic energy w.r.t. ``x``.

    Parameters
    ----------
    x : np.ndarray (n, 1)
        System positions.
    Q : np.ndarray or scipy.sparse matrix (n, n)
        Quadratic form matrix (assumed symmetric).
    b : np.ndarray (n, 1)
        Linear coefficient vector.

    Returns
    -------
    g : np.ndarray (n, 1)
        Energy gradient ``Q x + b``.
    """
    return Q @ x + b


def quadratic_hessian(Q: Matrix) -> Matrix:
    """Hessian of the quadratic energy w.r.t. ``x``.

    Provided for interface consistency with other energies.

    Parameters
    ----------
    Q : np.ndarray or scipy.sparse matrix (n, n)
        Quadratic form matrix.

    Returns
    -------
    Q : np.ndarray or scipy.sparse matrix (n, n)
        The Hessian, equal to ``Q``.
    """
    return Q