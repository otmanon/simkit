"""Gaussian radial basis function evaluation on point sets."""

import numpy as np

from .pairwise_distance import pairwise_distance


def gaussian_rbf(X: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Evaluate a Gaussian RBF at points ``X``.

    Uses the standard Gaussian radial basis
    ``phi(r) = exp(-0.5 * sigma^2 * r^2)`` as defined at
    https://en.wikipedia.org/wiki/Radial_basis_function, where ``sigma`` is
    stored in the last column of ``p``.

    Parameters
    ----------
    X : np.ndarray (n, dx)
        Evaluation points.
    p : np.ndarray (m, dx + 1)
        RBF centers in columns ``0:dx`` and scale ``sigma`` in column ``dx``.

    Returns
    -------
    phi : np.ndarray (n, m)
        RBF values at each point for each center/scale pair.
    """
    dx = X.shape[1]
    px = p[:, 0:dx]
    pg = p[:, dx]

    r = pairwise_distance(X, px)

    phi = np.exp(-0.5 * (pg**2) * (r**2))

    return phi
