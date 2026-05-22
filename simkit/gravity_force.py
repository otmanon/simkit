"""Gravitational body force from a uniform acceleration field."""

import numpy as np

from .massmatrix import massmatrix


def gravity_force(
    X: np.ndarray,
    T: np.ndarray,
    a: float = -9.8,
    rho: float = 1,
) -> np.ndarray:
    """Compute lumped gravitational force on mesh vertices.

    Applies uniform acceleration ``a`` along the second coordinate axis,
    scaled by the lumped mass matrix.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Vertex positions.
    T : np.ndarray (t, s)
        Simplex indices (triangles or tets).
    a : float, optional
        Gravitational acceleration (default ``-9.8``).
    rho : float, optional
        Material density passed to the mass matrix (default 1).

    Returns
    -------
    g : np.ndarray (n, dim)
        Gravitational force vector per vertex.
    """
    dim = X.shape[1]
    # Compute the volume of each triangle
    M = massmatrix(X, T, rho=rho)

    g = np.zeros((X.shape[0], dim))
    g[:, 1] = a
    g = (M @ g)
    return g
