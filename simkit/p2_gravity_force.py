"""Gravitational body force on a quadratic (P2) mesh.

Mirrors :func:`simkit.gravity_force` but uses the consistent P2 mass matrix so
the body force is distributed correctly over the corner and midpoint nodes.
"""

import numpy as np

from .p2_massmatrix import p2_massmatrix


def p2_gravity_force(
    V2: np.ndarray,
    T2: np.ndarray,
    bary: np.ndarray,
    weights: np.ndarray,
    a: float = -9.8,
    rho: float = 1.0,
) -> np.ndarray:
    """Gravitational force per P2 node from a uniform acceleration field.

    Applies a uniform acceleration ``a`` along the second coordinate axis,
    weighted by the consistent P2 mass matrix.

    Parameters
    ----------
    V2 : np.ndarray (n2, dim)
        Quadratic-mesh vertex positions.
    T2 : np.ndarray (t, n_nodes)
        Quadratic connectivity (6 nodes for triangles, 10 for tets).
    bary : np.ndarray (t, n_quad, dim+1)
        Barycentric cubature points (from :func:`gauss_legendre_quadrature`).
    weights : np.ndarray (t, n_quad)
        Physical cubature weights (from :func:`gauss_legendre_quadrature`).
    a : float, optional
        Gravitational acceleration (default ``-9.8``).
    rho : float or np.ndarray (t, 1), optional
        Mass density (default 1).

    Returns
    -------
    g : np.ndarray (n2, dim)
        Gravitational force per P2 node.
    """
    V2 = np.asarray(V2, dtype=float)
    dim = V2.shape[1]

    M = p2_massmatrix(V2, T2, bary, weights, rho=rho)

    g = np.zeros((V2.shape[0], dim))
    g[:, 1] = a
    g = M @ g
    return g
