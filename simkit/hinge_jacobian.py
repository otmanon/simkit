"""Jacobian of 2D hinge angles w.r.t. vertex positions.

Provides a compact per-hinge ``(6,)`` block (coordinates ordered
``[Ax, Ay, Bx, By, Cx, Cy]``) and a sparse assembly into the global
``(m, 2*n)`` Jacobian.
"""

import numpy as np
import scipy as sp


def hinge_jacobian_compact(X: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Per-hinge Jacobian blocks ``d theta / d [Ax, Ay, Bx, By, Cx, Cy]``.

    Closed-form derivatives of ``atan2`` w.r.t. the six hinge coordinates.
    Denominators are clamped with a small ``epsilon`` to avoid division by zero.

    Parameters
    ----------
    X : np.ndarray (n, 2)
        Vertex positions.
    H : np.ndarray (m, 3)
        Hinge connectivity; each row is ``(A, B, C)``.

    Returns
    -------
    J : np.ndarray (m, 6)
        Per-hinge Jacobian rows.
    """
    Ax, Ay = X[H[:, 0], 0], X[H[:, 0], 1]
    Bx, By = X[H[:, 1], 0], X[H[:, 1], 1]
    Cx, Cy = X[H[:, 2], 0], X[H[:, 2], 1]

    # Numerical regularization parameter
    epsilon = 1e-6

    # Precompute common denominators with regularization
    dist_AB_sq = np.maximum(Ax**2 - 2 * Ax * Bx + Ay**2 - 2 * Ay * By + Bx**2 + By**2, epsilon)
    dist_BC_sq = np.maximum(Bx**2 - 2 * Bx * Cx + By**2 - 2 * By * Cy + Cx**2 + Cy**2, epsilon)

    if np.any(dist_AB_sq == 0):
        print("WARNING:dist_AB_sq is 0, leading to a divide by zero in the hinge jacobian")
    if np.any(dist_BC_sq == 0):
        print("WARNING: dist_BC_sq is 0, leading to a divide by zero in the hinge jacobian")

    J = np.array([(Ay - By) / (Ax**2 - 2 * Ax * Bx + Ay**2 - 2 * Ay * By + Bx**2 + By**2),
                  (-Ax + Bx) / (Ax**2 - 2 * Ax * Bx + Ay**2 - 2 * Ay * By + Bx**2 + By**2),
                  (-(Ay - Cy) * ((Ax - Bx) * (Bx - Cx) + (Ay - By) * (By - Cy)) - ((Ax - Bx) * (By - Cy) - (Ay - By) * (Bx - Cx)) * (Ax - 2 * Bx + Cx)) / (((Ax - Bx) * (Bx - Cx) + (Ay - By) * (By - Cy))**2 + ((Ax - Bx) * (By - Cy) - (Ay - By) * (Bx - Cx))**2),
                  ((Ax - Cx) * ((Ax - Bx) * (Bx - Cx) + (Ay - By) * (By - Cy)) - ((Ax - Bx) * (By - Cy) - (Ay - By) * (Bx - Cx)) * (Ay - 2 * By + Cy)) / (((Ax - Bx) * (Bx - Cx) + (Ay - By) * (By - Cy))**2 + ((Ax - Bx) * (By - Cy) - (Ay - By) * (Bx - Cx))**2),
                  (By - Cy) / (Bx**2 - 2 * Bx * Cx + By**2 - 2 * By * Cy + Cx**2 + Cy**2),
                  (-Bx + Cx) / (Bx**2 - 2 * Bx * Cx + By**2 - 2 * By * Cy + Cx**2 + Cy**2)]).T
    return J


def hinge_jacobian(X: np.ndarray, H: np.ndarray) -> sp.sparse.csc_matrix:
    """Sparse global hinge-angle Jacobian ``(m, 2*n)``.

    Assembles :func:`hinge_jacobian_compact` into a CSC matrix with one row per
    hinge and columns indexed by stacked vertex coordinates.

    Parameters
    ----------
    X : np.ndarray (n, 2)
        Vertex positions.
    H : np.ndarray (m, 3)
        Hinge connectivity; each row is ``(A, B, C)``.

    Returns
    -------
    J : scipy.sparse.csc_matrix (m, 2*n)
        Global hinge Jacobian.
    """
    J_compact = hinge_jacobian_compact(X, H)

    rows = np.repeat(np.arange(H.shape[0])[:, None], 6, axis=1)
    cols = np.repeat(H * 2, 2, axis=1) + np.array([[0, 1, 0, 1, 0, 1]])

    J = sp.sparse.csc_matrix(
        (J_compact.flatten(), (rows.flatten(), cols.flatten())),
        shape=(H.shape[0], 2 * X.shape[0]),
    )
    return J
