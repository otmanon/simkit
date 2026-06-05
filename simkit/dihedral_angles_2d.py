"""2D dihedral (hinge) angles and their derivatives.

The 2D member of the dihedral-angle family: a curve in the plane folds at an
interior **vertex** shared by two edges. Each element is a vertex triple
``(A, B, C)`` and the dihedral angle is the signed turn at ``B`` between edges
``B->A`` and ``B->C``, via ``atan2`` of the 2D cross and dot products (signed,
full circle).

Element derivatives are returned in the compact per-hinge layout with columns
ordered ``[Ax, Ay, Bx, By, Cx, Cy]`` (matching ``kron(wedge_map(C, n), eye(2))``).
The dimension-dispatching front end lives in :mod:`simkit.dihedral_angles`; the
3D sibling is :mod:`simkit.dihedral_angles_3d`.
"""

import numpy as np


def dihedral_angles_2d(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Signed 2D dihedral (hinge) angle at each interior vertex ``B``.

    Parameters
    ----------
    X : np.ndarray (n, 2)
        Vertex positions.
    C : np.ndarray (m, 3)
        Hinge connectivity; each row is ``(A, B, C)`` with the angle at ``B``.

    Returns
    -------
    theta : np.ndarray (m, 1)
        Signed angle between edges ``B-A`` and ``C-B``, via ``atan2(s, c)``.
    """
    A = X[C[:, 0]]
    B = X[C[:, 1]]
    Cc = X[C[:, 2]]

    v1 = B - A
    v2 = Cc - B

    c = v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1]   # dot product
    s = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]   # cross product (z component)
    return np.arctan2(s, c).reshape(-1, 1)


def dihedral_angles_2d_velocities(X: np.ndarray, V: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Time derivative of each 2D dihedral angle given vertex velocities.

    Uses the difference of edge angular velocities about the hinge vertex ``B``.

    Parameters
    ----------
    X : np.ndarray (n, 2)
        Vertex positions.
    V : np.ndarray (n, 2)
        Vertex velocities.
    C : np.ndarray (m, 3)
        Hinge connectivity; each row is ``(A, B, C)``.

    Returns
    -------
    theta_dot : np.ndarray (m, 1)
        Angular velocity of each hinge angle.
    """
    A = X[C[:, 0]]
    B = X[C[:, 1]]
    Cc = X[C[:, 2]]
    VA = V[C[:, 0]]
    VB = V[C[:, 1]]
    VC = V[C[:, 2]]

    e1 = A - B          # B -> A
    e2 = Cc - B         # B -> C
    de1 = VA - VB
    de2 = VC - VB

    L1 = np.sum(e1 * e1, axis=1)
    L2 = np.sum(e2 * e2, axis=1)

    # angular velocity of each edge about B: omega = e_perp . de / ||e||^2
    w1 = (-e1[:, 1] * de1[:, 0] + e1[:, 0] * de1[:, 1]) / L1
    w2 = (-e2[:, 1] * de2[:, 0] + e2[:, 0] * de2[:, 1]) / L2
    return (w2 - w1).reshape(-1, 1)


def dihedral_angles_2d_gradient_element(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Per-hinge gradient ``d theta / d [Ax, Ay, Bx, By, Cx, Cy]``.

    Closed-form derivatives of ``atan2`` w.r.t. the six hinge coordinates.
    Denominators are clamped with a small ``epsilon`` to avoid division by zero.

    Parameters
    ----------
    X : np.ndarray (n, 2)
        Vertex positions.
    C : np.ndarray (m, 3)
        Hinge connectivity; each row is ``(A, B, C)``.

    Returns
    -------
    J : np.ndarray (m, 6)
        Per-hinge gradient rows in ``[Ax, Ay, Bx, By, Cx, Cy]`` order.
    """
    Ax, Ay = X[C[:, 0], 0], X[C[:, 0], 1]
    Bx, By = X[C[:, 1], 0], X[C[:, 1], 1]
    Cx, Cy = X[C[:, 2], 0], X[C[:, 2], 1]

    epsilon = 1e-6
    dist_AB_sq = np.maximum(Ax**2 - 2 * Ax * Bx + Ay**2 - 2 * Ay * By + Bx**2 + By**2, epsilon)
    dist_BC_sq = np.maximum(Bx**2 - 2 * Bx * Cx + By**2 - 2 * By * Cy + Cx**2 + Cy**2, epsilon)
    if np.any(dist_AB_sq == 0):
        print("WARNING:dist_AB_sq is 0, leading to a divide by zero in the dihedral 2d gradient")
    if np.any(dist_BC_sq == 0):
        print("WARNING: dist_BC_sq is 0, leading to a divide by zero in the dihedral 2d gradient")

    J = np.array([(Ay - By) / (Ax**2 - 2 * Ax * Bx + Ay**2 - 2 * Ay * By + Bx**2 + By**2),
                  (-Ax + Bx) / (Ax**2 - 2 * Ax * Bx + Ay**2 - 2 * Ay * By + Bx**2 + By**2),
                  (-(Ay - Cy) * ((Ax - Bx) * (Bx - Cx) + (Ay - By) * (By - Cy)) - ((Ax - Bx) * (By - Cy) - (Ay - By) * (Bx - Cx)) * (Ax - 2 * Bx + Cx)) / (((Ax - Bx) * (Bx - Cx) + (Ay - By) * (By - Cy))**2 + ((Ax - Bx) * (By - Cy) - (Ay - By) * (Bx - Cx))**2),
                  ((Ax - Cx) * ((Ax - Bx) * (Bx - Cx) + (Ay - By) * (By - Cy)) - ((Ax - Bx) * (By - Cy) - (Ay - By) * (Bx - Cx)) * (Ay - 2 * By + Cy)) / (((Ax - Bx) * (Bx - Cx) + (Ay - By) * (By - Cy))**2 + ((Ax - Bx) * (By - Cy) - (Ay - By) * (Bx - Cx))**2),
                  (By - Cy) / (Bx**2 - 2 * Bx * Cx + By**2 - 2 * By * Cy + Cx**2 + Cy**2),
                  (-Bx + Cx) / (Bx**2 - 2 * Bx * Cx + By**2 - 2 * By * Cy + Cx**2 + Cy**2)]).T
    return J


def dihedral_angles_2d_hessian_element(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Per-hinge Hessian ``d^2 theta / d [Ax, Ay, Bx, By, Cx, Cy]^2``.

    Closed-form second derivatives of the 2D dihedral angle.

    Parameters
    ----------
    X : np.ndarray (n, 2)
        Vertex positions.
    C : np.ndarray (m, 3)
        Hinge connectivity; each row is ``(A, B, C)``.

    Returns
    -------
    blocks : np.ndarray (m, 6, 6)
        Per-hinge Hessian blocks in ``[Ax, Ay, Bx, By, Cx, Cy]`` order.
    """
    Ax, Ay = X[C[:, 0], 0], X[C[:, 0], 1]
    Bx, By = X[C[:, 1], 0], X[C[:, 1], 1]
    Cx, Cy = X[C[:, 2], 0], X[C[:, 2], 1]

    epsilon = 0
    dist_AB_sq = np.maximum(Ax**2 - 2 * Ax * Bx + Ay**2 - 2 * Ay * By + Bx**2 + By**2, epsilon)
    dist_BC_sq = np.maximum(Bx**2 - 2 * Bx * Cx + By**2 - 2 * By * Cy + Cx**2 + Cy**2, epsilon)

    dot_prod_AB_BC = (Ax - Bx) * (Bx - Cx) + (Ay - By) * (By - Cy)
    cross_prod_AB_BC = (Ax - Bx) * (By - Cy) - (Ay - By) * (Bx - Cx)
    complex_denom = np.maximum(dot_prod_AB_BC**2 + cross_prod_AB_BC**2, epsilon)

    if np.any(dist_AB_sq == 0):
        print("WARNING: dist_AB_sq is 0, leading to a divide by zero in the dihedral 2d hessian")
    if np.any(dist_BC_sq == 0):
        print("WARNING: dist_BC_sq is 0, leading to a divide by zero in the dihedral 2d hessian")
    if np.any(complex_denom == 0):
        print("WARNING: complex_denom is 0, leading to a divide by zero in the dihedral 2d hessian")

    denom_AB_4th = np.maximum(dist_AB_sq**2, epsilon)
    denom_BC_4th = np.maximum(dist_BC_sq**2, epsilon)

    Z = np.zeros((Ax.shape[0]))
    H = np.array([
        [-2 * (Ax - Bx) * (Ay - By) / dist_AB_sq**2,
         (-Ax**2 + 2 * Ax * Bx - Ay**2 + 2 * Ay * By - Bx**2 - By**2 + 2 * (Ax - Bx)**2) / dist_AB_sq**2,
         2 * (Ax * Ay - Ax * By - Ay * Bx + Bx * By) / denom_AB_4th,
         (-Ax**2 + 2 * Ax * Bx + Ay**2 - 2 * Ay * By - Bx**2 + By**2) / denom_AB_4th,
         Z,
         Z],
        [(Ax**2 - 2 * Ax * Bx + Ay**2 - 2 * Ay * By + Bx**2 + By**2 - 2 * (Ay - By)**2) / dist_AB_sq**2,
         2 * (Ax - Bx) * (Ay - By) / dist_AB_sq**2,
         (-Ax**2 + 2 * Ax * Bx + Ay**2 - 2 * Ay * By - Bx**2 + By**2) / denom_AB_4th,
         2 * (-Ax * Ay + Ax * By + Ay * Bx - Bx * By) / denom_AB_4th,
         Z,
         Z],
        [2 * (Ax - Bx) * (Ay - By) / dist_AB_sq**2,
         (Ax**2 - 2 * Ax * Bx + Ay**2 - 2 * Ay * By + Bx**2 + By**2 - 2 * (Ax - Bx)**2) / dist_AB_sq**2,
         2 * (((Ax - Bx) * (By - Cy) - (Ay - By) * (Bx - Cx)) * complex_denom - ((Ay - Cy) * dot_prod_AB_BC + cross_prod_AB_BC * (Ax - 2 * Bx + Cx)) * ((Ay - Cy) * cross_prod_AB_BC - dot_prod_AB_BC * (Ax - 2 * Bx + Cx))) / complex_denom**2,
         (2 * ((Ax - Cx) * dot_prod_AB_BC - cross_prod_AB_BC * (Ay - 2 * By + Cy)) * ((Ay - Cy) * cross_prod_AB_BC - dot_prod_AB_BC * (Ax - 2 * Bx + Cx)) + ((Ax - Cx) * (Ax - 2 * Bx + Cx) + (Ay - Cy) * (Ay - 2 * By + Cy)) * complex_denom) / complex_denom**2,
         -2 * (Bx - Cx) * (By - Cy) / dist_BC_sq**2,
         (-Bx**2 + 2 * Bx * Cx - By**2 + 2 * By * Cy - Cx**2 - Cy**2 + 2 * (Bx - Cx)**2) / dist_BC_sq**2],
        [(-Ax**2 + 2 * Ax * Bx - Ay**2 + 2 * Ay * By - Bx**2 - By**2 + 2 * (Ay - By)**2) / dist_AB_sq**2,
         -2 * (Ax - Bx) * (Ay - By) / dist_AB_sq**2,
         (2 * ((Ax - Cx) * cross_prod_AB_BC + dot_prod_AB_BC * (Ay - 2 * By + Cy)) * ((Ay - Cy) * dot_prod_AB_BC + cross_prod_AB_BC * (Ax - 2 * Bx + Cx)) - ((Ax - Cx) * (Ax - 2 * Bx + Cx) + (Ay - Cy) * (Ay - 2 * By + Cy)) * complex_denom) / complex_denom**2,
         2 * (cross_prod_AB_BC * complex_denom - ((Ax - Cx) * dot_prod_AB_BC - cross_prod_AB_BC * (Ay - 2 * By + Cy)) * ((Ax - Cx) * cross_prod_AB_BC + dot_prod_AB_BC * (Ay - 2 * By + Cy))) / complex_denom**2,
         (Bx**2 - 2 * Bx * Cx + By**2 - 2 * By * Cy + Cx**2 + Cy**2 - 2 * (By - Cy)**2) / dist_BC_sq**2,
         2 * (Bx - Cx) * (By - Cy) / dist_BC_sq**2],
        [Z,
         Z,
         2 * (-Bx * By + Bx * Cy + By * Cx - Cx * Cy) / denom_BC_4th,
         (Bx**2 - 2 * Bx * Cx - By**2 + 2 * By * Cy + Cx**2 - Cy**2) / denom_BC_4th,
         2 * (Bx - Cx) * (By - Cy) / dist_BC_sq**2,
         (Bx**2 - 2 * Bx * Cx + By**2 - 2 * By * Cy + Cx**2 + Cy**2 - 2 * (Bx - Cx)**2) / dist_BC_sq**2],
        [Z,
         Z,
         (Bx**2 - 2 * Bx * Cx - By**2 + 2 * By * Cy + Cx**2 - Cy**2) / denom_BC_4th,
         2 * (Bx * By - Bx * Cy - By * Cx + Cx * Cy) / denom_BC_4th,
         (-Bx**2 + 2 * Bx * Cx - By**2 + 2 * By * Cy - Cx**2 - Cy**2 + 2 * (By - Cy)**2) / dist_BC_sq**2,
         -2 * (Bx - Cx) * (By - Cy) / dist_BC_sq**2]
    ])

    return H.transpose(2, 0, 1)
