"""Hessian of 2D hinge angles and assembly into the global sparse matrix.

Provides closed-form per-hinge ``(6, 6)`` blocks (coordinate order
``[Ax, Ay, Bx, By, Cx, Cy]``) and scatters weighted blocks onto the global
``(2*n, 2*n)`` Hessian, mirroring :mod:`hinge_jacobian`.
"""

# import numpy as np
# import scipy as sp
# def hinge_hessian_compact(X, H):
#     Ax, Ay = X[H[:,0], 0], X[H[:,0], 1]
#     Bx, By = X[H[:,1], 0], X[H[:,1], 1]
#     Cx, Cy = X[H[:,2], 0], X[H[:,2], 1]
#
#     # Numerical regularization parameter
#     epsilon = 0
#
#
#     # Precompute common denominators with regularization
#     dist_AB_sq = np.maximum(Ax**2 - 2*Ax*Bx + Ay**2 - 2*Ay*By + Bx**2 + By**2, epsilon)
#     dist_BC_sq = np.maximum(Bx**2 - 2*Bx*Cx + By**2 - 2*By*Cy + Cx**2 + Cy**2, epsilon)
#
#     dot_prod_AB_BC = (Ax - Bx)*(Bx - Cx) + (Ay - By)*(By - Cy)
#     cross_prod_AB_BC = (Ax - Bx)*(By - Cy) - (Ay - By)*(Bx - Cx)
#     complex_denom = np.maximum(dot_prod_AB_BC**2 + cross_prod_AB_BC**2, epsilon)
#
#     if np.any(dist_AB_sq == 0):
#         print("WARNING: dist_AB_sq is 0, leading to a divide by zero in the hinge hessian")
#     if np.any(dist_BC_sq == 0):
#         print("WARNING: dist_BC_sq is 0, leading to a divide by zero in the hinge hessian")
#     if np.any(complex_denom == 0):
#         print("WARNING: complex_denom is 0, leading to a divide by zero in the hinge hessian")
#
#     # Fourth-order denominators
#     denom_AB_4th = np.maximum(dist_AB_sq**2, epsilon)
#     denom_BC_4th = np.maximum(dist_BC_sq**2, epsilon)
#
#     Z = np.zeros((Ax.shape[0]))
#     H = np.array([
#         [-2*(Ax - Bx)*(Ay - By)/dist_AB_sq**2,
#          (-Ax**2 + 2*Ax*Bx - Ay**2 + 2*Ay*By - Bx**2 - By**2 + 2*(Ax - Bx)**2)/dist_AB_sq**2,
#          2*(Ax*Ay - Ax*By - Ay*Bx + Bx*By)/denom_AB_4th,
#          (-Ax**2 + 2*Ax*Bx + Ay**2 - 2*Ay*By - Bx**2 + By**2)/denom_AB_4th,
#          Z,
#          Z],
#         [(Ax**2 - 2*Ax*Bx + Ay**2 - 2*Ay*By + Bx**2 + By**2 - 2*(Ay - By)**2)/dist_AB_sq**2,
#          2*(Ax - Bx)*(Ay - By)/dist_AB_sq**2,
#          (-Ax**2 + 2*Ax*Bx + Ay**2 - 2*Ay*By - Bx**2 + By**2)/denom_AB_4th,
#          2*(-Ax*Ay + Ax*By + Ay*Bx - Bx*By)/denom_AB_4th,
#          Z,
#          Z],
#         [2*(Ax - Bx)*(Ay - By)/dist_AB_sq**2,
#          (Ax**2 - 2*Ax*Bx + Ay**2 - 2*Ay*By + Bx**2 + By**2 - 2*(Ax - Bx)**2)/dist_AB_sq**2,
#          2*(((Ax - Bx)*(By - Cy) - (Ay - By)*(Bx - Cx))*complex_denom - ((Ay - Cy)*dot_prod_AB_BC + cross_prod_AB_BC*(Ax - 2*Bx + Cx))*((Ay - Cy)*cross_prod_AB_BC - dot_prod_AB_BC*(Ax - 2*Bx + Cx)))/complex_denom**2,
#          (2*((Ax - Cx)*dot_prod_AB_BC - cross_prod_AB_BC*(Ay - 2*By + Cy))*((Ay - Cy)*cross_prod_AB_BC - dot_prod_AB_BC*(Ax - 2*Bx + Cx)) + ((Ax - Cx)*(Ax - 2*Bx + Cx) + (Ay - Cy)*(Ay - 2*By + Cy))*complex_denom)/complex_denom**2,
#          -2*(Bx - Cx)*(By - Cy)/dist_BC_sq**2,
#          (-Bx**2 + 2*Bx*Cx - By**2 + 2*By*Cy - Cx**2 - Cy**2 + 2*(Bx - Cx)**2)/dist_BC_sq**2],
#         [(-Ax**2 + 2*Ax*Bx - Ay**2 + 2*Ay*By - Bx**2 - By**2 + 2*(Ay - By)**2)/dist_AB_sq**2,
#          -2*(Ax - Bx)*(Ay - By)/dist_AB_sq**2,
#          (2*((Ax - Cx)*cross_prod_AB_BC + dot_prod_AB_BC*(Ay - 2*By + Cy))*((Ay - Cy)*dot_prod_AB_BC + cross_prod_AB_BC*(Ax - 2*Bx + Cx)) - ((Ax - Cx)*(Ax - 2*Bx + Cx) + (Ay - Cy)*(Ay - 2*By + Cy))*complex_denom)/complex_denom**2,
#          2*(cross_prod_AB_BC*complex_denom - ((Ax - Cx)*dot_prod_AB_BC - cross_prod_AB_BC*(Ay - 2*By + Cy))*((Ax - Cx)*cross_prod_AB_BC + dot_prod_AB_BC*(Ay - 2*By + Cy)))/complex_denom**2,
#          (Bx**2 - 2*Bx*Cx + By**2 - 2*By*Cy + Cx**2 + Cy**2 - 2*(By - Cy)**2)/dist_BC_sq**2,
#          2*(Bx - Cx)*(By - Cy)/dist_BC_sq**2],
#         [Z,
#          Z,
#          2*(-Bx*By + Bx*Cy + By*Cx - Cx*Cy)/denom_BC_4th,
#          (Bx**2 - 2*Bx*Cx - By**2 + 2*By*Cy + Cx**2 - Cy**2)/denom_BC_4th,
#          2*(Bx - Cx)*(By - Cy)/dist_BC_sq**2,
#          (Bx**2 - 2*Bx*Cx + By**2 - 2*By*Cy + Cx**2 + Cy**2 - 2*(Bx - Cx)**2)/dist_BC_sq**2],
#         [Z,
#          Z,
#          (Bx**2 - 2*Bx*Cx - By**2 + 2*By*Cy + Cx**2 - Cy**2)/denom_BC_4th,
#          2*(Bx*By - Bx*Cy - By*Cx + Cx*Cy)/denom_BC_4th,
#          (-Bx**2 + 2*Bx*Cx - By**2 + 2*By*Cy - Cx**2 - Cy**2 + 2*(By - Cy)**2)/dist_BC_sq**2,
#          -2*(Bx - Cx)*(By - Cy)/dist_BC_sq**2]
#     ])
#
#     return H.transpose(2, 0, 1)


import numpy as np
import scipy as sp


def hinge_hessian_compact(X: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Per-hinge Hessian blocks ``d^2 theta / d [Ax, Ay, Bx, By, Cx, Cy]^2``.

    Closed-form second derivatives of the hinge angle. Denominators use
    ``np.maximum(..., epsilon)`` with ``epsilon = 0`` (warnings on zero).

    Parameters
    ----------
    X : np.ndarray (n, 2)
        Vertex positions.
    H : np.ndarray (J, 3)
        Hinge connectivity; each row is ``(A, B, C)``.

    Returns
    -------
    blocks : np.ndarray (J, 6, 6)
        Per-hinge Hessian blocks in ``[Ax, Ay, Bx, By, Cx, Cy]`` order.
    """
    Ax, Ay = X[H[:, 0], 0], X[H[:, 0], 1]
    Bx, By = X[H[:, 1], 0], X[H[:, 1], 1]
    Cx, Cy = X[H[:, 2], 0], X[H[:, 2], 1]

    # Numerical regularization parameter
    epsilon = 0

    # Precompute common denominators with regularization
    dist_AB_sq = np.maximum(Ax**2 - 2 * Ax * Bx + Ay**2 - 2 * Ay * By + Bx**2 + By**2, epsilon)
    dist_BC_sq = np.maximum(Bx**2 - 2 * Bx * Cx + By**2 - 2 * By * Cy + Cx**2 + Cy**2, epsilon)

    dot_prod_AB_BC = (Ax - Bx) * (Bx - Cx) + (Ay - By) * (By - Cy)
    cross_prod_AB_BC = (Ax - Bx) * (By - Cy) - (Ay - By) * (Bx - Cx)
    complex_denom = np.maximum(dot_prod_AB_BC**2 + cross_prod_AB_BC**2, epsilon)

    if np.any(dist_AB_sq == 0):
        print("WARNING: dist_AB_sq is 0, leading to a divide by zero in the hinge hessian")
    if np.any(dist_BC_sq == 0):
        print("WARNING: dist_BC_sq is 0, leading to a divide by zero in the hinge hessian")
    if np.any(complex_denom == 0):
        print("WARNING: complex_denom is 0, leading to a divide by zero in the hinge hessian")

    # Fourth-order denominators
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


def hinge_hessian(
    X: np.ndarray, H: np.ndarray, blocks: np.ndarray
) -> sp.sparse.csc_matrix:
    """Scatter per-hinge Hessian blocks into a global sparse matrix.

    Mirrors :func:`hinge_jacobian` / :func:`hinge_jacobian_compact`: the compact
    routine produces the per-hinge ``(6, 6)`` blocks and this routine scatters
    them onto the global vertex DOFs and sums. The compact blocks are already
    ordered ``[Ax, Ay, Bx, By, Cx, Cy]`` (the same convention as the Jacobian),
    so the scatter uses the identical index map and no reordering is required.

    Any per-hinge weighting (e.g. multiplying by ``denergy_dtheta``) or PSD
    projection is the caller's responsibility and should be baked into
    ``blocks`` before calling.

    Parameters
    ----------
    X : np.ndarray (n, 2)
        Vertex positions.
    H : np.ndarray (J, 3)
        Hinge connectivity (vertex triples).
    blocks : np.ndarray (J, 6, 6)
        Per-hinge Hessian blocks to scatter, typically
        :func:`hinge_hessian_compact` after weighting/projection.

    Returns
    -------
    Hglobal : scipy.sparse.csc_matrix (2*n, 2*n)
        Assembled global hinge Hessian.
    """
    blocks = np.asarray(blocks)

    offset = np.array([0, 1, 0, 1, 0, 1])
    cols = np.repeat(H * 2, 2, axis=1) + offset          # (J, 6)
    cols2 = np.repeat(cols[:, None, :], 6, axis=1)       # (J, 6, 6)
    rows = cols2.transpose(0, 2, 1)

    Hglobal = sp.sparse.coo_matrix(
        (blocks.ravel(), (rows.ravel(), cols2.ravel())),
        shape=(2 * X.shape[0], 2 * X.shape[0]),
    )
    return Hglobal.tocsc()
