"""Dihedral hinge angles and their derivatives for bending energies.

Layout mirrors the rest of ``simkit``: an ``_element`` tier holds the per-hinge
formula on explicit corner positions, while the un-suffixed functions gather
the four hinge vertices from global positions ``X`` via ``dihedral_wedge_map``
and assemble the global gradient / Hessian.

The signed dihedral angle uses ``atan2`` of the two face normals, so it is
valid over the full range and not just ``[0, pi]``.
"""

import numpy as np
import scipy as sp

from .dihedral_wedge_map import dihedral_wedge_map


def dihedral_angles_element(
    x0: np.ndarray, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray
) -> np.ndarray:
    """Signed dihedral angle between the two triangles of each hinge.

    Parameters
    ----------
    x0 : np.ndarray (nd, 3)
        Apex vertex of triangle 1.
    x1, x2 : np.ndarray (nd, 3)
        Shared edge vertices.
    x3 : np.ndarray (nd, 3)
        Apex vertex of triangle 2.

    Returns
    -------
    theta : np.ndarray (nd, 1)
        Signed dihedral angle per hinge, from ``atan2(s, c)``.
    """
    e0 = x2 - x1
    e0_hat = e0 / np.linalg.norm(e0, axis=1).reshape(-1, 1)

    e2 = x0 - x1
    e2_tilde = x3 - x1

    # Face normals of the two triangles sharing edge e0.
    n1 = np.cross(e0, e2)
    n2 = np.cross(e2_tilde, e0)

    n1_norm = n1 / np.linalg.norm(n1, axis=1).reshape(-1, 1)
    n2_norm = n2 / np.linalg.norm(n2, axis=1).reshape(-1, 1)

    # cos from the normal dot product; sin from the signed component along e0.
    c = np.sum(n1_norm * n2_norm, axis=1).reshape(-1, 1)
    s = np.sum(e0_hat * np.cross(n1_norm, n2_norm, axis=1), axis=1).reshape(-1, 1)

    theta = np.arctan2(s, c)             # signed angle, full range
    return theta


def dihedral_angles(X: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Per-hinge dihedral angles gathered from global vertex positions.

    Parameters
    ----------
    X : np.ndarray (n, 3)
        Vertex positions.
    D : np.ndarray (nd, 4)
        Hinge vertex indices ``(x0, x1, x2, x3)``.

    Returns
    -------
    theta : np.ndarray (nd, 1)
        Signed dihedral angle per hinge.
    """
    x1 = X[D[:, 0]]
    x2 = X[D[:, 1]]
    x3 = X[D[:, 2]]
    x4 = X[D[:, 3]]
    theta = dihedral_angles_element(x1, x2, x3, x4)
    return theta


def dihedral_angles_gradient(X: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Global gradient of the hinge angles w.r.t. all vertex DOFs.

    Gathers per-hinge gradients and scatters them onto global DOFs through the
    hinge-to-vertex map.

    Parameters
    ----------
    X : np.ndarray (n, 3)
        Vertex positions.
    D : np.ndarray (nd, 4)
        Hinge vertex indices ``(x0, x1, x2, x3)``.

    Returns
    -------
    dtheta_dx : np.ndarray (3n, 1)
        Assembled gradient over the flattened vertex DOFs.
    """
    x1 = X[D[:, 0]]
    x2 = X[D[:, 1]]
    x3 = X[D[:, 2]]
    x4 = X[D[:, 3]]

    # Map stacked hinge vertices to global vertices, lifted to 3 components.
    M = dihedral_wedge_map(D, X.shape[0])
    Me = sp.sparse.kron(M, sp.sparse.identity(3))

    dtheta_dx = Me.T @ dihedral_angles_gradient_element(x1, x2, x3, x4).reshape(-1, 1)
    return dtheta_dx


def dihedral_angles_hessian(X: np.ndarray, D: np.ndarray) -> "sp.sparse.spmatrix":
    """Global Hessian of the hinge angles w.r.t. all vertex DOFs.

    Block-diagonalizes the per-hinge 12x12 Hessians and conjugates by the
    hinge-to-vertex map to land on global DOFs.

    Parameters
    ----------
    X : np.ndarray (n, 3)
        Vertex positions.
    D : np.ndarray (nd, 4)
        Hinge vertex indices ``(x0, x1, x2, x3)``.

    Returns
    -------
    d2theta_dx2 : scipy.sparse.spmatrix (3n, 3n)
        Assembled Hessian over the flattened vertex DOFs.
    """
    x1 = X[D[:, 0]]
    x2 = X[D[:, 1]]
    x3 = X[D[:, 2]]
    x4 = X[D[:, 3]]

    M = dihedral_wedge_map(D, X.shape[0])
    Me = sp.sparse.kron(M, sp.sparse.identity(3))

    q = dihedral_angles_hessian_element(x1, x2, x3, x4)
    Q = sp.sparse.block_diag(q)          # stack the per-hinge 12x12 blocks
    d2theta_dx2 = Me.T @ Q @ Me
    return d2theta_dx2


def dihedral_angles_gradient_element(
    x0: np.ndarray, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray
) -> np.ndarray:
    """Per-hinge gradient of the dihedral angle w.r.t. its four vertices.

    Uses the standard hinge-angle gradient (e.g. Tamstorf & Grinspun, "Discrete
    bending forces and their Jacobians", 2013): the apex gradients point along
    the face normals scaled by inverse wedge heights, and the shared-edge
    gradients are weighted combinations via the interior cosines.

    Parameters
    ----------
    x0 : np.ndarray (nd, 3)
        Apex vertex of triangle 1.
    x1, x2 : np.ndarray (nd, 3)
        Shared edge vertices.
    x3 : np.ndarray (nd, 3)
        Apex vertex of triangle 2.

    Returns
    -------
    dtheta_dx : np.ndarray (nd, 12)
        Gradient stacked as ``[d/dx0, d/dx1, d/dx2, d/dx3]``.
    """
    # Edges; vertices 1,2 shared, x0 on triangle 1, x3 on triangle 2.
    e0 = x2 - x1
    e1 = x0 - x2
    e2 = x0 - x1
    e1_tilde = x3 - x2
    e2_tilde = x3 - x1

    # Face normals and areas.
    n = np.cross(e0, e2)
    n_tilde = np.cross(e2_tilde, e0)
    area = np.linalg.norm(n, axis=1).reshape(-1, 1) / 2
    area_tilde = np.linalg.norm(n_tilde, axis=1).reshape(-1, 1) / 2

    n_normalized = 0.5 * n / area.reshape(-1, 1)
    n_tilde_normalized = 0.5 * n_tilde / area_tilde.reshape(-1, 1)

    # Wedge heights (2*area / edge length) for each edge of each triangle.
    h0 = 2.0 * area / np.linalg.norm(e0, axis=1).reshape(-1, 1)
    h1 = 2.0 * area / np.linalg.norm(e1, axis=1).reshape(-1, 1)
    h2 = 2.0 * area / np.linalg.norm(e2, axis=1).reshape(-1, 1)

    h0_tilde = 2.0 * area_tilde / np.linalg.norm(e0, axis=1).reshape(-1, 1)
    h1_tilde = 2.0 * area_tilde / np.linalg.norm(e1_tilde, axis=1).reshape(-1, 1)
    h2_tilde = 2.0 * area_tilde / np.linalg.norm(e2_tilde, axis=1).reshape(-1, 1)

    # Interior-angle cosines used to weight the shared-edge gradients.
    cos_alpha1 = (np.sum(e0 * e2, axis=1) / (np.linalg.norm(e0, axis=1) * np.linalg.norm(e2, axis=1))).reshape(-1, 1)
    cos_alpha2 = -(np.sum(e0 * e1, axis=1) / (np.linalg.norm(e0, axis=1) * np.linalg.norm(e1, axis=1))).reshape(-1, 1)

    cos_alpha1_tilde = (np.sum(e0 * e2_tilde, axis=1) / (np.linalg.norm(e0, axis=1) * np.linalg.norm(e2_tilde, axis=1))).reshape(-1, 1)
    cos_alpha2_tilde = -(np.sum(e0 * e1_tilde, axis=1) / (np.linalg.norm(e0, axis=1) * np.linalg.norm(e1_tilde, axis=1))).reshape(-1, 1)

    # Apex gradients point along their own face normal; shared-edge gradients
    # blend both normals by the opposite-angle cosines.
    dtheta_dx0 = -(1 / h0) * n_normalized
    dtheta_dx1 = cos_alpha2 * n_normalized / h1 + cos_alpha2_tilde * n_tilde_normalized / h1_tilde
    dtheta_dx2 = cos_alpha1 * n_normalized / h2 + cos_alpha1_tilde * n_tilde_normalized / h2_tilde
    dtheta_dx3 = -(1 / h0_tilde) * n_tilde_normalized

    dtheta_dx = np.concatenate([dtheta_dx0, dtheta_dx1, dtheta_dx2, dtheta_dx3], axis=1)
    return dtheta_dx


def dihedral_angles_hessian_element(
    x0: np.ndarray, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray
) -> np.ndarray:
    """Per-hinge 12x12 Hessian of the dihedral angle w.r.t. its four vertices.

    Follows the analytic hinge-angle Hessian (Tamstorf & Grinspun, 2013),
    assembling the four diagonal and off-diagonal 3x3 blocks from the wedge
    geometry and filling the remainder by symmetry.

    Parameters
    ----------
    x0 : np.ndarray (nd, 3)
        Apex vertex of triangle 1.
    x1, x2 : np.ndarray (nd, 3)
        Shared edge vertices.
    x3 : np.ndarray (nd, 3)
        Apex vertex of triangle 2.

    Returns
    -------
    H : np.ndarray (nd, 12, 12)
        Per-hinge Hessian over the stacked DOFs ``[x0, x1, x2, x3]``.
    """

    def S_func(A: np.ndarray) -> np.ndarray:
        """Symmetrize a stack of matrices: ``A + A^T``."""
        return A + A.transpose(0, 2, 1)

    # Edges; vertices 1,2 shared, x0 on triangle 1, x3 on triangle 2.
    e0 = x2 - x1
    e1 = x0 - x2
    e2 = x0 - x1
    e1_tilde = x3 - x2
    e2_tilde = x3 - x1

    e_hat0 = e0 / np.linalg.norm(e0, axis=1).reshape(-1, 1)
    e_hat1 = e1 / np.linalg.norm(e1, axis=1).reshape(-1, 1)
    e_hat2 = e2 / np.linalg.norm(e2, axis=1).reshape(-1, 1)
    e_hat1_tilde = e1_tilde / np.linalg.norm(e1_tilde, axis=1).reshape(-1, 1)
    e_hat2_tilde = e2_tilde / np.linalg.norm(e2_tilde, axis=1).reshape(-1, 1)

    # Face normals and areas.
    n = np.cross(e0, e2)
    n_tilde = np.cross(e2_tilde, e0)
    area = np.linalg.norm(n, axis=1).reshape(-1, 1) / 2
    area_tilde = np.linalg.norm(n_tilde, axis=1).reshape(-1, 1) / 2

    n_hat = 0.5 * n / area.reshape(-1, 1)
    n_tilde_hat = 0.5 * n_tilde / area_tilde.reshape(-1, 1)

    # Wedge heights.
    l0 = np.linalg.norm(e0, axis=1).reshape(-1, 1)
    l1 = np.linalg.norm(e1, axis=1).reshape(-1, 1)
    l2 = np.linalg.norm(e2, axis=1).reshape(-1, 1)
    l1_tilde = np.linalg.norm(e1_tilde, axis=1).reshape(-1, 1)
    l2_tilde = np.linalg.norm(e2_tilde, axis=1).reshape(-1, 1)

    h0 = 2.0 * area / l0
    h1 = 2.0 * area / l1
    h2 = 2.0 * area / l2
    h0_tilde = 2.0 * area_tilde / l0
    h1_tilde = 2.0 * area_tilde / l1_tilde
    h2_tilde = 2.0 * area_tilde / l2_tilde

    # Interior-angle cosines for both triangles.
    cos_alpha1 = -np.sum(e_hat0 * e_hat1, axis=1).reshape(-1, 1)
    cos_alpha2 = np.sum(e_hat0 * e_hat2, axis=1).reshape(-1, 1)
    cos_alpha1_tilde = -np.sum(e_hat0 * e_hat1_tilde, axis=1).reshape(-1, 1)
    cos_alpha2_tilde = np.sum(e_hat0 * e_hat2_tilde, axis=1).reshape(-1, 1)

    # In-plane edge directions used to build the rank-one normal blocks.
    m_0 = np.cross(e_hat0, n_hat)
    m_1 = np.cross(e_hat1, n_hat)
    m_2 = -np.cross(e_hat2, n_hat)
    m_0_tilde = -np.cross(e_hat0, n_tilde_hat)
    m_1_tilde = -np.cross(e_hat1_tilde, n_tilde_hat)
    m_2_tilde = np.cross(e_hat2_tilde, n_tilde_hat)

    # Inverse height products (omega_ij = 1 / (h_i h_j)).
    omega_00 = 1 / (h0 * h0)
    omega_01 = 1 / (h0 * h1)
    omega_02 = 1 / (h0 * h2)
    omega_10 = 1 / (h1 * h0)
    omega_11 = 1 / (h1 * h1)
    omega_12 = 1 / (h1 * h2)
    omega_20 = 1 / (h2 * h0)
    omega_21 = 1 / (h2 * h1)
    omega_22 = 1 / (h2 * h2)

    omega_00_tilde = 1 / (h0_tilde * h0_tilde)
    omega_01_tilde = 1 / (h0_tilde * h1_tilde)
    omega_02_tilde = 1 / (h0_tilde * h2_tilde)
    omega_10_tilde = 1 / (h1_tilde * h0_tilde)
    omega_11_tilde = 1 / (h1_tilde * h1_tilde)
    omega_12_tilde = 1 / (h1_tilde * h2_tilde)
    omega_20_tilde = 1 / (h2_tilde * h0_tilde)
    omega_21_tilde = 1 / (h2_tilde * h1_tilde)
    omega_22_tilde = 1 / (h2_tilde * h2_tilde)

    # Rank-one normal-times-edge blocks (M) and their length-scaled forms (N).
    M_0 = n_hat[:, :, None] @ m_0[:, None, :]
    M_1 = n_hat[:, :, None] @ m_1[:, None, :]
    M_2 = n_hat[:, :, None] @ m_2[:, None, :]
    M_0_tilde = n_tilde_hat[:, :, None] @ m_0_tilde[:, None, :]
    M_1_tilde = n_tilde_hat[:, :, None] @ m_1_tilde[:, None, :]
    M_2_tilde = n_tilde_hat[:, :, None] @ m_2_tilde[:, None, :]

    N_0 = M_0 / np.linalg.norm(e0, axis=1)[:, None, None] ** 2
    N_1 = M_1 / np.linalg.norm(e1, axis=1)[:, None, None] ** 2
    N_2 = M_2 / np.linalg.norm(e2, axis=1)[:, None, None] ** 2
    N_0_tilde = M_0_tilde / np.linalg.norm(e0, axis=1)[:, None, None] ** 2
    N_1_tilde = M_1_tilde / np.linalg.norm(e1_tilde, axis=1)[:, None, None] ** 2
    N_2_tilde = M_2_tilde / np.linalg.norm(e2_tilde, axis=1)[:, None, None] ** 2

    # Cosine-weighted blocks (P) entering the shared-edge second derivatives.
    P10 = (omega_10 * cos_alpha1)[:, :, None] * M_0.transpose(0, 2, 1)
    P11 = (omega_11 * cos_alpha1)[:, :, None] * M_1.transpose(0, 2, 1)
    P12 = (omega_12 * cos_alpha1)[:, :, None] * M_2.transpose(0, 2, 1)
    P20 = (omega_20 * cos_alpha2)[:, :, None] * M_0.transpose(0, 2, 1)
    P21 = (omega_21 * cos_alpha2)[:, :, None] * M_1.transpose(0, 2, 1)
    P22 = (omega_22 * cos_alpha2)[:, :, None] * M_2.transpose(0, 2, 1)

    P11_tilde = (omega_11_tilde * cos_alpha1_tilde)[:, :, None] * M_1_tilde.transpose(0, 2, 1)
    P12_tilde = (omega_12_tilde * cos_alpha1_tilde)[:, :, None] * M_2_tilde.transpose(0, 2, 1)
    P10_tilde = (omega_10_tilde * cos_alpha1_tilde)[:, :, None] * M_0_tilde.transpose(0, 2, 1)
    P20_tilde = (omega_20_tilde * cos_alpha2_tilde)[:, :, None] * M_0_tilde.transpose(0, 2, 1)
    P21_tilde = (omega_21_tilde * cos_alpha2_tilde)[:, :, None] * M_1_tilde.transpose(0, 2, 1)
    P22_tilde = (omega_22_tilde * cos_alpha2_tilde)[:, :, None] * M_2_tilde.transpose(0, 2, 1)

    # Apex blocks (Q) entering the diagonal apex second derivatives.
    Q0 = omega_00[..., None] * M_0
    Q1 = omega_01[..., None] * M_1
    Q2 = omega_02[..., None] * M_2
    Q0_tilde = omega_00_tilde[..., None] * M_0_tilde
    Q1_tilde = omega_01_tilde[..., None] * M_1_tilde
    Q2_tilde = omega_02_tilde[..., None] * M_2_tilde

    # Independent blocks; the rest follow by symmetry.
    H00 = -S_func(Q0)
    H03 = np.zeros(H00.shape)            # apex-apex coupling vanishes
    H10 = P10 - Q1
    H11 = S_func(P11) - N_0 + S_func(P11_tilde) - N_0_tilde
    H12 = P12 + P21.transpose(0, 2, 1) + N_0 + P12_tilde + P21_tilde.transpose(0, 2, 1) + N_0_tilde
    H13 = P10_tilde - Q1_tilde
    H20 = P20 - Q2
    H22 = S_func(P22) - N_0 + S_func(P22_tilde) - N_0_tilde
    H23 = P20_tilde - Q2_tilde
    H33 = -S_func(Q0_tilde)

    # Fill the symmetric counterparts.
    H01 = H10.transpose(0, 2, 1)
    H02 = H20.transpose(0, 2, 1)
    H21 = H12.transpose(0, 2, 1)
    H30 = H03.transpose(0, 2, 1)
    H31 = H13.transpose(0, 2, 1)
    H32 = H23.transpose(0, 2, 1)

    H = np.block([[H00, H01, H02, H03],
                  [H10, H11, H12, H13],
                  [H20, H21, H22, H23],
                  [H30, H31, H32, H33]])
    return H