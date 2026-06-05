"""3D dihedral angles and their derivatives for bending energies.

The 3D member of the dihedral-angle family: a surface in space folds across an
interior **edge** shared by two triangles. Each element is a vertex quadruple
``(x0, x1, x2, x3)`` where ``(x1, x2)`` is the shared edge and ``x0``/``x3`` are
the opposite apexes. The signed dihedral angle uses ``atan2`` of the two face
normals, so it is valid over the full range (not just ``[0, pi]``).

Element derivatives are returned in the compact per-hinge layout with columns
ordered ``[x0(3), x1(3), x2(3), x3(3)]`` (matching ``kron(wedge_map(C, n),
eye(3))``). The dimension-dispatching front end lives in
:mod:`simkit.dihedral_angles`; the 2D sibling is :mod:`simkit.dihedral_angles_2d`.
"""

import numpy as np


# --------------------------------------------------------------------------- #
# Per-vertex cores (private): the analytic formulas on explicit corners        #
# --------------------------------------------------------------------------- #
def _angle(x0: np.ndarray, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> np.ndarray:
    e0 = x2 - x1
    e0_hat = e0 / np.linalg.norm(e0, axis=1).reshape(-1, 1)
    e2 = x0 - x1
    e2_tilde = x3 - x1

    n1 = np.cross(e0, e2)
    n2 = np.cross(e2_tilde, e0)
    n1_norm = n1 / np.linalg.norm(n1, axis=1).reshape(-1, 1)
    n2_norm = n2 / np.linalg.norm(n2, axis=1).reshape(-1, 1)

    c = np.sum(n1_norm * n2_norm, axis=1).reshape(-1, 1)
    s = np.sum(e0_hat * np.cross(n1_norm, n2_norm, axis=1), axis=1).reshape(-1, 1)
    return np.arctan2(s, c)


def _grad(x0: np.ndarray, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> np.ndarray:
    """Per-hinge gradient ``(nd, 12)`` stacked as ``[d/dx0, d/dx1, d/dx2, d/dx3]``.

    Standard hinge-angle gradient (Tamstorf & Grinspun, "Discrete bending forces
    and their Jacobians", 2013).
    """
    e0 = x2 - x1
    e1 = x0 - x2
    e2 = x0 - x1
    e1_tilde = x3 - x2
    e2_tilde = x3 - x1

    n = np.cross(e0, e2)
    n_tilde = np.cross(e2_tilde, e0)
    area = np.linalg.norm(n, axis=1).reshape(-1, 1) / 2
    area_tilde = np.linalg.norm(n_tilde, axis=1).reshape(-1, 1) / 2

    n_normalized = 0.5 * n / area.reshape(-1, 1)
    n_tilde_normalized = 0.5 * n_tilde / area_tilde.reshape(-1, 1)

    h0 = 2.0 * area / np.linalg.norm(e0, axis=1).reshape(-1, 1)
    h1 = 2.0 * area / np.linalg.norm(e1, axis=1).reshape(-1, 1)
    h2 = 2.0 * area / np.linalg.norm(e2, axis=1).reshape(-1, 1)
    h0_tilde = 2.0 * area_tilde / np.linalg.norm(e0, axis=1).reshape(-1, 1)
    h1_tilde = 2.0 * area_tilde / np.linalg.norm(e1_tilde, axis=1).reshape(-1, 1)
    h2_tilde = 2.0 * area_tilde / np.linalg.norm(e2_tilde, axis=1).reshape(-1, 1)

    cos_alpha1 = (np.sum(e0 * e2, axis=1) / (np.linalg.norm(e0, axis=1) * np.linalg.norm(e2, axis=1))).reshape(-1, 1)
    cos_alpha2 = -(np.sum(e0 * e1, axis=1) / (np.linalg.norm(e0, axis=1) * np.linalg.norm(e1, axis=1))).reshape(-1, 1)
    cos_alpha1_tilde = (np.sum(e0 * e2_tilde, axis=1) / (np.linalg.norm(e0, axis=1) * np.linalg.norm(e2_tilde, axis=1))).reshape(-1, 1)
    cos_alpha2_tilde = -(np.sum(e0 * e1_tilde, axis=1) / (np.linalg.norm(e0, axis=1) * np.linalg.norm(e1_tilde, axis=1))).reshape(-1, 1)

    dtheta_dx0 = -(1 / h0) * n_normalized
    dtheta_dx1 = cos_alpha2 * n_normalized / h1 + cos_alpha2_tilde * n_tilde_normalized / h1_tilde
    dtheta_dx2 = cos_alpha1 * n_normalized / h2 + cos_alpha1_tilde * n_tilde_normalized / h2_tilde
    dtheta_dx3 = -(1 / h0_tilde) * n_tilde_normalized

    return np.concatenate([dtheta_dx0, dtheta_dx1, dtheta_dx2, dtheta_dx3], axis=1)


def _hess(x0: np.ndarray, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> np.ndarray:
    """Per-hinge ``(nd, 12, 12)`` Hessian over stacked DOFs ``[x0, x1, x2, x3]``.

    Analytic hinge-angle Hessian (Tamstorf & Grinspun, 2013).
    """
    def S_func(A: np.ndarray) -> np.ndarray:
        return A + A.transpose(0, 2, 1)

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

    n = np.cross(e0, e2)
    n_tilde = np.cross(e2_tilde, e0)
    area = np.linalg.norm(n, axis=1).reshape(-1, 1) / 2
    area_tilde = np.linalg.norm(n_tilde, axis=1).reshape(-1, 1) / 2

    n_hat = 0.5 * n / area.reshape(-1, 1)
    n_tilde_hat = 0.5 * n_tilde / area_tilde.reshape(-1, 1)

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

    cos_alpha1 = -np.sum(e_hat0 * e_hat1, axis=1).reshape(-1, 1)
    cos_alpha2 = np.sum(e_hat0 * e_hat2, axis=1).reshape(-1, 1)
    cos_alpha1_tilde = -np.sum(e_hat0 * e_hat1_tilde, axis=1).reshape(-1, 1)
    cos_alpha2_tilde = np.sum(e_hat0 * e_hat2_tilde, axis=1).reshape(-1, 1)

    m_0 = np.cross(e_hat0, n_hat)
    m_1 = np.cross(e_hat1, n_hat)
    m_2 = -np.cross(e_hat2, n_hat)
    m_0_tilde = -np.cross(e_hat0, n_tilde_hat)
    m_1_tilde = -np.cross(e_hat1_tilde, n_tilde_hat)
    m_2_tilde = np.cross(e_hat2_tilde, n_tilde_hat)

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

    Q0 = omega_00[..., None] * M_0
    Q1 = omega_01[..., None] * M_1
    Q2 = omega_02[..., None] * M_2
    Q0_tilde = omega_00_tilde[..., None] * M_0_tilde
    Q1_tilde = omega_01_tilde[..., None] * M_1_tilde
    Q2_tilde = omega_02_tilde[..., None] * M_2_tilde

    H00 = -S_func(Q0)
    H03 = np.zeros(H00.shape)
    H10 = P10 - Q1
    H11 = S_func(P11) - N_0 + S_func(P11_tilde) - N_0_tilde
    H12 = P12 + P21.transpose(0, 2, 1) + N_0 + P12_tilde + P21_tilde.transpose(0, 2, 1) + N_0_tilde
    H13 = P10_tilde - Q1_tilde
    H20 = P20 - Q2
    H22 = S_func(P22) - N_0 + S_func(P22_tilde) - N_0_tilde
    H23 = P20_tilde - Q2_tilde
    H33 = -S_func(Q0_tilde)

    H01 = H10.transpose(0, 2, 1)
    H02 = H20.transpose(0, 2, 1)
    H21 = H12.transpose(0, 2, 1)
    H30 = H03.transpose(0, 2, 1)
    H31 = H13.transpose(0, 2, 1)
    H32 = H23.transpose(0, 2, 1)

    return np.block([[H00, H01, H02, H03],
                     [H10, H11, H12, H13],
                     [H20, H21, H22, H23],
                     [H30, H31, H32, H33]])


# --------------------------------------------------------------------------- #
# Public (X, C) entrypoints                                                    #
# --------------------------------------------------------------------------- #
def _corners(X, C):
    return X[C[:, 0]], X[C[:, 1]], X[C[:, 2]], X[C[:, 3]]


def dihedral_angles_3d(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Per-hinge signed dihedral angle gathered from global positions.

    Parameters
    ----------
    X : np.ndarray (n, 3)
        Vertex positions.
    C : np.ndarray (nd, 4)
        Hinge vertex indices ``(x0, x1, x2, x3)``.

    Returns
    -------
    theta : np.ndarray (nd, 1)
        Signed dihedral angle per hinge.
    """
    return _angle(*_corners(X, C))


def dihedral_angles_3d_gradient_element(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Per-hinge compact gradient ``(nd, 12)`` in stacked-corner order."""
    return _grad(*_corners(X, C))


def dihedral_angles_3d_hessian_element(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Per-hinge compact Hessian ``(nd, 12, 12)`` in stacked-corner order."""
    return _hess(*_corners(X, C))
