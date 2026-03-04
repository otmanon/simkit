
import numpy as np
import scipy as sp

from .dihedral_wedge_map import dihedral_wedge_map
def dihedral_angles_element(x0, x1, x2, x3):
    """
    Compute dihedral angle between two triangles sharing an edge.
    
    Parameters:
    x0: vertex on triangle 1 (3D vector)
    x1, x2: shared vertices (3D vectors) - the edge
    x3: vertex on triangle 2 (3D vector)
    
    Returns:
    theta: dihedral angle (scalar)
    """
    
    e0 = x2 - x1
    
    e0_hat = e0 / np.linalg.norm(e0, axis=1).reshape(-1, 1)
    
    e1 = x0 - x2
    e2 = x0 - x1
    
    e1_tilde = x3 - x2
    e2_tilde = x3 - x1
    
    # Cross products to get normals
    n1 = np.cross(e0, e2)
    n2 = np.cross(  e2_tilde, e0)
    
    # Normalize normals
    n1_norm = n1 /np.linalg.norm(n1, axis=1).reshape(-1, 1)
    n2_norm = n2 / np.linalg.norm(n2, axis=1).reshape(-1, 1)
    
    # Dot product of normalized normals
    c = np.sum(n1_norm * n2_norm, axis=1).reshape(-1, 1)
    
    s = np.sum(e0_hat * np.cross(n1_norm, n2_norm, axis=1), axis=1).reshape(-1,1)
    # Dihedral angle
    # theta = np.arccos(c)
    theta = np.arctan2(s, c)
    return theta


def dihedral_angles(X, D):
    x1 = X[D[:, 0]]
    x2 = X[D[:, 1]]
    x3 = X[D[:, 2]]
    x4 = X[D[:, 3]]    
    
    
    theta = dihedral_angles_element(x1, x2, x3, x4)
    return theta

def dihedral_angles_gradient(X, D):
    x1 = X[D[:, 0]]
    x2 = X[D[:, 1]]
    x3 = X[D[:, 2]]
    x4 = X[D[:, 3]]    
    
    M = dihedral_wedge_map(D, X.shape[0])
    Me = sp.sparse.kron(M, sp.sparse.identity(3))
    
    dtheta_dx = Me.T @ dihedral_angles_gradient_element(x1, x2, x3, x4).reshape(-1, 1)
    
    return dtheta_dx

def dihedral_angles_hessian(X, D):
    x1 = X[D[:, 0]]
    x2 = X[D[:, 1]]
    x3 = X[D[:, 2]]
    x4 = X[D[:, 3]]    
    M = dihedral_wedge_map(D, X.shape[0])
    Me = sp.sparse.kron(M, sp.sparse.identity(3))
    q = dihedral_angles_hessian_element(x1, x2, x3, x4)
    Q = sp.sparse.block_diag(q)
    d2theta_dx2 = Me.T @ Q @ Me
    
    return d2theta_dx2
    
    

    

def dihedral_angles_gradient_element(x0, x1, x2, x3):
    """
    dtheta_dx: gradient of dihedral angle with respect to the four hinge vertices

    Returns |E| x 12
    """
    
    # vertices 1 and 2 are shared between the two triangles, vertices 3 is on triangle 1, vertices 4 is on triangle 2
    e0 = x2 - x1
    e1 = x0 - x2
    e2 = x0 - x1
    e1_tilde = x3 - x2
    e2_tilde = x3 - x1
    
    # Cross products to get normals
    n = np.cross(e0, e2)
    n_tilde = np.cross(  e2_tilde, e0)
    
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
    
    cos_alpha0 = (np.sum(e1 * e2, axis=1) / (np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1))).reshape(-1, 1)
    cos_alpha1 = (np.sum(e0 * e2, axis=1) / (np.linalg.norm(e0, axis=1) * np.linalg.norm(e2, axis=1))).reshape(-1, 1)
    cos_alpha2 = -(np.sum(e0 * e1, axis=1) / (np.linalg.norm(e0, axis=1) * np.linalg.norm(e1, axis=1))).reshape(-1, 1)
    
    cos_alpha0_tilde = (np.sum(e1_tilde * e2_tilde, axis=1) / (np.linalg.norm(e1_tilde, axis=1) * np.linalg.norm(e2_tilde, axis=1))).reshape(-1, 1)
    cos_alpha1_tilde = (np.sum(e0 * e2_tilde, axis=1) / (np.linalg.norm(e0, axis=1) * np.linalg.norm(e2_tilde, axis=1))).reshape(-1, 1)
    cos_alpha2_tilde = -(np.sum(e0 * e1_tilde, axis=1) / (np.linalg.norm(e0, axis=1) * np.linalg.norm(e1_tilde, axis=1))   ).reshape(-1, 1)
    
    dtheta_dx0 =  -(1/ h0) * n_normalized
    
    dtheta_dx1 = cos_alpha2 * n_normalized / h1 + cos_alpha2_tilde * n_tilde_normalized / h1_tilde
    dtheta_dx2 = cos_alpha1 * n_normalized  / h2 + cos_alpha1_tilde * n_tilde_normalized / h2_tilde
    
    dtheta_dx3 =  -(1/ h0_tilde) * n_tilde_normalized
    
    dtheta_dx =  np.concatenate([dtheta_dx0, dtheta_dx1, dtheta_dx2, dtheta_dx3], axis=1)
    return dtheta_dx


def dihedral_angles_hessian_element(x0, x1, x2, x3):
    """
    d2theta_dx2:  |E| x 12 x 12 Hessian of dihedral angle with respect to hinge vertices. 
    
    Returns
    """
    
    def S_func(A):
        return A + A.transpose(0, 2, 1)

    # vertices 1 and 2 are shared between the two triangles, vertices 3 is on triangle 1, vertices 4 is on triangle 2
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
    
    l0 = np.linalg.norm(e0, axis=1).reshape(-1, 1)
    l1 = np.linalg.norm(e1, axis=1).reshape(-1, 1)
    l2 = np.linalg.norm(e2, axis=1).reshape(-1, 1)
    l1_tilde = np.linalg.norm(e1_tilde, axis=1).reshape(-1, 1)
    l2_tilde = np.linalg.norm(e2_tilde, axis=1).reshape(-1, 1)
    
    # Cross products to get normals
    n = np.cross(e0, e2)
    n_tilde = np.cross( e2_tilde,  e0)
    
    area = np.linalg.norm(n, axis=1).reshape(-1, 1) / 2
    area_tilde = np.linalg.norm(n_tilde, axis=1).reshape(-1, 1) / 2
    
    n_hat = 0.5 * n / area.reshape(-1, 1)
    n_tilde_hat = 0.5 * n_tilde / area_tilde.reshape(-1, 1)
    
    
    h0 = 2.0 * area / l0
    h1 = 2.0 * area / l1
    h2 = 2.0 * area / l2
    
    h0_tilde = 2.0 * area_tilde / l0
    h1_tilde = 2.0 * area_tilde / l1_tilde
    h2_tilde = 2.0 * area_tilde / l2_tilde
    
    cos_alpha0 = np.sum(e_hat2 * e_hat1, axis=1).reshape(-1, 1)  
    cos_alpha1 = -np.sum(e_hat0 * e_hat1, axis=1).reshape(-1, 1)    
    cos_alpha2 = np.sum(e_hat0 * e_hat2, axis=1).reshape(-1, 1)
    
    cos_alpha0_tilde = np.sum(e_hat1_tilde * e_hat2_tilde, axis=1).reshape(-1, 1)  
    cos_alpha1_tilde = -np.sum(e_hat0 * e_hat1_tilde, axis=1).reshape(-1, 1)  
    cos_alpha2_tilde = np.sum(e_hat0 * e_hat2_tilde, axis=1).reshape(-1, 1)  
    

    # ms
    m_0 = np.cross(e_hat0, n_hat) 
    m_1 = np.cross(e_hat1, n_hat) 
    m_2 = -np.cross(e_hat2, n_hat) 
    # need to define m_0_tilde, m_1_tilde, m_2_tilde.
    
    # m_tildes
    m_0_tilde = -np.cross(e_hat0, n_tilde_hat) 
    m_1_tilde = -np.cross(e_hat1_tilde, n_tilde_hat) 
    m_2_tilde = np.cross(e_hat2_tilde, n_tilde_hat) 
    
    #omegas
    omega_00 =  1/(h0 * h0)
    omega_01 =  1/(h0 * h1)
    omega_02 =  1/(h0 * h2)
    omega_10 =  1/(h1 * h0)
    omega_11 =  1/(h1 * h1)
    omega_12 =  1/(h1 * h2)
    omega_20 =  1/(h2 * h0)
    omega_21 =  1/(h2 * h1)
    omega_22 =  1/(h2 * h2)
    
    # omega_tildes
    omega_00_tilde = 1/(h0_tilde * h0_tilde)
    omega_01_tilde = 1/(h0_tilde * h1_tilde)
    omega_02_tilde = 1/(h0_tilde * h2_tilde)
    omega_10_tilde = 1/(h1_tilde * h0_tilde)
    omega_11_tilde = 1/(h1_tilde * h1_tilde)
    omega_12_tilde = 1/(h1_tilde * h2_tilde)
    omega_20_tilde = 1/(h2_tilde * h0_tilde)
    omega_21_tilde = 1/(h2_tilde * h1_tilde)
    omega_22_tilde = 1/(h2_tilde * h2_tilde)
    
    #Ms
    M_0 = n_hat[:, :, None] @ m_0[:, None, :]
    M_1 = n_hat[:, :, None] @ m_1[:, None, :]
    M_2 = n_hat[:, :, None] @ m_2[:, None, :]
    
    # M_tildes
    M_0_tilde = n_tilde_hat[:, :, None] @ m_0_tilde[:, None, :]
    M_1_tilde = n_tilde_hat[:, :, None] @ m_1_tilde[:, None, :]
    M_2_tilde = n_tilde_hat[:, :, None] @ m_2_tilde[:, None, :]
    
    # Ns
    N_0 = M_0 / np.linalg.norm(e0, axis=1)[:, None,  None]**2
    N_1 = M_1 / np.linalg.norm(e1, axis=1)[:, None,  None]**2
    N_2 = M_2 / np.linalg.norm(e2, axis=1)[:, None,  None]**2
    
    # Ns_tilde
    N_0_tilde = M_0_tilde / np.linalg.norm(e0, axis=1)[:, None,  None]**2
    N_1_tilde = M_1_tilde / np.linalg.norm(e1_tilde, axis=1)[:, None,  None]**2
    N_2_tilde = M_2_tilde / np.linalg.norm(e2_tilde, axis=1)[:, None,  None]**2
    
    # Ps
    P00 = (omega_00 * cos_alpha0)[:, :, None] * M_0.transpose(0, 2, 1)
    P01 = (omega_01 * cos_alpha0)[:, :, None] * M_1.transpose(0, 2, 1)
    P02 = (omega_02 * cos_alpha0)[:, :, None] * M_2.transpose(0, 2, 1)
    
    P10 = (omega_10 * cos_alpha1)[:, :, None] * M_0.transpose(0, 2, 1)
    P11 = (omega_11 * cos_alpha1)[:, :, None] * M_1.transpose(0, 2, 1)
    P12 = (omega_12 * cos_alpha1)[:, :, None] * M_2.transpose(0, 2, 1)
    
    P20 = (omega_20 * cos_alpha2)[:, :, None] * M_0.transpose(0, 2, 1)
    P21 = (omega_21 * cos_alpha2)[:, :, None] * M_1.transpose(0, 2, 1)
    P22 = (omega_22 * cos_alpha2)[:, :, None] * M_2.transpose(0, 2, 1)
    
    
    # P_tildes
    P00_tilde = (omega_00_tilde * cos_alpha0_tilde)[:, :, None] * M_0_tilde.transpose(0, 2, 1)
    P01_tilde = (omega_01_tilde * cos_alpha0_tilde)[:, :, None] * M_1_tilde.transpose(0, 2, 1)
    P02_tilde = (omega_02_tilde * cos_alpha0_tilde)[:, :, None] * M_2_tilde.transpose(0, 2, 1)
    
    P10_tilde = (omega_10_tilde * cos_alpha1_tilde)[:, :, None] * M_0_tilde.transpose(0, 2, 1)
    P11_tilde = (omega_11_tilde * cos_alpha1_tilde)[:, :, None] * M_1_tilde.transpose(0, 2, 1)
    P12_tilde = (omega_12_tilde * cos_alpha1_tilde)[:, :, None] * M_2_tilde.transpose(0, 2, 1)
    
    P20_tilde = (omega_20_tilde * cos_alpha2_tilde)[:, :, None] * M_0_tilde.transpose(0, 2, 1)
    P21_tilde = (omega_21_tilde * cos_alpha2_tilde)[:, :, None] * M_1_tilde.transpose(0, 2, 1)
    P22_tilde = (omega_22_tilde * cos_alpha2_tilde)[:, :, None] * M_2_tilde.transpose(0, 2, 1)
    
    # Qs
    Q0 = omega_00[..., None] * M_0
    Q1 = omega_01[..., None] * M_1
    Q2 = omega_02[..., None] * M_2
    
    Q0_tilde = omega_00_tilde[..., None] * M_0_tilde
    Q1_tilde = omega_01_tilde[..., None] * M_1_tilde
    Q2_tilde = omega_02_tilde[..., None] * M_2_tilde
    
    # primary blocks
    H00 = -S_func(Q0)
    H03 = np.zeros(H00.shape)
    
    
    H10 =  P10 - Q1
    H11 = S_func(P11) - N_0 + S_func(P11_tilde) - N_0_tilde
    H12 = P12 + P21.transpose(0, 2, 1) + N_0 + P12_tilde + P21_tilde.transpose(0, 2, 1) + N_0_tilde
    H13 = P10_tilde - Q1_tilde
    
    H20 =  P20 - Q2
    H22 = S_func(P22) - N_0 + S_func(P22_tilde) - N_0_tilde
    H23 = P20_tilde - Q2_tilde
    
    H33 = -S_func(Q0_tilde)
    
    
    # now through symmetry get the rest
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