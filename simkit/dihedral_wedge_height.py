
import numpy as np
def dihedral_wedge_heights(X, D):
    x0 = X[D[:, 0]]
    x1 = X[D[:, 1]]
    x2 = X[D[:, 2]]
    x3 = X[D[:, 3]]
    h0, h1, h2, h0_tilde, h1_tilde, h2_tilde = dihedral_wedge_heights_element(x0, x1, x2, x3)
    
    return h0, h1, h2, h0_tilde, h1_tilde, h2_tilde

def dihedral_wedge_height(X, D):
    h0, h1, h2, h0_tilde, h1_tilde, h2_tilde = dihedral_wedge_heights(X, D)
    
    h = (h0 + h0_tilde)/2
    return h

def dihedral_wedge_heights_element(x0, x1, x2, x3):
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
    return h0, h1, h2, h0_tilde, h1_tilde, h2_tilde
    