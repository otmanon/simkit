import numpy as np


import scipy as sp

from ..dihedral_angles import dihedral_angles
from ..dihedral_wedge_map import dihedral_wedge_map
from ..dihedral_angles import dihedral_angles_gradient_element, dihedral_angles_hessian_element

def discrete_shells_bending_energy_dtheta(theta, theta0, ym_bending, he, le):
    dtheta = theta - theta0
    w = np.linalg.norm(le, axis=1).reshape(-1, 1) /he    
    psi = w * dtheta ** 2 * ym_bending
    energy = np.sum(psi)
    return energy

def discrete_shells_bending_gradient_dtheta(theta, theta0, ym_bending, he, le):
    dtheta = theta - theta0
    w = np.linalg.norm(le, axis=1).reshape(-1, 1) / he
    dpsi_dtheta = 2 * w * dtheta * ym_bending
    return dpsi_dtheta

def discrete_shells_bending_hessian_d2theta(theta, theta0, ym_bending, he, le):
    w = np.linalg.norm(le, axis=1).reshape(-1, 1) / he
    dpsi_dtheta2 = 2 * ym_bending * w
    return dpsi_dtheta2

def discrete_shells_bending_energy_dx(X, D, theta0, ym_bending, he, le):
    theta = dihedral_angles(X, D)
    energy = discrete_shells_bending_energy_dtheta(theta, theta0, ym_bending, he, le)
    return energy

def discrete_shells_bending_gradient_dx(X, D, theta0, ym_bending, he, le):
    theta = dihedral_angles(X, D)
    de_dtheta = discrete_shells_bending_gradient_dtheta(theta, theta0, ym_bending, he, le)

    M = dihedral_wedge_map(D, X.shape[0])
    x0 = X[D[:, 0]]
    x1 = X[D[:, 1]]
    x2 = X[D[:, 2]]
    x3 = X[D[:, 3]]
    dtheta_dx = dihedral_angles_gradient_element(x0, x1, x2, x3)
    Me = sp.sparse.kron(M, sp.sparse.identity(3))
    de_dx = Me.T @ (dtheta_dx * de_dtheta).reshape(-1, 1)
    return de_dx

def discrete_shells_bending_hessian_d2x(X, D, theta0, ym_bending, he, le):
    theta = dihedral_angles(X.reshape(-1, 3), D)
    x0 = X[D[:, 0]]
    x1 = X[D[:, 1]]
    x2 = X[D[:, 2]]
    x3 = X[D[:, 3]]
    
    dtheta_dx = dihedral_angles_gradient_element(x0, x1, x2, x3)
    d2theta_dx2 = dihedral_angles_hessian_element(x0, x1, x2, x3)
    
    de_dtheta = discrete_shells_bending_gradient_dtheta(theta, theta0, ym_bending, he, le)
    
    d2e_dtheta2 = discrete_shells_bending_hessian_d2theta(theta, theta0, ym_bending, he, le)
    
    term_1 = (de_dtheta[:, :, None]*d2theta_dx2)
    term_2 =  d2e_dtheta2[:, :, None] * (dtheta_dx[:, :, None] @ dtheta_dx[:, None, :])
    Q =  (term_1 + term_2)
    
    
    
    Q2 = sp.sparse.block_diag(Q)
    
    M = dihedral_wedge_map(D, X.shape[0])
    Me = sp.sparse.kron(M, sp.sparse.identity(3))
    H = Me.T @ Q2 @ Me
    return H




