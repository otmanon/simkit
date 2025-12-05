import numpy as np
import scipy as sp
from ..hinge_angles import hinge_angles
from ..hinge_jacobian import hinge_jacobian
from ..hinge_hessian import hinge_hessian_compact
from ..psd_project import psd_project


def bending_energy(x, H, theta0, ymI, l):
    """
    Bending energy for a beam.
    Parameters:
        x dim*n x 1  (np.ndarray): The positions of the nodes of the beam.
        H J x 3  (np.ndarray): The number of joints in the beam .
        theta0 J x 1 (np.ndarray): The initial hinge angles of the beam.
        ymI E x 1 (np.ndarray): The Young's modulus of the beam multiplied by the area moment of inertia (effectively the bending stiffness).
        l E x 1 (np.ndarray): The length of the beam.
    Returns:
        e float: The bending energy of the beam.
    """
    l = np.asarray(l).reshape(-1, 1)
    ymI = np.asarray(ymI).reshape(-1, 1)
    coeff = ymI / l
    theta = hinge_angles(x.reshape(-1, 2), H)
    dtheta = theta - theta0
    return 0.5 * np.sum( coeff * (dtheta**2))


def bending_gradient(x, H, theta0, ymI, l):
    """ Bending gradient for a beam.
    Parameters:
        x dim*n x 1 (np.ndarray): The positions of the nodes of the beam.
        H J x 3 (np.ndarray): The number of joints in the beam .
        theta0 J x 1 (np.ndarray): The initial hinge angles of the beam.
        ymI E x 1 (np.ndarray): The Young's modulus of the beam multiplied by the area moment of inertia (effectively the bending stiffness).
        l E x 1 (np.ndarray): The length of the beam.
        I E x 1 (np.ndarray): The area moment of inertia of the beam (width * thickness ** 3 / 12)
    Returns:
        g dim*n x 1 (np.ndarray): The bending gradient of the beam.
    """
    l = np.asarray(l).reshape(-1, 1)
    ymI = np.asarray(ymI).reshape(-1, 1)
    coeff = ymI / l
    theta = hinge_angles(x.reshape(-1, 2), H)
    dtheta = theta - theta0
    denergy_dtheta = coeff * dtheta
    dtheta_dx = hinge_jacobian(x.reshape(-1, 2), H)
    de = dtheta_dx.T @ denergy_dtheta
    return de.reshape(-1, 1)


def bending_hessian(x, H, theta0, ymI, l):
    """ Bending hessian for a beam.
    Parameters:
        x dim*n x 1 (np.ndarray): The positions of the nodes of the beam.
        H J x 3 (np.ndarray): The number of joints in the beam .
        theta0 J x 1 (np.ndarray): The initial hinge angles of the beam.
        ym E x 1 (np.ndarray): The Young's modulus of the beam.
        l E x 1 (np.ndarray): The length of the beam.
        I E x 1 (np.ndarray): The area moment of inertia of the beam (width * thickness ** 3 / 12)
    Returns:
        H dim*n x dim*n (sp.sparse.csc_matrix): The bending hessian of the beam.
    """
    l = np.asarray(l).reshape(-1, 1)
    ymI = np.asarray(ymI).reshape(-1, 1)
    coeff = ymI / l
    d2energy_dtheta2 = sp.sparse.diags(coeff.flatten())# * np.eye(dtheta.shape[0])
    dtheta_dx = hinge_jacobian(x.reshape(-1, 2), H)
    
    term_1 = dtheta_dx.T @ (d2energy_dtheta2 @ dtheta_dx)
        
    # need to dot third order term with derivative of d2theta_dx2.
    # Instead of building third order sparse tensor explicitly, just dot
    
    # don't even include this term because it can go indefinite. not worth the projection.
    
    # curvature = curvature_func(x.reshape(-1, 2), H)
    # dcurvature = curvature - curvature0
    # denergy_dcurvature =  dcurvature
    # d2curvature_dx2_vec = d2curvature_func(x.reshape(-1, 2), H)    
    # d2curvature_dx2_vec = psd_project(d2curvature_dx2_vec)
    # term_2_compact = (k.reshape(-1, 1, 1) * vol.reshape(-1, 1, 1)
    #                   * denergy_dcurvature[:, :, None]) * d2curvature_dx2_vec 

    # offset = np.array([0, 1, 0, 1, 0, 1])
    # cols   = np.repeat(H*2, 2, axis=1) + offset       # shape (|E|,6)
    # cols2  = np.repeat(cols[:, None, :], 6, axis=1)   # (|E|,6,6)
    # rows   = cols2.transpose(0, 2, 1) 

    # term_2 = sp.sparse.coo_matrix(
    #         (term_2_compact.ravel(), (rows.ravel(), cols2.ravel())),
    #         shape=(x.size, x.size))
    
    Q = term_1 #+ term_2 
    return Q.tocsc()


# def bending_energy(x, H, theta0, k):
#     theta = hinge_angles(x.reshape(-1, 2), H)
#     dtheta = theta - theta0
#     return 0.5 * np.sum( k * (dtheta**2))

# def bending_gradient(x, H, theta0, k):
#     theta = hinge_angles(x.reshape(-1, 2), H)
#     dtheta = theta - theta0
#     denergy_dtheta = k * dtheta
#     dtheta_dx = hinge_jacobian(x.reshape(-1, 2), H)
#     de = dtheta_dx.T @ denergy_dtheta
#     return de.reshape(-1, 1)


# def bending_hessian(x, H, theta0, k):
#     theta = hinge_angles(x.reshape(-1, 2), H)
#     dtheta = theta - theta0
#     denergy_dtheta = k * dtheta
#     d2energy_dtheta2 = k * np.eye(dtheta.shape[0])
#     dtheta_dx = hinge_jacobian(x.reshape(-1, 2), H)
    
#     term_1 = dtheta_dx.T @ (d2energy_dtheta2 @ dtheta_dx)
        
    
#     # need to dot third order term with derivative of d2theta_dx2.
#     # Instead of building third order sparse tensor explicitly, just dot
#     d2theta_dx2_vec = hinge_hessian_compact(x.reshape(-1, 2), H)    
#     term_2_compact = denergy_dtheta[:, :, None] * d2theta_dx2_vec 

#     offset = np.array([0, 1, 0, 1, 0, 1])
#     cols   = np.repeat(H*2, 2, axis=1) + offset       # shape (|E|,6)
#     cols2  = np.repeat(cols[:, None, :], 6, axis=1)   # (|E|,6,6)
#     rows   = cols2.transpose(0, 2, 1) 

#     term_2 = sp.sparse.coo_matrix(
#             (term_2_compact.ravel(), (rows.ravel(), cols2.ravel())),
#             shape=(x.size, x.size))
    
#     # for name, arr in {
#     #     "theta": theta,
#     #     "dtheta_dx": dtheta_dx,
#     #     "d2theta_dx2_vec": d2theta_dx2_vec
#     # }.items():
#     #     if has_nan_or_inf(arr):
#     #         raise ValueError(f"{name} contains NaN or inf")
            
#     Q = term_1 + term_2 
#     return Q