# import numpy as np
# import scipy as sp
# from ..pairwise_distance import pairwise_distance

# def contact_springs_sphere_energy(X, k, p, r, M=None):
#     """
#     Compute the energy of the contact springs with the ground plane.
    
#     """

#     if M is None:
#         M = sp.sparse.identity(X.shape[0])

#     energy_density =  np.zeros(X.shape[0])
#     # if the contact point is above the ground plane, the energy is 0
#     if p.ndim==1:
#         p = p[None, :]
#     D = pairwise_distance(X, p)

#     # if the contact point is above the ground plane, the energy is 0
#     inside_sphere = (D < r).flatten()     

#     # if inside_sphere.sum() > 0:
#     #     m = M.diagonal()
#     #     MI = sp.sparse.diags(m[inside_sphere])
#     #     d = X[inside_sphere] - p
#     #     length = np.linalg.norm(d, axis=1)[:, None]
#     #     l = X[inside_sphere] - (d * r / length + p)

#     #     energy_density[inside_sphere] = 0.5 * ((MI @ l ) * l).sum(axis=1) * k

#     num_contacts = inside_sphere.sum()
#     dim = X.shape[1]
#     if num_contacts > 0:
#         m = M.diagonal()
#         MI = sp.sparse.diags(m[inside_sphere])
#         d = X[inside_sphere] - p                    # displacement from center
#         length = np.linalg.norm(d, axis=1)[:, None] # distance from center (shouldn't this just be D)
        
#         n =  d / length                             # normal of contact frame

#         # built normal matrx
#         contacting_inds = np.where(inside_sphere)[0][:, None]
#         I = np.tile(np.arange(num_contacts)[:, None], (1, dim))
#         J = dim*contacting_inds +  np.arange(dim)[None, :]
#         V = n
#         N = sp.sparse.csc_matrix((V.flatten(), (I.flatten(), J.flatten())), (num_contacts, X.shape[0]* dim))

#         x = (X).reshape(-1, 1)

#         error = N @ x - (n[:, None, :] @ p.T[None, :, :])[:, 0] - r
        
#         energy_density = 0.5 * k* (MI @ error) * error 

#     energy = energy_density.sum()
#     return energy



# def contact_springs_sphere_gradient(X, k, p, r, M=None):
#     """
#     Compute the energy of the contact springs with the ground plane.
#     """

#     if M is None:
#         M = sp.sparse.identity(X.shape[0])

    
#     gradient =  np.zeros(X.shape)

#     # if the contact point is above the ground plane, the energy is 0
#     if p.ndim==1:
#         p = p[None, :]
#     D = pairwise_distance(X, p)

#     # if the contact point is above the ground plane, the energy is 0
#     inside_sphere = (D < r).flatten() 
    
#     # this is just trying to set position equal to closest point on sphere
#     # if inside_sphere.sum() > 0:
#     #     m = M.diagonal()
#     #     MI = sp.sparse.diags(m[inside_sphere])
#     #     d = X[inside_sphere] - p
#     #     length = np.linalg.norm(d, axis=1)[:, None]
#     #     l = X[inside_sphere] - (d * r / length + p)
#     #     gradient[inside_sphere] =   (MI @ l ) * k


#     # this is trying to set normal position to closest point on sphere, leaving tangent free
#     num_contacts = inside_sphere.sum()

#     dim = X.shape[1]
#     if num_contacts > 0:
#         m = M.diagonal()
#         MI = sp.sparse.diags(m[inside_sphere])

#         d = X[inside_sphere] - p
#         length = np.linalg.norm(d, axis=1)[:, None] # length

#         n =  d / length # normal of contact frame
        
#         r0 = np.ones((num_contacts, 1)) * r
#         # built normal matrx
#         contacting_inds = np.where(inside_sphere)[0][:, None]
#         I = np.tile(np.arange(num_contacts)[:, None], (1, dim))
#         J = dim*contacting_inds +  np.arange(dim)[None, :]
#         V = n
#         N = sp.sparse.csc_matrix((V.flatten(), (I.flatten(), J.flatten())), (num_contacts, X.shape[0]* dim))


#         x = X.reshape(-1, 1)
        

#         NM = N.T @ MI
#         NMN = NM @ N
        
#         gradient = k* (NMN @ x - NM @ (r0 +  (n[:, None, :] @ p.T[None, :, :])[:, 0]))

#     gradient = gradient.reshape(-1, 1)

#     return gradient





# def contact_springs_sphere_hessian(X, k, p, r, M=None):
#     """
#     Compute the energy of the contact springs with the ground plane.
    
#     """

#     if M is None:
#         M = sp.sparse.identity(X.shape[0])



#     # if the contact point is above the ground plane, the energy is 0
#     if p.ndim==1:
#         p = p[None, :]
#     D = pairwise_distance(X, p)

#     # if the contact point is above the ground plane, the energy is 0
#     inside_sphere = (D < r).flatten() 
    


#     H = sp.sparse.csc_matrix((X.shape[0]*X.shape[1], X.shape[0]*X.shape[1]))
   
#     # if inside_sphere.sum() > 0:
#     #     m = M.diagonal()

#     #     mI = np.zeros(X.shape)

#     #     mI[inside_sphere, :] = m[inside_sphere, None]

#     #     MI = sp.sparse.diags(mI.flatten()) * k
#     #     H = MI


#     num_contacts = inside_sphere.sum()

#     dim = X.shape[1]
#     if num_contacts > 0:
#         m = M.diagonal()
#         MI = sp.sparse.diags(m[inside_sphere])

#         d = X[inside_sphere] - p
#         length = np.linalg.norm(d, axis=1)[:, None] # length

#         n =  d / length # normal of contact frame
        
#         # built normal matrx
#         contacting_inds = np.where(inside_sphere)[0][:, None]
#         I = np.tile(np.arange(num_contacts)[:, None], (1, dim))
#         J = dim*contacting_inds +  np.arange(dim)[None, :]
#         V = n
#         N = sp.sparse.csc_matrix((V.flatten(), (I.flatten(), J.flatten())), (num_contacts, X.shape[0]* dim))

        
#         NM = N.T @ MI
#         NMN = NM @ N
#         H = k  * NMN

#     return H


"""Penalty contact springs against a sphere.

A one-sided quadratic penalty: points that fall inside the sphere are pushed
back out along the (per-point) contact normal, leaving the tangential component
free. This is an external potential, not a per-element material energy, so there
is no element / ``F`` tier; the functions take vertex positions directly and
detect the active (penetrating) set internally.
"""

from typing import Optional, Tuple

import numpy as np
import scipy as sp

from ..pairwise_distance import pairwise_distance


def _sphere_normal_matrix(inside: np.ndarray, X: np.ndarray, p: np.ndarray, num_verts: int, dim: int) -> Tuple[sp.sparse.csc_matrix, np.ndarray]:
    """Build the sparse normal-projection matrix for the active contact set.

    Parameters
    ----------
    inside : np.ndarray (num_verts,) bool
        Mask of points inside the sphere.
    X : np.ndarray (num_verts, dim)
        Vertex positions.
    p : np.ndarray (1, dim)
        Sphere center.
    num_verts : int
        Total number of vertices.
    dim : int
        Spatial dimension.

    Returns
    -------
    N : scipy.sparse.csc_matrix (num_contacts, num_verts*dim)
        Normal-projection matrix.
    n : np.ndarray (num_contacts, dim)
        Per-contact outward normals.
    """
    num_contacts = int(inside.sum())
    d = X[inside] - p
    length = np.linalg.norm(d, axis=1)[:, None]
    n = d / length
    contacting_inds = np.where(inside)[0][:, None]
    I = np.tile(np.arange(num_contacts)[:, None], (1, dim))
    Jc = dim * contacting_inds + np.arange(dim)[None, :]
    N = sp.sparse.csc_matrix(
        (n.flatten(), (I.flatten(), Jc.flatten())),
        (num_contacts, num_verts * dim),
    )
    return N, n


def contact_springs_sphere_energy(X: np.ndarray, k: float, p: np.ndarray, r: float, M: Optional[sp.sparse.spmatrix] = None) -> float:
    """Penalty energy for contact with a sphere.

    Parameters
    ----------
    X : np.ndarray (num_verts, dim)
        Vertex positions.
    k : float
        Penalty stiffness.
    p : np.ndarray (dim,) or (1, dim)
        Sphere center.
    r : float
        Sphere radius.
    M : scipy.sparse matrix (num_verts, num_verts), optional
        Mass matrix. Defaults to the identity.

    Returns
    -------
    energy : float
        Total contact energy.
    """
    if M is None:
        M = sp.sparse.identity(X.shape[0])
    if p.ndim == 1:
        p = p[None, :]

    D = pairwise_distance(X, p)
    inside = (D < r).flatten()
    dim = X.shape[1]
    energy_density = np.zeros((X.shape[0], 1))

    if inside.sum() > 0:
        m = M.diagonal()
        MI = sp.sparse.diags(m[inside])
        N, n = _sphere_normal_matrix(inside, X, p, X.shape[0], dim)
        x = X.reshape(-1, 1)
        error = N @ x - (n[:, None, :] @ p.T[None, :, :])[:, 0] - r
        energy_density = 0.5 * k * (MI @ error) * error

    return float(energy_density.sum())


def contact_springs_sphere_gradient(X: np.ndarray, k: float, p: np.ndarray, r: float, M: Optional[sp.sparse.spmatrix] = None) -> np.ndarray:
    """Penalty gradient for contact with a sphere.

    Parameters
    ----------
    X : np.ndarray (num_verts, dim)
        Vertex positions.
    k : float
        Penalty stiffness.
    p : np.ndarray (dim,) or (1, dim)
        Sphere center.
    r : float
        Sphere radius.
    M : scipy.sparse matrix (num_verts, num_verts), optional
        Mass matrix. Defaults to the identity.

    Returns
    -------
    gradient : np.ndarray (num_verts*dim, 1)
        Assembled gradient.
    """
    if M is None:
        M = sp.sparse.identity(X.shape[0])
    if p.ndim == 1:
        p = p[None, :]

    D = pairwise_distance(X, p)
    inside = (D < r).flatten()
    dim = X.shape[1]
    gradient = np.zeros(X.shape)

    if inside.sum() > 0:
        m = M.diagonal()
        MI = sp.sparse.diags(m[inside])
        N, n = _sphere_normal_matrix(inside, X, p, X.shape[0], dim)
        r0 = np.ones((int(inside.sum()), 1)) * r
        x = X.reshape(-1, 1)
        NM = N.T @ MI
        NMN = NM @ N
        gradient = k * (NMN @ x - NM @ (r0 + (n[:, None, :] @ p.T[None, :, :])[:, 0]))

    return gradient.reshape(-1, 1)


def contact_springs_sphere_hessian(X: np.ndarray, k: float, p: np.ndarray, r: float, M: Optional[sp.sparse.spmatrix] = None) -> sp.sparse.spmatrix:
    """Penalty Hessian for contact with a sphere.

    Parameters
    ----------
    X : np.ndarray (num_verts, dim)
        Vertex positions.
    k : float
        Penalty stiffness.
    p : np.ndarray (dim,) or (1, dim)
        Sphere center.
    r : float
        Sphere radius.
    M : scipy.sparse matrix (num_verts, num_verts), optional
        Mass matrix. Defaults to the identity.

    Returns
    -------
    H : scipy.sparse matrix (num_verts*dim, num_verts*dim)
        Assembled Hessian. PSD by construction.
    """
    if M is None:
        M = sp.sparse.identity(X.shape[0])
    if p.ndim == 1:
        p = p[None, :]

    D = pairwise_distance(X, p)
    inside = (D < r).flatten()
    dim = X.shape[1]
    H = sp.sparse.csc_matrix((X.shape[0] * dim, X.shape[0] * dim))

    if inside.sum() > 0:
        m = M.diagonal()
        MI = sp.sparse.diags(m[inside])
        N, n = _sphere_normal_matrix(inside, X, p, X.shape[0], dim)
        H = k * (N.T @ MI @ N)

    return H