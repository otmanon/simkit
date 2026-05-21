# import numpy as np
# import scipy as sp

# from ..pairwise_displacement import pairwise_displacement


# def contact_springs_plane_energy(X, k, p, n, M=None, return_contact_inds=False):
#     """ 
#     Compute the Hessian of the contact springs with the ground plane.
    
#     Parameters
#     ----------
#     X : np.ndarray
#         The positions of the contact points.
    
#     k : float
#         The stiffness of the contact springs.
#     p : np.ndarray
#         The position of a point on the ground plane.
#     n : np.ndarray
#         The normal vector of the ground plane.
#     M : np.ndarray, optional
#         The mass matrix of the contact points.
#         Defaults to the identity matrix.
#     return_contact_inds : bool, optional
#         Whether to return the indices of the contact points.
#         Defaults to False.

#     """

#     if M is None:
#         M = sp.sparse.identity(X.shape[0])

#     energy_density =  np.zeros(X.shape[0])
#     # if the contact point is above the ground plane, the energy is 0
#     if p.ndim==1:
#         p = p[None, :]
#     D = pairwise_displacement(X, p)
#     offset = D @ n
#     # if the contact point is above the ground plane, the energy is 0
#     under_ground_plane = (offset < 0).flatten() 
#     num_contacts = under_ground_plane.sum()
#     dim = X.shape[1]
#     contacting_inds = None
#     if num_contacts > 0:
#         m = M.diagonal()
#         m = m * under_ground_plane
#         MI = sp.sparse.diags(m[under_ground_plane])

#         # r = offset[under_ground_plane] * n
#         contacting_inds = np.where(under_ground_plane)[0][:, None]
#         I = np.tile(np.arange(num_contacts)[:, None], (1, dim))
#         J = dim*contacting_inds +  np.arange(dim)[None, :]
#         V = np.tile(n, (num_contacts, 1))
#         N = sp.sparse.csc_matrix((V.flatten(), (I.flatten(), J.flatten())), (num_contacts, X.shape[0]* dim))

#         x = (X).reshape(-1, 1)
#         error = N @ x -  (V[:, None, :] @ p.T[None, :, :])[:, 0]
#         energy_density[under_ground_plane] = 0.5 * ((MI @ error ) * error).sum(axis=1) * k


#     energy = energy_density.sum()
#     if return_contact_inds:
#         return energy, contacting_inds
#     else:
#         return energy
    
    
    

# def contact_springs_plane_gradient(X, k, p, n, M=None, return_contact_inds=False):
#     """
#     Compute the energy of the contact springs with the ground plane.
    
#     Parameters
#     ----------
#     x : np.ndarray
#         The positions of the contact points.
#     height : float
#         The height of the ground plane.
    
#     Returns
#     -------
#     float
#         The energy of the contact springs.
#     """

#     if M is None:
#         M = sp.sparse.identity(X.shape[0])

#     gradient =  np.zeros(X.shape)


#     # if the contact point is above the ground plane, the energy is 0
#     if p.ndim==1:
#         p = p[None, :]
#     D = pairwise_displacement(X, p)
#     offset = D @ n

#     # if the contact point is above the ground plane, the energy is 0
#     under_ground_plane = (offset < 0).flatten() 
#     contacting_inds = None
#     under_ground_plane = (offset < 0).flatten() 
#     num_contacts = under_ground_plane.sum()
#     dim = X.shape[1]
    
#     if np.sum(under_ground_plane) > 0:
            
#         m = M.diagonal()
#         m = m * under_ground_plane
#         MI = sp.sparse.diags(m[under_ground_plane])

#         # r = offset[under_ground_plane] * n
#         # gradient[under_ground_plane, :] = ((MI @ r )) * k

#         contacting_inds = np.where(under_ground_plane)[0][:, None]
#         I = np.tile(np.arange(num_contacts)[:, None], (1, dim))
#         J = dim*contacting_inds +  np.arange(dim)[None, :]
#         V = np.tile(n, (num_contacts, 1))
#         N = sp.sparse.csc_matrix((V.flatten(), (I.flatten(), J.flatten())), (num_contacts, X.shape[0]* dim))


#         x = X.reshape(-1, 1)

#         NM = N.T @ MI
#         NMN = NM @ N
        
#         gradient = k* (NMN @ x - NM @ ((V[:, None, :] @ p.T[None, :, :])[:, 0]))

#     gradient = gradient.reshape(-1, 1)
#     if return_contact_inds:
#         return gradient, contacting_inds
#     else:
#         return gradient
    
    

# def contact_springs_plane_hessian(X, k, p, n, M=None, return_contact_inds=False):
#     """
#     Compute the energy of the contact springs with the ground plane.
    
#     Parameters
#     ----------
#     x : np.ndarray
#         The positions of the contact points.
#     height : float
#         The height of the ground plane.
    
#     Returns
#     -------
#     float
#         The energy of the contact springs.
#     """

#     if M is None:
#         M = sp.sparse.identity(X.shape[0])


#     # if the contact point is above the ground plane, the energy is 0
#     if p.ndim==1:
#         p = p[None, :]
#     D = pairwise_displacement(X, p)
#     offset = D @ n
#     # if the contact point is above the ground plane, the energy is 0
#     under_ground_plane = (offset < 0).flatten() 
#     num_contacts = under_ground_plane.sum()
#     dim = X.shape[1]
#     contacting_inds = None
#     H = sp.sparse.csc_matrix((X.shape[0]*X.shape[1], X.shape[0]*X.shape[1]))
#     if np.sum(under_ground_plane) > 0:
#         m = M.diagonal()
#         m = m * under_ground_plane
#         MI = sp.sparse.diags(m[under_ground_plane])

#         contacting_inds = np.where(under_ground_plane)[0][:, None]
#         I = np.tile(np.arange(num_contacts)[:, None], (1, dim))
#         J = dim*contacting_inds +  np.arange(dim)[None, :]
#         V = np.tile(n, (num_contacts, 1))
#         N = sp.sparse.csc_matrix((V.flatten(), (I.flatten(), J.flatten())), (num_contacts, X.shape[0]* dim))

#         NM = N.T @ MI
#         NMN = NM @ N

#         H = k  * NMN
#     if return_contact_inds:
#         return H, contacting_inds
#     else:
#         return H


"""Penalty contact springs against a ground plane.

A one-sided quadratic penalty: points that cross below the plane are pulled
back along the plane normal. This is an external potential, not a per-element
material energy, so there is no element / ``F`` tier. The functions take vertex
positions directly and detect the active (penetrating) set internally.

Each function optionally returns the indices of the contacting points via
``return_contact_inds``.
"""

from typing import Optional, Tuple, Union

import numpy as np
import scipy as sp

from ..pairwise_displacement import pairwise_displacement


def _contact_normal_matrix(under: np.ndarray, n: np.ndarray, num_verts: int, dim: int) -> Tuple[sp.sparse.csc_matrix, np.ndarray, np.ndarray]:
    """Build the sparse normal-projection matrix for the active contact set.

    Parameters
    ----------
    under : np.ndarray (num_verts,) bool
        Mask of points penetrating the plane.
    n : np.ndarray (dim,)
        Plane normal.
    num_verts : int
        Total number of vertices.
    dim : int
        Spatial dimension.

    Returns
    -------
    N : scipy.sparse.csc_matrix (num_contacts, num_verts*dim)
        Normal-projection matrix.
    contacting_inds : np.ndarray (num_contacts, 1)
        Indices of contacting vertices.
    V : np.ndarray (num_contacts, dim)
        Tiled plane normals.
    """
    num_contacts = int(under.sum())
    contacting_inds = np.where(under)[0][:, None]
    I = np.tile(np.arange(num_contacts)[:, None], (1, dim))
    Jc = dim * contacting_inds + np.arange(dim)[None, :]
    V = np.tile(n, (num_contacts, 1))
    N = sp.sparse.csc_matrix(
        (V.flatten(), (I.flatten(), Jc.flatten())),
        (num_contacts, num_verts * dim),
    )
    return N, contacting_inds, V


def contact_springs_plane_energy(X: np.ndarray, k: float, p: np.ndarray, n: np.ndarray, M: Optional[sp.sparse.spmatrix] = None, return_contact_inds: bool = False) -> Union[float, Tuple[float, np.ndarray]]:
    """Penalty energy for contact with a ground plane.

    Parameters
    ----------
    X : np.ndarray (num_verts, dim)
        Vertex positions.
    k : float
        Penalty stiffness.
    p : np.ndarray (dim,) or (1, dim)
        A point on the plane.
    n : np.ndarray (dim,)
        Plane normal.
    M : scipy.sparse matrix (num_verts, num_verts), optional
        Mass matrix. Defaults to the identity.
    return_contact_inds : bool, optional
        If ``True``, also return the contacting vertex indices.

    Returns
    -------
    energy : float
        Total contact energy.
    contacting_inds : np.ndarray (num_contacts, 1), optional
        Returned only if ``return_contact_inds`` is ``True``.
    """
    if M is None:
        M = sp.sparse.identity(X.shape[0])
    if p.ndim == 1:
        p = p[None, :]

    energy_density = np.zeros(X.shape[0])
    offset = pairwise_displacement(X, p) @ n
    under = (offset < 0).flatten()
    dim = X.shape[1]
    contacting_inds = None

    if under.sum() > 0:
        m = M.diagonal() * under
        MI = sp.sparse.diags(m[under])
        N, contacting_inds, V = _contact_normal_matrix(under, n, X.shape[0], dim)
        x = X.reshape(-1, 1)
        error = N @ x - (V[:, None, :] @ p.T[None, :, :])[:, 0]
        energy_density[under] = 0.5 * ((MI @ error) * error).sum(axis=1) * k

    energy = float(energy_density.sum())
    if return_contact_inds:
        return energy, contacting_inds
    return energy


def contact_springs_plane_gradient(X: np.ndarray, k: float, p: np.ndarray, n: np.ndarray, M: Optional[sp.sparse.spmatrix] = None, return_contact_inds: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Penalty gradient for contact with a ground plane.

    Parameters
    ----------
    X : np.ndarray (num_verts, dim)
        Vertex positions.
    k : float
        Penalty stiffness.
    p : np.ndarray (dim,) or (1, dim)
        A point on the plane.
    n : np.ndarray (dim,)
        Plane normal.
    M : scipy.sparse matrix (num_verts, num_verts), optional
        Mass matrix. Defaults to the identity.
    return_contact_inds : bool, optional
        If ``True``, also return the contacting vertex indices.

    Returns
    -------
    gradient : np.ndarray (num_verts*dim, 1)
        Assembled gradient.
    contacting_inds : np.ndarray (num_contacts, 1), optional
        Returned only if ``return_contact_inds`` is ``True``.
    """
    if M is None:
        M = sp.sparse.identity(X.shape[0])
    if p.ndim == 1:
        p = p[None, :]

    gradient = np.zeros(X.shape)
    offset = pairwise_displacement(X, p) @ n
    under = (offset < 0).flatten()
    dim = X.shape[1]
    contacting_inds = None

    if under.sum() > 0:
        m = M.diagonal() * under
        MI = sp.sparse.diags(m[under])
        N, contacting_inds, V = _contact_normal_matrix(under, n, X.shape[0], dim)
        x = X.reshape(-1, 1)
        NM = N.T @ MI
        NMN = NM @ N
        gradient = k * (NMN @ x - NM @ ((V[:, None, :] @ p.T[None, :, :])[:, 0]))

    gradient = gradient.reshape(-1, 1)
    if return_contact_inds:
        return gradient, contacting_inds
    return gradient


def contact_springs_plane_hessian(X: np.ndarray, k: float, p: np.ndarray, n: np.ndarray, M: Optional[sp.sparse.spmatrix] = None, return_contact_inds: bool = False) -> Union[sp.sparse.spmatrix, Tuple[sp.sparse.spmatrix, np.ndarray]]:
    """Penalty Hessian for contact with a ground plane.

    Parameters
    ----------
    X : np.ndarray (num_verts, dim)
        Vertex positions.
    k : float
        Penalty stiffness.
    p : np.ndarray (dim,) or (1, dim)
        A point on the plane.
    n : np.ndarray (dim,)
        Plane normal.
    M : scipy.sparse matrix (num_verts, num_verts), optional
        Mass matrix. Defaults to the identity.
    return_contact_inds : bool, optional
        If ``True``, also return the contacting vertex indices.

    Returns
    -------
    H : scipy.sparse matrix (num_verts*dim, num_verts*dim)
        Assembled Hessian. PSD by construction.
    contacting_inds : np.ndarray (num_contacts, 1), optional
        Returned only if ``return_contact_inds`` is ``True``.
    """
    if M is None:
        M = sp.sparse.identity(X.shape[0])
    if p.ndim == 1:
        p = p[None, :]

    offset = pairwise_displacement(X, p) @ n
    under = (offset < 0).flatten()
    dim = X.shape[1]
    contacting_inds = None
    H = sp.sparse.csc_matrix((X.shape[0] * dim, X.shape[0] * dim))

    if under.sum() > 0:
        m = M.diagonal() * under
        MI = sp.sparse.diags(m[under])
        N, contacting_inds, V = _contact_normal_matrix(under, n, X.shape[0], dim)
        H = k * (N.T @ MI @ N)

    if return_contact_inds:
        return H, contacting_inds
    return H