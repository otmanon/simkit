import scipy as sp
import numpy as np

from ..pairwise_displacement import pairwise_displacement



def contact_springs_plane_energy(X, k, p, n, M=None, return_contact_inds=False):
    """ 
    Compute the Hessian of the contact springs with the ground plane.
    
    Parameters
    ----------
    X : np.ndarray
        The positions of the contact points.
    
    k : float
        The stiffness of the contact springs.
    p : np.ndarray
        The position of a point on the ground plane.
    n : np.ndarray
        The normal vector of the ground plane.
    M : np.ndarray, optional
        The mass matrix of the contact points.
        Defaults to the identity matrix.
    return_contact_inds : bool, optional
        Whether to return the indices of the contact points.
        Defaults to False.

    """

    if M is None:
        M = sp.sparse.identity(X.shape[0])

    energy_density =  np.zeros(X.shape[0])
    # if the contact point is above the ground plane, the energy is 0
    if p.ndim==1:
        p = p[None, :]
    D = pairwise_displacement(X, p)
    offset = D @ n
    # if the contact point is above the ground plane, the energy is 0
    under_ground_plane = (offset < 0).flatten() 
    num_contacts = under_ground_plane.sum()
    dim = X.shape[1]
    contacting_inds = None
    if num_contacts > 0:
        m = M.diagonal()
        m = m * under_ground_plane
        MI = sp.sparse.diags(m[under_ground_plane])

        # r = offset[under_ground_plane] * n
        contacting_inds = np.where(under_ground_plane)[0][:, None]
        I = np.tile(np.arange(num_contacts)[:, None], (1, dim))
        J = dim*contacting_inds +  np.arange(dim)[None, :]
        V = np.tile(n, (num_contacts, 1))
        N = sp.sparse.csc_matrix((V.flatten(), (I.flatten(), J.flatten())), (num_contacts, X.shape[0]* dim))

        x = (X).reshape(-1, 1)
        error = N @ x -  (V[:, None, :] @ p.T[None, :, :])[:, 0]
        energy_density[under_ground_plane] = 0.5 * ((MI @ error ) * error).sum(axis=1) * k


    energy = energy_density.sum()
    if return_contact_inds:
        return energy, contacting_inds
    else:
        return energy