import scipy as sp
import numpy as np

from ..pairwise_displacement import pairwise_displacement



def contact_springs_plane_hessian(X, k, p, n, M=None, return_contact_inds=False):
    """
    Compute the energy of the contact springs with the ground plane.
    
    Parameters
    ----------
    x : np.ndarray
        The positions of the contact points.
    height : float
        The height of the ground plane.
    
    Returns
    -------
    float
        The energy of the contact springs.
    """

    if M is None:
        M = sp.sparse.identity(X.shape[0])


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
    H = sp.sparse.csc_matrix((X.shape[0]*X.shape[1], X.shape[0]*X.shape[1]))
    if np.sum(under_ground_plane) > 0:
        m = M.diagonal()
        m = m * under_ground_plane
        MI = sp.sparse.diags(m[under_ground_plane])

        contacting_inds = np.where(under_ground_plane)[0][:, None]
        I = np.tile(np.arange(num_contacts)[:, None], (1, dim))
        J = dim*contacting_inds +  np.arange(dim)[None, :]
        V = np.tile(n, (num_contacts, 1))
        N = sp.sparse.csc_matrix((V.flatten(), (I.flatten(), J.flatten())), (num_contacts, X.shape[0]* dim))

        NM = N.T @ MI
        NMN = NM @ N

        H = k  * NMN
    if return_contact_inds:
        return H, contacting_inds
    else:
        return H