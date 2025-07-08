import scipy as sp
import numpy as np
from simkit import pairwise_distance

def contact_springs_sphere_hessian(X, k, p, r, M=None):
    """
    Compute the energy of the contact springs with the ground plane.
    
    """

    if M is None:
        M = sp.sparse.identity(X.shape[0])



    # if the contact point is above the ground plane, the energy is 0
    if p.ndim==1:
        p = p[None, :]
    D = pairwise_distance(X, p)

    # if the contact point is above the ground plane, the energy is 0
    inside_sphere = (D < r).flatten() 
    


    H = sp.sparse.csc_matrix((X.shape[0]*X.shape[1], X.shape[0]*X.shape[1]))
   
    # if inside_sphere.sum() > 0:
    #     m = M.diagonal()

    #     mI = np.zeros(X.shape)

    #     mI[inside_sphere, :] = m[inside_sphere, None]

    #     MI = sp.sparse.diags(mI.flatten()) * k
    #     H = MI


    num_contacts = inside_sphere.sum()

    dim = X.shape[1]
    if num_contacts > 0:
        m = M.diagonal()
        MI = sp.sparse.diags(m[inside_sphere])

        d = X[inside_sphere] - p
        length = np.linalg.norm(d, axis=1)[:, None] # length

        n =  d / length # normal of contact frame
        
        # built normal matrx
        contacting_inds = np.where(inside_sphere)[0][:, None]
        I = np.tile(np.arange(num_contacts)[:, None], (1, dim))
        J = dim*contacting_inds +  np.arange(dim)[None, :]
        V = n
        N = sp.sparse.csc_matrix((V.flatten(), (I.flatten(), J.flatten())), (num_contacts, X.shape[0]* dim))

        
        NM = N.T @ MI
        NMN = NM @ N
        H = k  * NMN

    return H