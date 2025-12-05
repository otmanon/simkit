import numpy as np

import scipy.sparse as sp

def quadratic_dynamic_friction_energy(z, z_curr, tangents, S, k_dyn_friction ):
    """
    Quadratic dynamic friction energy.

    E = 0.5 * k_dyn_friction * (lagged_tangential_dir @ current_tangential_displacement)
    

    Parameters
    ----------
    z : array_like
        The current positions of the system. 
    z_dot : array_like
        The current velocity of the system.
    tangents : array_like
        The tangential directions of the contact points.
    S : array_like
        Sparse matrix mapping from configuration z to points in contact.
    k_dyn_friction : float
        The dynamic friction coefficient.

    """
    dim = tangents.shape[1]

    
    # v_rel = (S @ z_dot_curr).reshape(-1, dim)
    # v_rel_t = ((v_rel * tangents).sum(axis=1)[:, None] * tangents) # tangential velocity 
    
    # v_t_dir = v_rel_t / np.linalg.norm(v_rel_t, axis=1, keepdims=True)
    # v_t_dir = np.nan_to_num(v_t_dir, 0.0)
    Se = sp.kron(S, sp.identity(dim))
        
    v_t_dir = tangents
    # d = Se @ (z - z_curr)
    # e = 0.5 * k_dyn_friction * ((v_t_dir * d.reshape(-1, dim)).sum(axis=1) ** 2).sum()
  
    D = sp.block_diag(v_t_dir[:, None, :])
   
    # e2 = 0.5 * k_dyn_friction * (d.T @ D.T @ D @ d)
    
    J = D @ Se
    # e3 = 0.5 * k_dyn_friction * (z_curr.T @ (J.T @ J) @ z_curr -
    #                              2* z_curr.T @ (J.T @ J) @ z +
    #                               z.T @ (J.T @ J) @ z)
    
    u = z - z_curr
    e4 = 0.5 * k_dyn_friction * (u.T @ (J.T @ J) @ u)
    return  e4

    
    
    