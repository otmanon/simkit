
import numpy as np
import scipy as sp

def edge_length_jacobian(X, E):
    '''
    Compute the Jacobian of the edge length with respect to the vertex positions.
    
    Args:
        X (np.ndarray): n x dim The vertex positions of the mesh.
        E (np.ndarray): m x 2 The edge indices of the mesh.
        
    Returns:   
        dl_dx (sp.sparse.csr_matrix) : m x n*dim The Jacobian of the edge length with respect to the vertex positions.
    '''
    dim = X.shape[1]
    p = X[E[:, 0], :] - X[E[:, 1], :]
    p_norm = np.linalg.norm(p, axis=1)[:, None]
    d = p / p_norm
       
    dldx = np.concatenate([d, -d], axis=1) 
      
    # sparse matrix
    ii = np.repeat(np.arange(E.shape[0])[:, None], dim*2, axis=1)
    jj = np.repeat(E[:, :]*dim, dim, axis=1) + np.tile(np.arange(dim), 2)
    # ii = np.repeat(E[:, None]*self.dim, self.dim, axis=1) + np.arange(self.dim)
    vals = dldx

    dl_dx = sp.sparse.csc_matrix((vals.flatten(), (ii.flatten(), jj.flatten())), (E.shape[0], X.shape[0]*dim))

    return dl_dx