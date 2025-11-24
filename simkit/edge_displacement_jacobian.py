
import numpy as np
import scipy as sp

def edge_displacement_jacobian(X, E):
    '''
    Compute the Jacobian of the edge displacement with respect to the vertex positions.
    
    Args:
        X (np.ndarray): n x dim The vertex positions of the mesh.
        E (np.ndarray): m x 2 The edge indices of the mesh.
    Returns:   
        dl_dx (sp.sparse.csr_matrix) : m x n The Jacobian of the edge length with respect to the vertex positions.
    '''
    dim = X.shape[1]

    ones = np.ones((E.shape[0], 1))
    vals = np.concatenate([ones, -ones], axis=1) 
      
    # sparse matrix
    ii = np.repeat(np.arange(E.shape[0])[:, None], 2, axis=1)
    jj = E

    J = sp.sparse.csc_matrix((vals.flatten(), (ii.flatten(), jj.flatten())), (E.shape[0], X.shape[0]))

    return J