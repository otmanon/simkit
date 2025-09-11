import scipy as sp
import numpy as np

from simkit.edge_lengths import edge_lengths


def edge_laplacian(X, E):
    """
    Computes the Laplacian of an edge mesh defined by nodes X and edges E.
    
    Specifically, the basis functions are hat functions defined as one on the vertices that decay to zero
    as you move away from the vertex along the edge.
    
    Parameters
    ----------
    X : (n, dim) array
        Nodes of the mesh
    E : (m, 2) array
        Edges of the mesh
        
    Returns
    -------
    L : (n, n) array
        Edge Laplacian of the mesh
        
    """
    l = edge_lengths(X, E)
    li = 1 / l
    I = np.repeat(np.arange((E.shape[0]))[:, None], 2, axis=1)
    J = E
    V = np.hstack([-li[:, None], li[:, None]])
    G = sp.sparse.csc_matrix((V.flatten(), (I.flatten(), J.flatten())), shape=(E.shape[0], X.shape[0]))
    A = sp.sparse.diags(l)
    L = G.T @ A @ G
    return L