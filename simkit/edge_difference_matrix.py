import numpy as np
import scipy as sp

def edge_difference_matrix(E, nv):
    '''
    E is a list of edges, nv is the number of vertices
    returns a sparse matrix A such that A*x is the vector of edge differences
    
    In other words, if x is a vector containing scalar values of each vertex,
    then A*x returns the value at each edge of the difference between the two vertices val[e[0]] - val[e[1]]
    '''
    ii = E.copy().flatten()
    jj = np.repeat(np.arange(E.shape[0])[:, None], 2, axis=1).flatten()
    vv = np.ones((E.shape[0], 2))
    vv[:, -1] = -1
    A =  sp.sparse.csc_matrix((vv.flatten(), (ii, jj)), shape=(nv, E.shape[0])).T

    return A