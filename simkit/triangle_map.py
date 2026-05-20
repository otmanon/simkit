import numpy as np
import scipy as sp


def triangle_map(F, nv):
    '''
    Matrix that goes from vertices to a flattened list of triangles
    
    if faces are
    [v0, v1, v2]
    [v0, v2, v3]
    
    multiplying vertices by map gets you
    
    v0
    v1
    v2
    v0
    v2
    v3
    
    Returns a map M so that
    
    (M @ vertices).reshape(-1, 3) 
    ''' 
    J = F.flatten()
    I = np.arange(J.shape[0])
    V = np.ones(I.shape[0])
    M = sp.sparse.csc_matrix((V, (I, J)), (3*F.shape[0], nv))
    return M