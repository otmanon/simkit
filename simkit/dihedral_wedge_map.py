import numpy as np
import scipy as sp


def dihedral_wedge_map(D, nv):
    '''
    Write me a function called dihedral_wedge_map that takes as input a list of dihedral wedge indices D (nd x 4), 
    and returns a list of edge indices E (nd x 2), where nd is the number of edges that have valid dihedral angles in
    the mesh (no boundary edges). 
    The columns of E are organized by (x1, x2) where x1 and x2 are the shared vertices.
    ''' 
    J = D.flatten()
    I = np.arange(J.shape[0])
    V = np.ones(I.shape[0])
    M = sp.sparse.csc_matrix((V, (I, J)), (4*D.shape[0], nv))
    return M