import numpy as np
import igl

from .edge_face_adjacency import edge_face_adjacency
def dihedral_wedges(F):
    """
    From x1, x2, x3, x4, return the dihedral wedges.

    Args:
        F (np.ndarray): Face indices

    Returns:
        np.ndarray: Dihedral wedges
        x1 (np.ndarray): Vertex 1
        x2 (np.ndarray): Vertex 2
        x3 (np.ndarray): Vertex 3
        x4 (np.ndarray): Vertex 4
        
    """

    E = igl.edges(F)
   
    A = edge_face_adjacency(F, E)
    
    faces_adjacent = A.sum(axis=0)
    valid_edges = np.where(faces_adjacent == 2)[1]
    
    A_sub = A[:, valid_edges]
    
    i, j = A_sub.T.nonzero()    

    face_pairs = j.reshape(-1, 2)
    F1 = F[face_pairs[:, 0]]
    F2 = F[face_pairs[:, 1]]
 
    shared_mask_F1 = np.zeros((F1.shape[0], 3), dtype=bool)
    shared_mask_F2 = np.zeros((F2.shape[0], 3), dtype=bool)
    for i in range(3):
        shared_mask_F1 += np.repeat(F1[:, [i]], 3, axis=1) == F2
        shared_mask_F2 += np.repeat(F2[:, [i]], 3, axis=1) == F1
            
    i1i2 = F1[shared_mask_F2].reshape(-1, 2) 
    i0 = F1[~shared_mask_F2]
    i3 = F2[~shared_mask_F1]
    D = np.column_stack([i0, i1i2, i3])
    
    return D