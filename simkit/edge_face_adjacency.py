import numpy as np
import scipy as sp
def edge_face_adjacency(F, E):
    F = np.asarray(F, dtype=int)
    E = np.sort(np.asarray(E, dtype=int), axis=1)
    
    # Extract all edges from faces
    face_edges = np.stack([
        F[:, [0,1]],
        F[:, [1,2]],
        F[:, [2,0]]
    ], axis=1)  # shape (|F|, 3, 2)
    face_edges = np.sort(face_edges, axis=2).reshape(-1,2)  # (|F|*3, 2)

    # Broadcast comparison to find matching edges
    matches = np.all(face_edges[:, None, :] == E[None, :, :], axis=2)  # (|F|*3, |E|)
    face_idx, edge_idx = np.nonzero(matches)
    
    # Map back to face numbers (since each face has 3 edges)
    face_idx = face_idx // 3

    data = np.ones_like(face_idx)
    A = sp.sparse.csc_matrix((data, (face_idx, edge_idx)), shape=(len(F), len(E)))
    return A
    