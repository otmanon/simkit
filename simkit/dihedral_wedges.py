"""Extract dihedral hinge vertex sets from a triangle mesh."""

import numpy as np

from .edges import edges

from .edge_face_adjacency import edge_face_adjacency


def dihedral_wedges(F: np.ndarray) -> np.ndarray:
    """Per-interior-edge hinge vertices ``(x0, x1, x2, x3)``.

    For each interior edge shared by exactly two triangles, returns the two
    shared edge vertices plus the two opposite apex vertices. Boundary edges
    (touched by only one face) are dropped.

    Parameters
    ----------
    F : np.ndarray (f, 3)
        Triangle face indices.

    Returns
    -------
    D : np.ndarray (nd, 4)
        Hinge vertex indices, one row per interior edge. Columns are
        ``(x0, x1, x2, x3)`` where ``(x1, x2)`` is the shared edge and
        ``x0``, ``x3`` are the apex vertices of the two incident triangles.
    """
    E = edges(F)

    # Edge-face incidence; column e has a 1 in each face row touching edge e.
    A = edge_face_adjacency(F, E)

    # Keep only interior edges: those incident to exactly two faces.
    faces_adjacent = A.sum(axis=0)
    valid_edges = np.where(faces_adjacent == 2)[1]
    A_sub = A[:, valid_edges]

    # For each interior edge, recover its two incident face indices as a pair.
    _, j = A_sub.T.nonzero()
    face_pairs = j.reshape(-1, 2)
    F1 = F[face_pairs[:, 0]]
    F2 = F[face_pairs[:, 1]]

    # Mark, per face, which of its three vertices are shared with the other
    # face of the pair. Comparing each column of one face against all of the
    # other accumulates a boolean "is shared" mask.
    shared_mask_F1 = np.zeros((F1.shape[0], 3), dtype=bool)
    shared_mask_F2 = np.zeros((F2.shape[0], 3), dtype=bool)
    for i in range(3):
        shared_mask_F1 += np.repeat(F1[:, [i]], 3, axis=1) == F2
        shared_mask_F2 += np.repeat(F2[:, [i]], 3, axis=1) == F1

    # Shared edge vertices come from F1; the two apexes are the unshared
    # vertices of each face.
    i1i2 = F1[shared_mask_F2].reshape(-1, 2)   # the two shared edge vertices
    i0 = F1[~shared_mask_F2]                   # apex on face 1
    i3 = F2[~shared_mask_F1]                   # apex on face 2

    D = np.column_stack([i0, i1i2, i3])
    return D