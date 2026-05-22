"""Sparse face–edge adjacency for triangle meshes."""

from __future__ import annotations

import numpy as np
import scipy as sp


def edge_face_adjacency(F: np.ndarray, E: np.ndarray) -> sp.sparse.csc_matrix:
    """Face–edge incidence matrix for a triangle mesh.

    Entry ``(f, e)`` is 1 when face ``f`` contains undirected edge ``e``.

    Parameters
    ----------
    F : np.ndarray (|F|, 3)
        Triangle face indices.
    E : np.ndarray (|E|, 2)
        Undirected edge vertex pairs (sorted internally).

    Returns
    -------
    A : scipy.sparse.csc_matrix (|F|, |E|)
        Face–edge adjacency; ``A[f, e] = 1`` if edge ``e`` lies on face ``f``.
    """
    F = np.asarray(F, dtype=int)
    E = np.sort(np.asarray(E, dtype=int), axis=1)

    face_edges = np.stack([
        F[:, [0, 1]],
        F[:, [1, 2]],
        F[:, [2, 0]],
    ], axis=1)
    face_edges = np.sort(face_edges, axis=2).reshape(-1, 2)

    matches = np.all(face_edges[:, None, :] == E[None, :, :], axis=2)
    face_idx, edge_idx = np.nonzero(matches)

    face_idx = face_idx // 3

    data = np.ones_like(face_idx)
    A = sp.sparse.csc_matrix((data, (face_idx, edge_idx)), shape=(len(F), len(E)))
    return A
