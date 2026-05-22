"""Per-edge lengths of a mesh graph."""

import numpy as np


def edge_lengths(X: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Euclidean length of each mesh edge.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Vertex positions.
    E : np.ndarray (m, 2)
        Edge vertex indices.

    Returns
    -------
    L : np.ndarray (m,)
        Length of each edge.
    """
    L = np.linalg.norm(X[E[:, 0], :] - X[E[:, 1], :], axis=1)
    return L
