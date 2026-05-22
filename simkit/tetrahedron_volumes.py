"""Signed volumes of tetrahedral mesh elements.

Each tet volume is ``det([e1, e2, e3]) / 6`` where ``e1, e2, e3`` are edge
vectors from the first corner to the other three.
"""

import numpy as np


def tetrahedron_volumes(X: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Per-tet signed volume from vertex positions and connectivity.

    Parameters
    ----------
    X : np.ndarray (n, 3)
        Vertex positions.
    T : np.ndarray (m, 4)
        Tetrahedron vertex indices; corner 0 is the reference for the three
        edge vectors.

    Returns
    -------
    V : np.ndarray (m,)
        Signed volume of each tetrahedron.
    """
    e = X[T[:, 1:]] - X[T[:, [0]]]   # (m, 3, 3): rows e1, e2, e3
    return (np.linalg.det(e)) / 6.0
