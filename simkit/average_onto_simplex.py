"""Average per-vertex quantities onto the simplices that contain them."""

import numpy as np


def average_onto_simplex(A: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Average per-node values onto each simplex via its incident nodes.

    Each simplex value is the unweighted mean of the values at its vertices.

    Parameters
    ----------
    A : np.ndarray (n, d)
        Per-node values.
    T : np.ndarray (t, s)
        Simplex connectivity; ``s`` is the number of vertices per simplex
        (e.g. 3 for triangles, 4 for tets).

    Returns
    -------
    At : np.ndarray (t, d)
        Per-simplex averaged values.
    """
    At = np.zeros((T.shape[0], A.shape[1]))
    # Accumulate each corner's contribution, dividing by the corner count so
    # the result is a plain mean over the simplex's vertices.
    for corner in range(T.shape[1]):
        At += A[T[:, corner], :] / T.shape[1]
    return At