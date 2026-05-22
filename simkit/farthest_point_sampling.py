"""Farthest-point sampling for point sets."""

from typing import Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances


def farthest_point_sampling(
    V: np.ndarray, d: int, sI: Optional[np.ndarray] = None
) -> np.ndarray:
    """Select ``d`` vertices via iterative farthest-point sampling.

    Starting from a seed index, repeatedly adds the point whose minimum
    distance to the current sample set is largest.

    Parameters
    ----------
    V : np.ndarray (n, dim)
        Point cloud or vertex positions.
    d : int
        Number of samples to select.
    sI : np.ndarray, optional
        Seed index (single element). If ``None``, uses the vertex with
        minimum first coordinate.

    Returns
    -------
    I : np.ndarray (d,)
        Selected vertex indices.
    """
    # pick random first index

    if sI is None:
        idx = np.argmin(V[:, 0])  # np.random.randint(0, V.shape[0])
        I = np.array([idx])
    else:
        I = np.array(sI)
        I = I.reshape(1)
    for i in range(0, d - 1):

        D = pairwise_distances(V, V[I, :])
        idx = np.argmax(np.min(D, axis=1))
        I = np.append(I, idx)

    return I
