"""Volume-weighted cubature points from spectral clustering of mesh modes.

Averages a per-vertex spectral basis onto simplices, clusters in that
simplex-averaged space, and picks representative vertices weighted by
element volume.
"""

from typing import Optional, Tuple, Union

import numpy as np

from .average_onto_simplex import average_onto_simplex
from .pairwise_distance import pairwise_distance
from .spectral_clustering import spectral_clustering
from .volume import volume


def spectral_cubature(
    X: np.ndarray,
    T: np.ndarray,
    W: np.ndarray,
    k: int,
    return_labels: bool = False,
    return_centroids: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """Select cubature vertices via spectral clustering on simplex-averaged modes.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Rest vertex positions.
    T : np.ndarray (t, s)
        Simplex connectivity.
    W : np.ndarray (n, p)
        Per-vertex spectral basis.
    k : int
        Number of cubature points / clusters.
    return_labels : bool, optional
        If True, also return per-vertex cluster labels. Default False.
    return_centroids : bool, optional
        If True, also return cluster centroids in spectral space. Default False.

    Returns
    -------
    lI : np.ndarray (k,)
        Index of the cubature vertex chosen for each cluster.
    mc : np.ndarray (k,)
        Total simplex volume per cluster.
    labels : np.ndarray (n,), optional
        Cluster label per vertex. Returned only if ``return_labels`` is True.
    centroids : np.ndarray (k, p), optional
        Cluster centroids. Returned only if ``return_centroids`` is True.
    """
    Wt = average_onto_simplex(W, T)

    [labels, centroids] = spectral_clustering(Wt, k)
    D = pairwise_distance(centroids, Wt)
    lI = np.argmin(D, axis=1)
    m = volume(X, T)
    mc = np.bincount(labels, m.flatten())

    ret = (lI, mc)

    if return_labels:
        ret = ret + (labels,)
    if return_centroids:
        ret = ret + (centroids,)

    return ret
