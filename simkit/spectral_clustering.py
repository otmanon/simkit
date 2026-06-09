"""K-means clustering in a weighted spectral basis."""

from typing import Optional, Tuple

import numpy as np
from scipy.cluster.vq import kmeans2


def spectral_clustering(
    W: np.ndarray,
    k: int,
    D: Optional[np.ndarray] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster rows of a spectral basis with k-means after per-row weighting.

    Each row of ``W`` is multiplied element-wise by ``D`` before clustering.

    Parameters
    ----------
    W : np.ndarray (n, p)
        Spectral basis (one row per vertex or element).
    k : int
        Number of clusters.
    D : np.ndarray (n, 1), optional
        Per-row weights. Defaults to ones if None.
    seed : int, optional
        Random seed for k-means. Default 0.

    Returns
    -------
    l : np.ndarray (n,)
        Cluster label per row.
    c : np.ndarray (k, p)
        Cluster centroids in the weighted spectral space.
    """
    if D is None:
        D = np.ones((W.shape[0], 1))

    B = W * D

    # scipy's kmeans2 returns (centroids, labels); k-means++ init keeps the
    # result close to (though not identical to) sklearn's KMeans.
    c, l = kmeans2(B, k, seed=seed, minit="++")
    return l, c
