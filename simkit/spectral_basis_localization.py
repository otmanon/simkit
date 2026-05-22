"""Localized harmonic/biharmonic weights from spectral mesh clustering.

Clusters vertices in a weighted spectral basis, selects the vertex nearest
each cluster centroid, and builds localized coordinate weights (harmonic or
biharmonic) on those handles.
"""

from typing import Optional, Tuple, Union

import numpy as np

from .average_onto_simplex import average_onto_simplex
from .biharmonic_coordinates import biharmonic_coordinates
from .harmonic_coordinates import harmonic_coordinates
from .pairwise_distance import pairwise_distance
from .skinning_eigenmodes import skinning_eigenmodes
from .spectral_clustering import spectral_clustering


def spectral_basis_localization(
    X: np.ndarray,
    T: np.ndarray,
    m: int,
    W: Optional[np.ndarray] = None,
    order: int = 2,
    return_clustering_info: bool = False,
    threshold: float = 0,
) -> Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """Build localized weights from spectral clustering of a mesh basis.

    If ``W`` is omitted, skinning eigenmodes are computed first. Cluster
    centroids in spectral space are mapped to nearest mesh vertices; harmonic
    (``order == 1``) or biharmonic (``order == 2``) coordinates on those
    handles yield the localization weights.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Rest vertex positions.
    T : np.ndarray (t, s)
        Simplex connectivity.
    m : int
        Number of spectral modes / clusters.
    W : np.ndarray (n, m), optional
        Per-vertex spectral basis. Computed via skinning eigenmodes if None.
    order : int, optional
        Coordinate order: ``1`` harmonic, ``2`` biharmonic. Default 2.
    return_clustering_info : bool, optional
        If True, also return cluster labels and centroids. Default False.
    threshold : float, optional
        Zero out weights with magnitude below this value. Default 0.

    Returns
    -------
    Wh : np.ndarray (n, m)
        Localized coordinate weights per vertex.
    cI : np.ndarray (m,)
        Index of the handle vertex chosen for each cluster.
    l : np.ndarray (n,), optional
        Cluster label per vertex. Returned only if ``return_clustering_info``
        is True.
    c : np.ndarray (m, p)
        Cluster centroids in spectral space. Returned only if
        ``return_clustering_info`` is True.
    """
    if W is None:
        [W, _E, _B] = skinning_eigenmodes(X, T, m)

    [l, c] = spectral_clustering(W, m)

    D = pairwise_distance(c, W)
    cI = D.argmin(axis=1)

    Wh = None
    if order == 1:
        Wh = harmonic_coordinates(X, T, cI)
    elif order == 2:
        Wh = biharmonic_coordinates(X, T, cI)

    if threshold > 0:
        Wh[np.abs(Wh) < threshold] = 0

    out = (Wh, cI)
    if return_clustering_info:
        out = out + (l, c)
    return out
