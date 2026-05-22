"""Impulse-response vibe weights from skinning modes and spectral clustering.

Builds a damped structural response matrix, clusters vertices in mode space,
and normalizes per-vertex impulse weights at cluster representatives.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy as sp

from .dirichlet_laplacian import dirichlet_laplacian
from .massmatrix import massmatrix
from .pairwise_distance import pairwise_distance
from .skinning_eigenmodes import skinning_eigenmodes
from .spectral_clustering import spectral_clustering


def random_impulse_vibes(
    X: np.ndarray,
    T: np.ndarray,
    m: int,
    h: float = 1e-2,
    ord: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute normalized impulse vibe weights and cluster representatives.

    Solves ``(L + M/h^2) G = F`` with impulse forces at spectral-cluster
    centers, then raises and renormalizes weights per vertex.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Rest vertex positions.
    T : np.ndarray (nt, simplex_size)
        Mesh simplices (triangles or tets).
    m : int
        Number of skinning eigenmodes / clusters.
    h : float, optional
        Timestep scaling in the damped operator ``L + M/h^2``.
    ord : int, optional
        Exponent applied before row-wise normalization of ``G``.

    Returns
    -------
    G : np.ndarray (n, m)
        Normalized per-vertex weights (floored, raised to ``ord``, summed to 1).
    cI : np.ndarray (n,)
        Index of the nearest cluster center for each vertex.
    G_full : np.ndarray (n, m)
        Raw solved response before flooring and normalization.
    """
    L = dirichlet_laplacian(X, T)
    M = massmatrix(X, T)
    H = L + M / h**2

    Mi = sp.sparse.diags(1 / M.diagonal()).tocsc().toarray()

    W, _E, _B = skinning_eigenmodes(X, T, m)
    _l, c = spectral_clustering(W, m)
    D = pairwise_distance(c, W)
    cI = D.argmin(axis=1)

    F = Mi[:, cI]
    G_full = sp.sparse.linalg.spsolve(H, F).reshape(F.shape)
    G_full = G_full / G_full.max(axis=1)[:, None]
    G = G_full.copy()

    G[G < 1e-8] = 1e-8
    G = G**ord / (G**ord).sum(axis=1)[:, None]

    return G, cI, G_full
