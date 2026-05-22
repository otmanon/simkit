"""Build cluster grouping/averaging operators over a simplicial mesh."""

from typing import Tuple, Union

import numpy as np
import scipy as sp

from .volume import volume


def cluster_grouping_matrices(
    l: np.ndarray,
    V: np.ndarray,
    T: np.ndarray,
    return_mass: bool = False,
) -> Union[
    Tuple["sp.sparse.csc_matrix", "sp.sparse.spmatrix"],
    Tuple["sp.sparse.csc_matrix", "sp.sparse.spmatrix", np.ndarray, np.ndarray, np.ndarray],
]:
    """Grouping and mass-weighted averaging matrices for element clusters.

    ``G`` sums an element-indexed quantity into its cluster; ``Gm`` takes the
    mass-weighted average over each cluster (so applying it to a constant
    returns that constant).

    Parameters
    ----------
    l : np.ndarray (t,)
        Per-element cluster label in ``[0, c)``.
    V : np.ndarray (n, d)
        Vertex positions.
    T : np.ndarray (t, s)
        Simplex connectivity; ``s`` is 3 (triangles) or 4 (tets).
    return_mass : bool, optional
        If True, also return per-cluster and per-element masses. Default False.

    Returns
    -------
    G : scipy.sparse.csc_matrix (c, t)
        Cluster summation matrix (one indicator row per cluster).
    Gm : scipy.sparse.spmatrix (c, t)
        Mass-weighted cluster averaging matrix ``Mci @ G @ Mt``.
    mc : np.ndarray (c,), optional
        Per-cluster mass. Returned only if ``return_mass`` is True.
    mt : np.ndarray (t, 1), optional
        Per-element mass. Returned only if ``return_mass`` is True.
    f : np.ndarray (t,), optional
        Each element's fraction of its cluster's mass. Returned only if
        ``return_mass`` is True.
    """
    t = T.shape[0]
    c = l.max() + 1
    assert T.shape[1] == 4 or T.shape[1] == 3

    # Per-element masses (volumes); keep as a column even for a single element.
    mt = volume(V, T)
    if mt.ndim == 0:
        mt = mt[None]

    # Per-cluster mass is the sum of its elements' masses.
    mc = np.bincount(l, mt[:, 0])
    Mci = sp.sparse.diags(1 / mc, 0)     # inverse cluster mass (for averaging)
    Mt = sp.sparse.diags(mt[:, 0], 0)    # element mass

    # G[cluster, element] = 1 when the element belongs to that cluster.
    I = l
    J = np.arange(t)
    VV = np.ones(t)
    G = sp.sparse.csc_matrix((VV, (I, J)), shape=(c, t))

    # Mass-weighted average: scale elements by mass, sum, divide by cluster mass.
    Gm = Mci @ G @ Mt

    if return_mass:
        f = mt[:, 0] / mc[l]             # element share of its cluster mass
        return G, Gm, mc, mt, f
    return G, Gm