"""Shortest-path compliance distances on a compliance graph (Faure et al. 2011)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.csgraph as csgraph


def compliance_distances(
    G: sp.sparse.spmatrix, sources: Sequence[int] | int
) -> np.ndarray:
    """Compliance distance from each source vertex to every vertex.

    Runs Dijkstra on the edge graph produced by :func:`simkit.compliance_graph`,
    so the result is the paper's compliance distance: the cost of the
    stiffest path between two points, with each step priced at
    ``compliance x length``.

    Parameters
    ----------
    G : scipy.sparse matrix (n, n)
        Compliance-edge graph from :func:`simkit.compliance_graph`.
    sources : int or sequence of int
        Source vertex indices.

    Returns
    -------
    D : np.ndarray (len(sources), n)
        ``D[a, v]`` is the compliance distance from ``sources[a]`` to vertex
        ``v`` (``inf`` if unreachable). Always two-dimensional, even for a
        single source.
    """
    sources = np.atleast_1d(np.asarray(sources, dtype=int))
    D = csgraph.dijkstra(G, directed=False, indices=sources)
    return np.atleast_2d(D)
