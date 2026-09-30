"""Embed points in a mesh as affine combinations of nearby vertices.

Each point ``q_j`` is written as ``sum_i w_ji x_i`` over a small set of candidate
vertices ``i`` (for example, the vertices of the rigid part it is attached to),
with weights that reproduce affine functions exactly::

    sum_i w_ji = 1,        sum_i w_ji X_i = q_j.

The embedded point therefore follows any affine (in particular any rigid) motion
of its candidate vertices exactly, which is what attaching a spring, a pin or a
tendon to a stiff block needs when no mesh vertex sits at the attachment point.
"""

from __future__ import annotations

import numpy as np
import scipy as sp


def affine_embedding_matrix(X: np.ndarray, points: np.ndarray, candidates: list, k: int = 64) -> sp.sparse.csr_matrix:
    """Sparse operator mapping stacked vertex positions to stacked embedded points.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Rest vertex positions.
    points : np.ndarray (m, dim)
        Points to embed, at rest.
    candidates : list of np.ndarray
        ``candidates[j]`` holds the vertex indices point ``j`` may be embedded in.
        The ``k`` of them nearest to ``points[j]`` are used; they must not all be
        coplanar (a degenerate set raises ``LinAlgError``).
    k : int, optional
        Number of nearest candidate vertices per point.

    Returns
    -------
    G : scipy.sparse.csr_matrix (m*dim, n*dim)
        Interleaved (row ``dim*j + axis``, column ``dim*i + axis``) embedding
        operator: ``(G @ X.reshape(-1)).reshape(m, dim) == points``. Among all
        weights satisfying the affine constraints it uses the minimum-norm ones.
    """
    X = np.asarray(X, float)
    points = np.atleast_2d(np.asarray(points, float))
    n, dim = X.shape
    rows, cols, vals = [], [], []
    for j, (q, cand) in enumerate(zip(points, candidates)):
        cand = np.asarray(cand)
        ids = cand[np.argsort(np.linalg.norm(X[cand] - q, axis=1))[:k]]
        c = X[ids].mean(0)
        A = np.vstack([(X[ids] - c).T, np.ones(len(ids))])
        w = A.T @ np.linalg.solve(A @ A.T, np.append(q - c, 1.0))
        for d in range(dim):
            rows += [dim * j + d] * len(ids)
            cols += list(dim * ids + d)
            vals += list(w)
    return sp.sparse.csr_matrix((vals, (rows, cols)), shape=(dim * len(points), dim * n))
