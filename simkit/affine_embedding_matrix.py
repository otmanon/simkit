import numpy as np
import scipy as sp


def affine_embedding_matrix(X, points, candidates, k=64):
    """Sparse ``G`` (m*dim, n*dim) with ``G @ X.ravel() == points.ravel()``: each point is the
    minimum-norm affine combination of its ``k`` nearest ``candidates[j]`` vertices, so it
    follows any affine (hence rigid) motion of them exactly."""
    n, dim = X.shape
    rows, cols, vals = [], [], []
    for j, (q, cand) in enumerate(zip(np.atleast_2d(points), candidates)):
        ids = np.asarray(cand)[np.argsort(np.linalg.norm(X[cand] - q, axis=1))[:k]]
        c = X[ids].mean(0)
        A = np.vstack([(X[ids] - c).T, np.ones(len(ids))])
        w = A.T @ np.linalg.solve(A @ A.T, np.append(q - c, 1.0))
        for d in range(dim):
            rows += [dim * j + d] * len(ids)
            cols += list(dim * ids + d)
            vals += list(w)
    return sp.sparse.csr_matrix((vals, (rows, cols)), shape=(dim * len(candidates), dim * n))
