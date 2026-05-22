"""Sparse map from simplex corner indices to global vertex indices."""

from typing import Optional

import numpy as np
import scipy as sp


def simplex_vertex_map(
    T: np.ndarray,
    nv: Optional[int] = None,
) -> sp.sparse.csc_matrix:
    """Build a sparse matrix mapping simplex corners to vertex DOFs.

    Each row corresponds to one corner of one simplex; column ``j`` is set
    when corner index ``j`` appears in ``T``.

    Parameters
    ----------
    T : np.ndarray (nt, dt) or (*, dt)
        Simplex vertex indices (triangles, tets, etc.).
    nv : int, optional
        Number of vertices. Defaults to ``T.max() + 1``.

    Returns
    -------
    S : scipy.sparse.csc_matrix (dt*nt, nv)
        Corner-to-vertex incidence map.
    """
    dt = T.shape[-1]
    T = T.reshape(-1, dt)
    nt = T.shape[0]

    if nv is None:
        nv = T.max() + 1

    J = T
    I = np.repeat(np.arange(dt)[None, :], nt, axis=0) + np.arange(nt)[:, None] * dt
    V = np.ones(J.shape)
    S = sp.sparse.csc_matrix(
        (V.flatten(), (I.flatten(), J.flatten())),
        (dt * nt, nv),
    )

    return S
