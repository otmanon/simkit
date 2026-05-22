"""Equispaced vertices on the unit interval ``[0, 1]``.

Optionally returns edge connectivity for a 1D line mesh (segment simplices).
"""

from __future__ import annotations

import numpy as np


def uniform_line(
    n: int, return_simplex: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Vertices ``k / (n - 1)`` for ``k = 0, ..., n - 1``.

    Parameters
    ----------
    n : int
        Number of vertices (must be at least 2 for a well-defined spacing).
    return_simplex : bool, optional
        If ``True``, also return segment connectivity ``(n - 1, 2)``.

    Returns
    -------
    X : np.ndarray (n, 1)
        Vertex positions along the line, when ``return_simplex`` is ``False``.
    X, T : tuple of np.ndarray
        When ``return_simplex`` is ``True``: positions ``(n, 1)`` and edges
        ``T`` of shape ``(n - 1, 2)`` linking consecutive vertices.
    """
    X = np.arange(0, n, dtype=float)[:, None] / (n - 1)
    ret = X
    if return_simplex:
        T = np.hstack(
            (np.arange(X.shape[0] - 1)[:, None], np.arange(1, X.shape[0])[:, None])
        )
        ret = (ret, )
        ret = ret + (T,)
    return ret
