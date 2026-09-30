"""Signed distance to a sphere."""

from __future__ import annotations

import numpy as np


def sphere_sdf(P: np.ndarray, center: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Signed distance ``|p - c| - r`` and its gradient.

    Parameters
    ----------
    P : np.ndarray (m, dim)
        Query points.
    center : np.ndarray (dim,)
        Sphere centre.
    radius : float
        Sphere radius.

    Returns
    -------
    phi : np.ndarray (m,)
        Signed distance, negative inside.
    grad : np.ndarray (m, dim)
        Unit outward normal ``(p - c) / |p - c|``.
    """
    r = np.atleast_2d(P) - np.asarray(center, float)
    dist = np.linalg.norm(r, axis=1)
    return dist - radius, r / np.maximum(dist, 1e-300)[:, None]
