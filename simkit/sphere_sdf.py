import numpy as np


def sphere_sdf(P, center, radius):
    """Signed distance ``|p - c| - r`` of points ``P`` (m, dim) and its gradient (m, dim)."""
    r = np.atleast_2d(P) - np.asarray(center, float)
    dist = np.linalg.norm(r, axis=1)
    return dist - radius, r / np.maximum(dist, 1e-300)[:, None]
