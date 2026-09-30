import numpy as np


def _cylinder(p, r, h0, h1):
    q = np.stack([np.hypot(p[:, 1], p[:, 2]) - r, np.abs(p[:, 0] - 0.5 * (h0 + h1)) - 0.5 * (h1 - h0)], 1)
    return np.minimum(q.max(1), 0.0) + np.linalg.norm(np.maximum(q, 0.0), axis=1)


def cup_sdf(P, center, radius, wall, base, length, R=np.eye(3), eps=1e-7):
    """Signed distance (m,) and gradient (m, 3) of an open cup: a cylinder along its local x
    axis (world axis ``R[:, 0]``), open at +x, with a base of thickness ``base`` at -x.
    Solid minus cavity of two exact capped cylinders; gradient by central differences."""
    def phi(Q):
        p = (Q - np.asarray(center, float)) @ R
        return np.maximum(_cylinder(p, radius, -length / 2, length / 2),
                          -_cylinder(p, radius - wall, base - length / 2, length / 2 + 1.0))
    P = np.atleast_2d(P)
    grad = np.stack([(phi(P + e) - phi(P - e)) / (2 * eps) for e in eps * np.eye(3)], 1)
    return phi(P), grad
