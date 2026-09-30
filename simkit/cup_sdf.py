"""Signed distance to an open cylindrical cup."""

from __future__ import annotations

import numpy as np


def _capped_cylinder(p: np.ndarray, r: float, h0: float, h1: float) -> np.ndarray:
    """Exact signed distance to the solid cylinder of radius ``r`` about the local
    x axis, ``h0 <= x <= h1``."""
    q = np.stack([np.hypot(p[:, 1], p[:, 2]) - r, np.abs(p[:, 0] - 0.5 * (h0 + h1)) - 0.5 * (h1 - h0)], 1)
    return np.minimum(q.max(1), 0.0) + np.linalg.norm(np.maximum(q, 0.0), axis=1)


def cup_sdf(P: np.ndarray, center: np.ndarray, radius: float, wall: float, base: float, length: float,
            R: np.ndarray | None = None, eps: float = 1e-7) -> tuple[np.ndarray, np.ndarray]:
    """Signed distance to an open cup and its gradient.

    The cup is a cylinder of outer radius ``radius`` and length ``length`` along its
    local x axis, centred at ``center``, with a cavity of radius ``radius - wall``
    open at local ``+x`` and a solid base of thickness ``base`` at ``-x``. It is
    evaluated as ``max(phi_outer, -phi_cavity)`` (solid minus cavity) of two exact
    capped-cylinder distances.

    Parameters
    ----------
    P : np.ndarray (m, 3)
        Query points.
    center : np.ndarray (3,)
        Cup centre.
    radius, wall, base, length : float
        Outer radius, wall thickness, base thickness and length.
    R : np.ndarray (3, 3), optional
        Rotation from the local frame to the world (the cup axis is ``R[:, 0]``).
        Identity by default.
    eps : float, optional
        Central-difference step for the gradient.

    Returns
    -------
    phi : np.ndarray (m,)
        Signed distance, negative inside the cup material.
    grad : np.ndarray (m, 3)
        Gradient of ``phi`` (unit length almost everywhere), by central
        differences of the analytic distance.
    """
    R = np.eye(3) if R is None else np.asarray(R, float)
    c = np.asarray(center, float)

    def phi(Q):
        p = (Q - c) @ R
        outer = _capped_cylinder(p, radius, -0.5 * length, 0.5 * length)
        cavity = _capped_cylinder(p, radius - wall, -0.5 * length + base, 0.5 * length + 1.0)
        return np.maximum(outer, -cavity)

    P = np.atleast_2d(np.asarray(P, float))
    grad = np.empty_like(P)
    for i in range(3):
        e = np.zeros(3)
        e[i] = eps
        grad[:, i] = (phi(P + e) - phi(P - e)) / (2 * eps)
    return phi(P), grad
