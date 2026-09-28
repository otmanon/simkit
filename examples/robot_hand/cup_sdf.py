"""Analytic signed distance function of the tapered thin-walled cup.

Matches ``gripper_geometry.build_cup`` exactly (cup local frame: axis +y, base
on y = 0): an outer capped cone, minus the inner capped cone (the hollow, open
at the top, closed by the base), plus the rolled rim torus.

    solid = max(sd_outer, -sd_inner)   then   min(solid, sd_rim)

Capped-cone distance: Inigo Quilez, "distance functions" (exact for a cone
frustum). ``sdf_and_grad`` returns the gradient by central differences
(vectorised, 6 extra evaluations), which is plenty for a Gauss-Newton contact
Hessian.
"""
from __future__ import annotations

import numpy as np


def _dot2(v):
    return (v * v).sum(-1)


def sd_capped_cone(p, y0, y1, r0, r1):
    """Exact SDF of a cone frustum along +y from (y0, radius r0) to (y1, r1)."""
    h = 0.5 * (y1 - y0)
    q = np.stack([np.hypot(p[:, 0], p[:, 2]), p[:, 1] - (y0 + h)], -1)
    k1 = np.array([r1, h])
    k2 = np.array([r1 - r0, 2.0 * h])
    rr = np.where(q[:, 1] < 0.0, r0, r1)
    ca = np.stack([q[:, 0] - np.minimum(q[:, 0], rr), np.abs(q[:, 1]) - h], -1)
    t = np.clip(((k1 - q) * k2).sum(-1) / _dot2(k2), 0.0, 1.0)
    cb = q - k1 + k2 * t[:, None]
    s = np.where((cb[:, 0] < 0.0) & (ca[:, 1] < 0.0), -1.0, 1.0)
    return s * np.sqrt(np.minimum(_dot2(ca), _dot2(cb)))


def sd_torus_y(p, R, r, yc):
    return np.hypot(np.hypot(p[:, 0], p[:, 2]) - R, p[:, 1] - yc) - r


class CupSDF:
    def __init__(self, cp):
        rb, rt, H, t = cp.radius_bottom, cp.radius_top, cp.height, cp.wall
        slope = (rt - rb) / H
        self.outer = (0.0, H, rb, rt)
        # the hollow: same construction as build_cup (extended past the rim so
        # the top is open)
        b = cp.base + 1e-4
        self.inner = (b, H + 0.05, rb - t + slope * cp.base, rt - t + 1e-4 + slope * 0.05)
        self.rim = (rt - 0.5 * t, cp.rim_radius, H - 0.6 * cp.rim_radius)
        self.H = H

    def __call__(self, P):
        P = np.atleast_2d(P)
        solid = np.maximum(sd_capped_cone(P, *self.outer), -sd_capped_cone(P, *self.inner))
        return np.minimum(solid, sd_torus_y(P, *self.rim))

    def sdf_and_grad(self, P, eps=1e-6):
        P = np.atleast_2d(P)
        d = self(P)
        g = np.empty_like(P)
        for k in range(3):
            e = np.zeros(3)
            e[k] = eps
            g[:, k] = (self(P + e) - self(P - e)) / (2 * eps)
        n = np.linalg.norm(g, axis=1, keepdims=True)
        return d, g / np.maximum(n, 1e-12)
