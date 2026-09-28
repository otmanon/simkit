"""Pose the tet mesh with forward kinematics and solve for a cup grasp.

``pose_vertices(scene, meta, q)`` rigidly moves every vertex with its link's
FK transform (relative to the flat meshing pose ``q = 0``). This is the
actuation used by the simulation: joint angles -> link motion.

``fit_grasp`` picks, per finger, the joint angles that wrap the finger around
the cup: rubber tips pressed ``squeeze`` into the cup wall, no aluminium
penetration, and the links drawn towards the cup surface (a power grasp),
within the Allegro joint limits.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.optimize import minimize

from allegro_kinematics import AllegroHand
from hand_geometry import XML, BASE, FINGERS, CupParamsTall, CUP_AXIS_XZ, CUP_Y_BOTTOM

HERE = os.path.dirname(os.path.abspath(__file__))


def vertex_bodies(scene):
    vb = np.full(len(scene["X"]), -1)
    vb[scene["T"].ravel()] = np.repeat(scene["body"], 4)
    return vb


def body_transforms(hand, body_names, q):
    T0 = hand.fk({}, BASE)
    Tq = hand.fk(q, BASE)
    out = []
    for b in body_names:
        if b == "cup":
            out.append(np.eye(4))
        else:
            out.append(Tq[b] @ np.linalg.inv(T0[b]))
    return np.array(out)


def pose_vertices(X, vb, hand, body_names, q):
    A = body_transforms(hand, body_names, q)
    R, t = A[vb, :3, :3], A[vb, :3, 3]
    return np.einsum("nij,nj->ni", R, X) + t


def cup_sdf(P, cp=CupParamsTall(), axis_xz=CUP_AXIS_XZ, y0=CUP_Y_BOTTOM):
    """Signed distance (approx.) to the cup's outer surface; <0 inside the wall."""
    y = np.clip(P[:, 1] - y0, 0.0, cp.height)
    r = cp.radius_bottom + (cp.radius_top - cp.radius_bottom) * y / cp.height
    rho = np.hypot(P[:, 0] - axis_xz[0], P[:, 2] - axis_xz[1])
    d_side = rho - r
    d_cap = np.maximum(y0 - P[:, 1], P[:, 1] - (y0 + cp.height))
    return np.maximum(d_side, d_cap)


FINGER_JOINTS = {f: [f"{f}j{i}" for i in range(4)] for f in FINGERS}


def fit_grasp(scene, meta, squeeze=1.5e-3, n_samples=400, seed=0):
    hand = AllegroHand(XML)
    body_names = list(scene["body_names"])
    vb = vertex_bodies(scene)
    rng = np.random.default_rng(seed)
    part_of_v = np.full(len(scene["X"]), -1)
    part_of_v[scene["T"].ravel()] = np.repeat(scene["part"], 4)
    rubber = np.array(["_tip_rubber" in n for n in scene["part_names"]])[part_of_v.clip(0)]
    q = {}
    for f in FINGERS:
        joints = FINGER_JOINTS[f]
        chain = [b for b in body_names if b.startswith(f + "_")]
        idx = np.where(np.isin(vb, [body_names.index(b) for b in chain]))[0]
        idx = rng.choice(idx, min(len(idx), n_samples * len(chain)), replace=False)
        X = scene["X"][idx]
        is_rub = rubber[idx]
        lo = np.array([meta["joints"][j]["range"][0] for j in joints])
        hi = np.array([meta["joints"][j]["range"][1] for j in joints])

        def cost(a):
            qq = dict(zip(joints, a))
            P = pose_vertices(X, vb[idx], hand, body_names, qq)
            d = cup_sdf(P)
            metal_pen = np.minimum(d[~is_rub] - 2e-4, 0.0)
            tip = d[is_rub].min()
            wrap = np.maximum(d[~is_rub], 0.0)
            return (1e6 * (metal_pen ** 2).sum() + 1e5 * (tip + squeeze) ** 2
                    + 10.0 * np.sort(wrap)[: len(wrap) // 4].mean())

        best = None
        for trial in range(12):
            a0 = lo + (hi - lo) * rng.uniform(0.1, 0.7, len(lo))
            if f != "th":
                a0[0] = 0.0
            r = minimize(cost, a0, method="Powell", bounds=list(zip(lo, hi)),
                         options=dict(maxiter=3000, xtol=1e-4, ftol=1e-10))
            if best is None or r.fun < best.fun:
                best = r
        q.update(dict(zip(joints, best.x.tolist())))
        print(f"{f}: cost={best.fun:.3e} q={np.round(best.x, 3)}", flush=True)
    return q


if __name__ == "__main__":
    out = os.path.join(HERE, "output")
    scene = dict(np.load(os.path.join(out, "scene_tets.npz")))
    meta = json.load(open(os.path.join(out, "hand_meta.json")))
    q = fit_grasp(scene, meta)
    meta["grasp_q"] = q
    json.dump(meta, open(os.path.join(out, "hand_meta.json"), "w"), indent=1)
