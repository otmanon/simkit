"""Tetrahedralize ``hand.obj`` + ``cup.obj`` and label every tet by winding number.

* fTetWild per shell (one shell per rigid link): the palm (which only moves
  rigidly with the wrist) is coarse, the phalanges and rubber tips are fine.
* For every part OBJ the generalized winding number (libigl) is evaluated at
  every tet centroid; the highest-priority part with ``w > 1/2`` wins
  (shafts > rubber > tip cores > link housings).
* Every tet also records its kinematic body (palm, ff_proximal, ...), which the
  simulation / posing code uses to drive it with forward kinematics.

Output: ``output/scene_tets.npz``.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

import numpy as np
import igl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "robot_gripper"))
from gripper_geometry import read_obj  # noqa: E402
from materials import MATERIALS, part_material, part_priority  # noqa: E402


def interior_point(V, F, rng):
    lo, hi = V.min(0), V.max(0)
    P = rng.uniform(lo, hi, size=(4000, 3))
    w = igl.winding_number(V, F, P)
    P = P[w > 0.5]
    # prefer the point deepest inside (largest distance to the surface)
    d = np.abs(igl.signed_distance(P, V, F)[0])
    return P[np.argmax(d)]


def tetrahedralize_shell(V, F, edge):
    """fTetWild (``wildmeshing``) with target edge length ``edge`` [m].

    TetGen chokes on (or massively over-refines) the Allegro CAD surfaces;
    fTetWild's envelope approach remeshes them robustly. The tet boundary then
    only approximates the CAD within ``epsilon`` -- fine, because materials
    come from winding numbers against the original part meshes anyway.
    """
    import wildmeshing as wm
    diag = np.linalg.norm(V.max(0) - V.min(0))
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:   # fTetWild drops scratch files in cwd
        os.chdir(tmp)
        try:
            tet = wm.Tetrahedralizer(stop_quality=10, edge_length_r=edge / diag,
                                     epsilon=1e-4 / diag, max_its=40)
            tet.set_mesh(V, F)
            tet.tetrahedralize()
            X, T = tet.get_tet_mesh()[:2]
            del tet
        finally:
            os.chdir(cwd)
    return np.asarray(X, float), np.asarray(T, np.int64)


def mesh_hand_shells(Vh, Fh, out_dir, finger_edge, palm_edge, rng):
    """fTetWild each shell separately (the shells are separate bodies anyway,
    and one awkward CAD surface cannot fail the whole run). The palm only moves
    rigidly, so it is coarse."""
    comp = igl.facet_components(Fh)[1]
    palm_V, palm_F = read_obj(os.path.join(out_dir, "parts", "palm.obj"))
    Xs, Ts, nv = [], [], 0
    for k in range(comp.max() + 1):
        Fk = Fh[comp == k]
        used = np.unique(Fk)
        remap = -np.ones(len(Vh), np.int64)
        remap[used] = np.arange(len(used))
        Vk, Fk = Vh[used], remap[Fk]
        p = interior_point(Vk, Fk, rng)
        is_palm = igl.winding_number(palm_V, palm_F, p[None])[0] > 0.5
        Xk, Tk = tetrahedralize_shell(Vk, Fk, palm_edge if is_palm else finger_edge)
        print(f"  shell {k:2d}: {len(Tk):6d} tets", flush=True)
        Xs.append(Xk)
        Ts.append(Tk + nv)
        nv += len(Xk)
    return np.vstack(Xs), np.vstack(Ts)


def build(out_dir, finger_edge=2.5e-3, palm_edge=8e-3, cup_edge=3e-3):
    rng = np.random.default_rng(0)
    meta = json.load(open(os.path.join(out_dir, "hand_meta.json")))
    Vh, Fh = read_obj(os.path.join(out_dir, "hand.obj"))
    Vc, Fc = read_obj(os.path.join(out_dir, "cup.obj"))

    # Each shell (one per rigid link) is meshed on its own; cached on the hash
    # of hand.obj so that cup-only changes re-mesh in seconds.
    key = hashlib.sha1(open(os.path.join(out_dir, "hand.obj"), "rb").read()).hexdigest()
    key += f"-{finger_edge}-{palm_edge}"
    cache = os.path.join(out_dir, "hand_tets_cache.npz")
    if os.path.exists(cache) and str(np.load(cache)["key"]) == key:
        Xh, Th = np.load(cache)["X"], np.load(cache)["T"]
        print("  reusing cached hand tets", flush=True)
    else:
        Xh, Th = mesh_hand_shells(Vh, Fh, out_dir, finger_edge, palm_edge, rng)
        np.savez_compressed(cache, X=Xh, T=Th, key=key)
    Xc, Tc = tetrahedralize_shell(Vc, Fc, cup_edge)

    X = np.vstack([Xh, Xc]).astype(float)
    T = np.vstack([Th, Tc + len(Xh)]).astype(np.int64)
    C = X[T].mean(1)

    names = sorted(meta["parts"], key=lambda n: (part_priority(n), n)) + ["cup"]
    part = np.full(len(T), -1)
    part[len(Th):] = len(names) - 1
    hand_tets = np.arange(len(Th))
    for i, n in enumerate(names[:-1]):
        V, F = read_obj(os.path.join(out_dir, "parts", f"{n}.obj"))
        todo = hand_tets[part[hand_tets] < 0]
        w = igl.winding_number(V, F, C[todo])
        part[todo[w > 0.5]] = i
    # slivers along bonded interfaces: take the majority label of neighbours
    TT = igl.tet_tet_adjacency(T)[0]
    for _ in range(20):
        miss = np.where(part < 0)[0]
        if not len(miss):
            break
        for t in miss:
            nb = part[TT[t][TT[t] >= 0]]
            nb = nb[nb >= 0]
            if len(nb):
                part[t] = np.bincount(nb).argmax()

    bodies = sorted(set(meta["parts"].values())) + ["cup"]
    body_of_part = [bodies.index(meta["parts"].get(n, "cup")) for n in names]
    mats = [part_material(n) for n in names]
    prop = lambda key: np.array([MATERIALS[mats[p]][key] for p in part])
    return dict(X=X, T=T, part=part, part_names=np.array(names),
                material_names=np.array(mats), body=np.array(body_of_part)[part],
                body_names=np.array(bodies), E=prop("E"), nu=prop("nu"),
                rho=prop("rho"), strength=prop("strength"))


if __name__ == "__main__":
    out_dir = os.path.join(HERE, "output")
    out = build(out_dir)
    np.savez_compressed(os.path.join(out_dir, "scene_tets.npz"), **out)
    print(f"{len(out['X'])} vertices, {len(out['T'])} tets")
    for m in np.unique(out["material_names"]):
        k = np.isin(out["part"], np.where(out["material_names"] == m)[0]).sum()
        print(f"  {m:20s} {k:7d} tets")
