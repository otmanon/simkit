"""Tetrahedralize the unified gripper + cup OBJs and label materials by winding number.

Pipeline
--------
1. TetGen (constrained Delaunay, quality ``-pq``) meshes ``gripper.obj`` and
   ``cup.obj`` independently; the two tet meshes are concatenated.
2. For every part OBJ in ``parts/`` we evaluate libigl's generalized winding
   number ``w_part(c)`` at every tet centroid ``c``. A tet belongs to the first
   part (in ``materials.PART_MATERIAL`` priority order) with ``w > 1/2``.
3. Per-tet Young's modulus, Poisson ratio, density and strength follow from the
   part -> material table.

Output: ``scene_tets.npz`` with ``X (n,3)``, ``T (t,4)``, ``part (t,)`` (index
into ``part_names``), ``E, nu, rho, strength (t,)`` and ``body (t,)``
(0 = gripper, 1 = cup).
"""
from __future__ import annotations

import os
import numpy as np
import igl
import tetgen

from gripper_geometry import read_obj
from materials import MATERIALS, PART_MATERIAL


def tetrahedralize(V, F, max_volume, min_ratio=1.5, region_max_volume=()):
    """TetGen ``-pq{min_ratio}a{max_volume}``; ``region_max_volume`` is a list of
    ``(point_inside_shell, max_vol)`` overriding the volume bound per shell."""
    tg = tetgen.TetGen(V, F.astype(np.int32))
    for i, (pt, mv) in enumerate(region_max_volume):
        tg.add_region(i + 1, pt, mv)
    X, T = tg.tetrahedralize(order=1, quality=True, minratio=min_ratio,
                             maxvolume=max_volume, fixedvolume=True,
                             varvolume=len(region_max_volume) > 0,
                             regionattrib=len(region_max_volume) > 0,
                             verbose=0)[:2]
    return np.asarray(X, float), np.asarray(T, np.int64)


def label_by_winding_number(X, T, part_meshes):
    """Return per-tet part index using generalized winding numbers."""
    C = X[T].mean(axis=1)
    part = np.full(T.shape[0], -1, dtype=np.int64)
    for i, (V, F) in enumerate(part_meshes):
        w = igl.winding_number(V, F, C)
        part[(part < 0) & (w > 0.5)] = i
    return part


def build(data_dir, jaw_max_vol=6.0e-9, housing_max_vol=2.0e-7, cup_max_vol=1.0e-8):
    Vg, Fg = read_obj(os.path.join(data_dir, "gripper.obj"))
    Vc, Fc = read_obj(os.path.join(data_dir, "cup.obj"))
    # The housing only ever moves rigidly, so it gets coarse tets (the global
    # bound); each jaw shell (carriage + finger + pad) gets a finer region bound
    # so the soft pads can conform to the cup.
    jaw_pts = [[s * 0.053, 0.05, 0.0] for s in (-1.0, 1.0)]
    Xg, Tg = tetrahedralize(Vg, Fg, housing_max_vol,
                            region_max_volume=[(p, jaw_max_vol) for p in jaw_pts])
    Xc, Tc = tetrahedralize(Vc, Fc, cup_max_vol)

    X = np.vstack([Xg, Xc])
    T = np.vstack([Tg, Tc + Xg.shape[0]])
    body = np.r_[np.zeros(len(Tg), int), np.ones(len(Tc), int)]

    part_names = list(PART_MATERIAL)
    meshes = []
    for name in part_names:
        path = os.path.join(data_dir, "parts", f"{name}.obj")
        meshes.append(read_obj(path) if name != "cup" else (Vc, Fc))
    part = label_by_winding_number(X, T, meshes)
    # A tet that no part claims (possible only for slivers along a fused
    # interface) takes the label of the part claiming most of its neighbours.
    if (part < 0).any():
        TT = igl.tet_tet_adjacency(T)[0]
        for _ in range(10):
            miss = np.where(part < 0)[0]
            if len(miss) == 0:
                break
            for t in miss:
                nb = part[TT[t][TT[t] >= 0]]
                nb = nb[nb >= 0]
                if len(nb):
                    part[t] = np.bincount(nb).argmax()

    mats = [MATERIALS[PART_MATERIAL[n]] for n in part_names]
    E = np.array([mats[p]["E"] for p in part])
    nu = np.array([mats[p]["nu"] for p in part])
    rho = np.array([mats[p]["rho"] for p in part])
    strength = np.array([mats[p]["strength"] for p in part])
    return dict(X=X, T=T, part=part, part_names=np.array(part_names),
                material_names=np.array([PART_MATERIAL[n] for n in part_names]),
                E=E, nu=nu, rho=rho, strength=strength, body=body)


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    out = build(data_dir)
    np.savez_compressed(os.path.join(data_dir, "scene_tets.npz"), **out)
    print(f"{len(out['X'])} vertices, {len(out['T'])} tets")
    for i, n in enumerate(out["part_names"]):
        k = (out["part"] == i).sum()
        print(f"  {n:15s} {out['material_names'][i]:20s} {k:7d} tets")
